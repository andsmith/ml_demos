"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""

from policy_eval import PolicyEvalDemoAlg
from collections import OrderedDict
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import numpy as np
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT, NEON_BLUE, NEON_GREEN, NEON_RED
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE, TITLE_INDENT
import logging
import time
from threading import Thread, Event, get_ident
import cv2
from abc import ABC, abstractmethod

ITEM_INDENT = 20
#from loop_timing.loop_profiler import LoopPerfTimer as LPT


class AlgDepPanel(Panel, ABC):
    """
    Panels that require updating from the algorithm.
    """

    def __init__(self, app, alg, bbox_rel, margin_rel=0.0):
        super().__init__(app, bbox_rel, margin_rel=margin_rel)
        self.change_algorithm(alg)


    @abstractmethod
    def refresh_images(self, is_paused):
        """
        Get new image from the app, set the image for the current tab.
        """
        pass


class StatePanel(AlgDepPanel):
    """
    Largest, taking up the entire right half, displaying all the game states or their values, etc.
    Tabbed interface, queries algorithm for details.
    """

    def __init__(self, app, alg, bbox_rel,margin_rel=0.0):
        """
        :param app: The application object.
        :param bbox_rel: The bounding box for the panel, relative to the parent frame.
        :param tab_info:  dict with one key/value pair per tab:
            * key:  name of tab
            * value:    display name of tab

        :param resize_callback:  function to call to resize the panel when the tab changes.
            Whoever makes the tab images needs to know the new size, etc.
        """
        self._state_tabs = {}
        self._state_image_size = None
        super().__init__(app, alg, bbox_rel, margin_rel=margin_rel)

    def change_algorithm(self, alg):
        """
        Changing to a new algorithm, from a load or user change + reset.
        Also need to: 
          1. clear the old tabs & set the new tabs.
          2. Refresh the stat

        """
        super().change_algorithm(alg)
        tab_info = alg.get_state_tab_info()
        if tab_info is not None:
            self.set_tabs(tab_info)

    def _init_widgets(self):
        """
        Initialize the widgets for the state panel.
        """
        # Create a notebook for the tabs:
        self._notebook = ttk.Notebook(self._frame, style='TNotebook')

        style = ttk.Style(self._frame)
        # Override default font for tabs
        style.configure('TNotebook.Tab', font=LAYOUT['fonts']['tabs'])

        # Set callback for the tab:
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._notebook.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Set the default tab to the first one:
        self.cur_tab = None

    def set_tabs(self, tab_info):
        """
        Set the tabs for the state panel.
        :param tab_info: (see __init__)
        :param resize_callback: function to call to resize the panel when the tab changes.
           Whoever makes the tab imges needs to know the new size, etc.
        """
        self._tabs_by_text = {tab_txt: tab_name for tab_name, tab_txt in tab_info.items()}
        # Clear old tabs:
        for tab, img_label in self._state_tabs.values():
            tab.destroy()
            img_label.destroy()
        self._state_tabs = {}
        # Add new tabs:
        for tab_name, tab_str in tab_info.items():
            # Create a new frame for the tab:
            tab_frame = tk.Frame(self._notebook, bg=self._bg_color)
            tab_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Create a label for the tab to hold the image (filling the whole tab frame):
            img_label = tk.Label(tab_frame, bg=self._bg_color)
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Add the tab to the notebook:
            self._notebook.add(tab_frame, text=tab_str)
            # Add the tab to the state tabs dictionary:
            self._state_tabs[tab_name] = (tab_frame, img_label)

        # get the current tab:
        self.cur_tab = self._tabs_by_text[self._notebook.tab(self._notebook.select(), "text")]
        print("Setting current tab to %s" % self.cur_tab)

    def get_frame_size(self):
        frame_size = self._frame.winfo_width(), self._frame.winfo_height()
        return frame_size

    def _on_resize(self, event):
        # if self._resize_callback is not None:
        #    self._resize_callback(event)
        # Get the new size of the tap images (use the current tab):
        tab_frame = self._state_tabs[self.cur_tab][0]
        tab_frame.update_idletasks()
        super()._on_resize(event)
        self._state_image_size = tab_frame.winfo_width(), tab_frame.winfo_height()
        # self._blank = np.zeros((self._state_image_size[1], self._state_image_size[0], 3), dtype=np.uint8)
        # self._blank[:] = self._bg_color_rgb
        self.refresh_images(is_paused=self._alg.paused)

    #@LPT.time_function
    def refresh_images(self, is_paused):
        """
        Get new image from the app, set the image for the current tab.
        """
        if self._state_image_size is None:
            return
        new_img = self._alg.get_state_image(self._state_image_size, self.cur_tab, is_paused=is_paused)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        label = self._state_tabs[self.cur_tab][1]
        label.config(image=new_img)
        label.image = new_img
        #label.update_idletasks()

    def _on_tab_changed(self, event):
        """
        """
        logging.info("Tab changed to %s" % self._notebook.tab(self._notebook.select(), "text"))

        # get the current tab:
        self.cur_tab = self._tabs_by_text[self._notebook.tab(self._notebook.select(), "text")]
        self.refresh_images(is_paused=self._alg.paused)


class VisualizationPanel(AlgDepPanel):
    """
    Visualization panel for the algorithm (bottom left).

    Depending on the current run-mode:
      - in continuous-run mode, progress bars, histograms, etc.
      - while paused, show the current state of the algorithm.

    This panel will display the current state of the algorithm, including the current state, action, and value function.
    It will also display the current policy and the current value function.
    """

    def __init__(self, app, alg,  bbox_rel, margin_rel=0.0):
        self._state_image_size = None
        self._state_images = {}
        super().__init__(app, alg, bbox_rel, margin_rel=margin_rel)

    def _init_widgets(self):
        # no widgets, just a label for the viz image
        self._viz_label = tk.Label(self._frame, bg=self._bg_color)
        self._viz_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_resize(self, event):
        self._state_image_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_images(is_paused=self._alg.paused, control_point = self._alg.current_ctrl_pt)


    #@LPT.time_function
    def refresh_images(self, is_paused, control_point):
        """
        Get new image from the app, set the image for the current tab.
        """
        if self._state_image_size is None:
            return
        new_img = self._alg.get_viz_image(self._state_image_size,
                                          control_point=control_point,
                                          is_paused=is_paused)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        self._viz_label.config(image=new_img)
        self._viz_label.image = new_img
        #self._viz_label.update_idletasks()
