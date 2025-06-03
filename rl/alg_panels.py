"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""

from policy_eval import PolicyEvalDemoAlg
from collections import OrderedDict
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import numpy as np
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE, TITLE_INDENT
import logging
import time
from threading import Thread, Event, get_ident
import cv2
from abc import ABC, abstractmethod

ITEM_INDENT = 20
# from loop_timing.loop_profiler import LoopPerfTimer as LPT


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
    
class TabPanel(AlgDepPanel):
    """
    Largest, taking up the entire right half, displaying all the game states or their values, results, etc.
    Tabbed interface, queries algorithm's TabContentPages for frames, sends them mouse events.

    Track which states are selected but not mouseovered.
      


    """

    def __init__(self, app, alg, bbox_rel, margin_rel=0.0):
        """
        :param app: The application object.
        :param bbox_rel: The bounding box for the panel, relative to the parent frame.
        :param tab_info:  dict with one key/value pair per tab:
            * key:  name of tab
            * value:    display name of tab

        :param resize_callback:  function to call to resize the panel when the tab changes.
            Whoever makes the tab images needs to know the new size, etc.
        """
        self._tabs = OrderedDict()  # (tab_name: (tab_frame, img_label))
        self._tab_image_size = None

        #  Selecting states is common across all tabs, changing in one changes in all, etc.
  
        super().__init__(app, alg, bbox_rel, margin_rel=margin_rel)



    def change_algorithm(self, alg):
        """
        Changing to a new algorithm, from a load or user change + reset.
        Also need to: 
          1. clear the old tabs & set the new tabs.
          2. Refresh the stat
        """
        super().change_algorithm(alg)
        
        tab_content = alg.get_tabs()
        if tab_content is not None:
            self.set_tabs(tab_content)

    def _init_widgets(self):
        """
        Initialize the widgets for the tab panel.
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

    def set_tabs(self, tab_content):
        """
        Set the tabs for the panel.
           * add 'frame' and 'label' keys to each tab in the tab_content dict.

        :param tab_info: (see __init__)
        :param resize_callback: function to call to resize the panel when the tab changes.
           Whoever makes the tab imges needs to know the new size, etc.
        """

        # Clear old tabs' images and frames:
        for old_tab_name in self._tabs.values():
            self._tabs[old_tab_name]['label'].destroy()
            self._tabs[old_tab_name]['frame'].destroy()
        
        # Set new tabs:
        self._tabs = tab_content
        names = [name for name in tab_content.keys()]
        disp_names = {tab_name:tab_content[tab_name]['disp_text'] for tab_name in names}
        self._tab_name_by_disp = {disp_names[tab_name]: tab_name for tab_name in names}

        # Add new tabs:
        for tab_name in names:
            tab_disp_name = disp_names[tab_name]
            # Create a new frame for the tab:
            tab_frame = tk.Frame(self._notebook, bg=self._color_bg)
            tab_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Create a label for the tab to hold the image (filling the whole tab frame):
            img_label = tk.Label(tab_frame, bg=self._color_bg)
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            img_label.bind("<Button-1>", lambda event: self._on_mouse_click(event, tab_name))
            img_label.bind("<Motion>", lambda event: self._on_mouse_move(event, tab_name))
            # Bind mouse-out events
            img_label.bind("<Leave>", lambda event: self._on_mouse_leave(event, tab_name))
            # Bind resize events to the tab frame:
            tab_frame.bind("<Configure>", self.on_tab_resize)

            # Add the tab to the notebook:
            self._notebook.add(tab_frame, text=tab_disp_name)
            # Add the tab to the tabs dictionary:
            self._tabs[tab_name]['frame'] = tab_frame
            self._tabs[tab_name]['label'] = img_label

        # get the current tab:
        self.cur_tab = self._tab_name_by_disp[self._notebook.tab(self._notebook.select(), "text")]
        logging.info("TabPanel set initial tab to %s" % self.cur_tab)

    def get_frame_size(self):
        frame_size = self._frame.winfo_width(), self._frame.winfo_height()
        return frame_size

    def on_tab_resize(self, event):
        new_tab_image_size = (event.width, event.height)
        if self._tab_image_size is None or   (self._tab_image_size != new_tab_image_size):
            logging.info("Resizing tab images to %s" % str(new_tab_image_size))
            self._tab_image_size = new_tab_image_size
            self._alg.resize('state-tabs', new_tab_image_size)
            self.refresh_images(is_paused=self._alg.paused)

    def _on_resize(self, event):
        pass
        
    def refresh_images(self, is_paused, clear=False):
        """
        Get new image from the app, set the image for the current tab.
        """
        if self._tab_image_size is None:
            return
        tab =self._tabs[self.cur_tab]['tab_content']

        if clear:
            # Marking/annotations can change while a different tab is active, other interactions already update the image.
            tab.clear_images(marked_only=True)

        new_img = tab.get_tab_frame(self._tab_image_size, annotated=is_paused)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        label = self._tabs[self.cur_tab]['label']
        label.config(image=new_img)
        label.image = new_img

    def _on_tab_changed(self, event):
        """
        """
        logging.info("Tab changed to %s" % self._notebook.tab(self._notebook.select(), "text"))
        # get the current tab:
        self.cur_tab = self._tab_name_by_disp[self._notebook.tab(self._notebook.select(), "text")]
        self.refresh_images(is_paused=self._alg.paused, clear_tab=True)

    def _on_mouse_click(self, event, tab):
        #logging.info("Mouse click at (%d, %d) on tab %s" % (event.x, event.y, tab))
        content_page = self._tabs[tab]['tab_content']
        if content_page.mouse_click((event.x, event.y)):
            self.refresh_images(is_paused=self._alg.paused)

    def _on_mouse_move(self, event, tab):
        #logging.info("Mouse move at (%d, %d) on tab %s" % (event.x, event.y, tab))
        content_page = self._tabs[tab]['tab_content']
        if content_page.mouse_move((event.x, event.y)):
            self.refresh_images(is_paused=self._alg.paused)

    def _on_mouse_leave(self, event, tab):
        #logging.info("Mouse leave on tab %s" % tab)
        content_page = self._tabs[tab]['tab_content']
        if content_page.mouse_leave():
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
        self._viz_image_size = None
        super().__init__(app, alg, bbox_rel, margin_rel=margin_rel)

    def _init_widgets(self):
        # no widgets, just a label for the viz image
        self._viz_label = tk.Label(self._frame, bg=self._color_bg)
        self._viz_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_resize(self, event):
        self._viz_image_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_images(is_paused=self._alg.paused, control_point=self._alg.current_ctrl_pt)

    # @LPT.time_function

    def refresh_images(self, is_paused, control_point):
        """
        Get new image from the app, set the image for the current tab.
        """
        if self._viz_image_size is None:
            return
        new_img = self._alg.get_viz_image(self._viz_image_size,
                                          control_point=control_point,
                                          is_paused=is_paused)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        self._viz_label.config(image=new_img)
        self._viz_label.image = new_img
        # self._viz_label.update_idletasks()
