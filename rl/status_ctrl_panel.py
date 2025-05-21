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
from layout import LAYOUT, WIN_SIZE
import logging
import time
from threading import Thread, Event, get_ident
import cv2
from loop_timing.loop_profiler import LoopPerfTimer as LPT

TITLE_INDENT = 0
ITEM_INDENT = 20


class StatusControlPanel(Panel):
    """
    Status panel for displaying the current status of the algorithm and buttons for controlling it.
    """

    def __init__(self, app, alg, bbox_rel):
        """
        :param app: The application object.
        :param alg: a DemoAlg object (RL state)
        """
        self._alg = alg
        self._run_control_options = []
        super().__init__(app, bbox_rel)

    def _init_widgets(self):

        # init three frames:
        self._status_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._status_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self._run_control_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._run_control_frame.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self._button_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._button_frame.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)

        self._run_control_vars = {}

        self._init_status()  # top half of the panel
        self._init_run_control()  # bottom left half of the panel
        self._init_buttons()  # bottom right half of the panel

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        # Add dark line below the title:
        # self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._line_color)
        # self._title_line.pack(side=tk.TOP)
        self._add_spacer()

    def _init_status(self):
        """
        Goes at the top of the status panel, one line per status message.
        """
        status_title = tk.Label(self._status_frame, text="Status",
                                font=LAYOUT['fonts']['title'],
                                bg=self._bg_color, fg=self._text_color, anchor="w", justify="left")
        status_title.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

        test_status = self._alg.get_status()

        self._status_labels = []  # list of status labels, aligned on the left
        for i, (text, font) in enumerate(test_status):
            label = tk.Label(self._status_frame, text=text, bg=self._bg_color, font=font, anchor="w", justify="left")
            label.pack(side=tk.TOP, fill=tk.X, padx=ITEM_INDENT, pady=0)
            self._status_labels.append(label)

    @LPT.time_function
    def refresh_status(self):
        """
        Refresh the status labels with the current status of the algorithm.
        """
        test_status = self._alg.get_status()
        for i, (text, font) in enumerate(test_status):
            self._status_labels[i].config(text=text, font=font)
        # TODO: Add/shrink labels as needed.

    def _on_resize(self, event):
        return super()._on_resize(event)

    def _init_run_control(self):
        """
        Below status, left half of the panel.
        """
        breakpoint_label = tk.Label(self._run_control_frame, text="Checkpoints",
                                    bg=self._bg_color, font=LAYOUT['fonts']['title'], anchor="w", justify="left")
        breakpoint_label.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

        self._add_spacer(frame=self._run_control_frame)
        self._run_control_frame.grid_rowconfigure(0, weight=1)  # make the bottom frame fill the remaining space
        self._run_control_frame.grid_columnconfigure(0, weight=1)
        self._reset_run_control_options()

    def get_run_control_settings(self):
        """
        :returns: {option-name:  bool} dict of the current state of the run control options.
        """
        options = {key: (var.get() == 1) for _, var, key in self._run_control_options}
        return options

    def _reset_run_control_options(self):
        """
        Initializing or algorithm changed, look up options, will be ordered dict of (key, string) pairs.
        The algorithm will send its updates labeled with the key.
        The strings for the options will display with check boxes.
        If there are currently the wrong number of options remove/add them. 
        """
        alg_options = self._alg.get_run_control_options()
        # remove old options:
        for option in self._run_control_options:
            option[0].destroy()
        self._run_control_options = []

        # add new options:
        for key, text in alg_options.items():
            var = tk.IntVar()
            check = tk.Checkbutton(self._run_control_frame, text=text, variable=var, bg=self._bg_color,
                                   font=LAYOUT['fonts']['default'], anchor="w", justify="left", command=self._on_control_change)
            check.pack(side=tk.TOP, fill=tk.X, padx=10, pady=0)
            self._run_control_options.append((check, var, key))
            # select all stops
            var.set(1)

    def _on_control_change(self):
        new_vals = self.get_run_control_settings()
        logging.info("Run control options changed: %s" % new_vals)
        self._alg.update_run_control(new_vals)

    def _init_buttons(self):
        """
        Below status, right half of the panel.
        """
        self._add_spacer(20, frame=self._button_frame)
        # "Clear breakpoints" button:
        self._clear_button = tk.Button(self._button_frame, text="Clear Stops",
                                       font=LAYOUT['fonts']['buttons'], bg=self._bg_color,
                                       command=self._clear_breakpoints, padx=7, pady=5)
        self._clear_button.pack(side=tk.TOP, pady=10)

        # "Go/Stop" button
        self._go_button = tk.Button(self._button_frame, text="Go",
                                    font=LAYOUT['fonts']['buttons'], bg=self._bg_color,
                                    command=self._go_stop, padx=7, pady=5)
        self._go_button.pack(side=tk.TOP, pady=10)

    def _clear_breakpoints(self):
        pass

    def _go_stop(self):
        rcs = self.get_run_control_settings()
        self._alg.update_run_control(rcs)
        self._alg.advance()


    def change_algorithm(self, alg):
        """
        Change the algorithm for the panel.
        :param alg: The new algorithm.
        """
        self._alg = alg
        self._reset_run_control_options()
        self.refresh_status()