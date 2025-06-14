"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""
import traceback

from policy_eval import PolicyEvalDemoAlg
from collections import OrderedDict
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import numpy as np
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE
import logging
import time
from threading import Thread, Event, get_ident
import cv2
# from loop_timing.loop_profiler import LoopPerfTimer as LPT

TITLE_INDENT = 5
ITEM_INDENT = 20


class StatusControlPanel(Panel):
    """
    Status panel for displaying the current status of the algorithm and buttons for controlling it.
    """

    def __init__(self, app, alg, bbox_rel, margin_rel=0.0):
        """
        :param app: The application object.
        :param alg: a DemoAlg object (RL state)
        """
        self._alg = alg
        self._run_control_options = []
        super().__init__(app, bbox_rel, margin_rel=margin_rel)

    def _init_widgets(self):

        # init three frames:
        self._n_status_lines = 8
        y1 = .50
        y2 = .82
        self._status_frame = tk.Frame(self._frame, bg=self._color_bg)
        self._status_frame.place(relx=0, rely=0, relwidth=1, relheight=y1)
        self._run_control_frame = tk.Frame(self._frame, bg=self._color_bg)
        self._run_control_frame.place(relx=0, rely=y1, relwidth=1., relheight=y2-y1)
        self._button_frame = tk.Frame(self._frame, bg=self._color_bg)
        self._button_frame.place(relx=0.0, rely=y2, relwidth=1., relheight=1-y2)
        self._frame.bind("<Configure>", self._on_resize)

        self._run_control_vars = {}  # state of checkboxes
        self._run_control_options = []  # (name, display text) tuples

        self._init_title()

        self._init_status()  # top half of the panel
        self._init_run_control()  # bottom left half of the panel
        self._init_buttons()  # bottom right half of the panel

    def set_run_control_setting(self, control_point, value=True):
        """
        Set the value of a run control setting.
        :param control_point: The name of the control point to set.
        :param value: The value to set it to (default: True).
        """
        for _, var, key in self._run_control_options:
            if key == control_point:
                var.set(1 if value else 0)
                self._on_control_change()
                return
        raise ValueError("Control point '%s' not found in run control options." % control_point)

    def get_run_control_settings(self):
        """
        :returns: {option-name:  bool} dict of the current state of the run control options.
        """
        options = {key: (var.get() == 1) for _, var, key in self._run_control_options}
        return options

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        # Add dark line below the title:
        # self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._line_color)
        # self._title_line.pack(side=tk.TOP)

        # add new status labels:
        status_title = tk.Label(self._status_frame, text="Status",
                                font=LAYOUT['fonts']['title'],
                                bg=self._color_bg, fg=self._color_text, anchor="w", justify="left")
        status_title.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

    def _init_status(self):
        """
        Goes at the top of the status panel, one line per status message.
        """
        self._status_labels = []
        for i in range(self._n_status_lines):
            label = tk.Label(self._status_frame, text="", bg=self._color_bg, anchor="w", justify="left")
            label.pack(side=tk.TOP, fill=tk.X, padx=ITEM_INDENT, pady=0)
            self._status_labels.append(label)

    # @LPT.time_function
    def refresh_status(self):
        """
        Refresh the status labels with the current status of the algorithm.
        """

        status_lines = self._alg.get_status()
        if len(status_lines) > self._n_status_lines:
            raise ValueError("Too many status lines: %i > %i" % (len(status_lines), self._n_status_lines))
        
        
        #print("\n(Obj %s, alg %s) status lines:\n\t%s\n\n"%(id(self), id(self._alg),"\n\t".join([t for t, _ in status_lines])))
        #traceback.print_stack()

        for i in range(self._n_status_lines):
            if i >= len(status_lines):
                text = " "
                font = LAYOUT['fonts']['status']
            else:
                text, font = status_lines[i]
            self._status_labels[i].config(text=text, font=font)

    def _on_resize(self, event):
        return super()._on_resize(event)

    def _init_run_control(self):
        """
        Below status, left half of the panel.
        """
        breakpoint_label = tk.Label(self._run_control_frame, text="Checkpoints",
                                    bg=self._color_bg, font=LAYOUT['fonts']['title'], anchor="w", justify="left")
        breakpoint_label.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

        self._add_spacer(frame=self._run_control_frame)
        self._run_control_frame.grid_rowconfigure(0, weight=1)  # make the bottom frame fill the remaining space
        self._run_control_frame.grid_columnconfigure(0, weight=1)
        self._reset_run_control_options()

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
            check = tk.Checkbutton(self._run_control_frame, text=text, variable=var, bg=self._color_bg,
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
                                       font=LAYOUT['fonts']['buttons'], bg=self._color_bg,
                                       command=self._clear_breakpoints, padx=7, pady=5)
        self._clear_button.pack(side=tk.LEFT, pady=10, padx=5)

        # "Go/Stop" button
        self._go_button = tk.Button(self._button_frame, text="Go",
                                    font=LAYOUT['fonts']['buttons'], bg=self._color_bg,
                                    command=self._go_stop, padx=7, pady=5)
        self._go_button.pack(side=tk.LEFT, pady=10, padx=5)

    def _clear_breakpoints(self):
        self.app.clear_stop_states()

    def _go_stop(self):

        rcs = self.get_run_control_settings()
        self._alg.update_run_control(rcs)
        self._alg.advance()

    def change_algorithm(self, alg):
        """
        Change the algorithm for the panel.
        :param alg: The new algorithm.
        """
        super().change_algorithm(alg)
        self._reset_run_control_options()
        #self.refresh_status()
