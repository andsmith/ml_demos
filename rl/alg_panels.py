"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""

import tkinter as tk
import numpy as np
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE
import logging

TITLE_INDENT = 0
ITEM_INDENT = 20
class StatusControlPanel(Panel):
    """
    Status panel for displaying the current status of the algorithm and buttons for controlling it.
    """

    def __init__(self, app,alg,  alg_types, bbox_rel):
        """
        :param app: The application object.
        :param alg: a DemoAlg object (RL state)
        """
        self._algs_by_name = {alg_type.get_name(): alg_type for alg_type in alg_types}    
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

        self._init_status()  # top half of the panel
        self._init_run_control() # bottom left half of the panel
        self._init_buttons() # bottom right half of the panel

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        # Add dark line below the title:
        #self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._line_color)
        #self._title_line.pack(side=tk.TOP)
        self._add_spacer()
    
    def _init_status(self):
        """
        Goes at the top of the status panel, one line per status message.
        """
        status_title = tk.Label(self._status_frame, text="Status",
                               font=LAYOUT['fonts']['title'],
                               bg=self._bg_color, fg=self._text_color, anchor="w", justify="left")
        status_title.pack(side=tk.TOP,fill=tk.X, padx=TITLE_INDENT, pady=4)

        test_status = self._alg.get_status()

        self._status_labels = []  # list of status labels, aligned on the left
        for i, (text, font) in enumerate(test_status):
            label = tk.Label(self._status_frame, text=text, bg=self._bg_color, font=font, anchor="w", justify="left")
            label.pack(side=tk.TOP, fill=tk.X, padx=ITEM_INDENT, pady=0)
            self._status_labels.append(label)

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
        breakpoint_label = tk.Label(self._run_control_frame, text="Breakpoints", bg=self._bg_color, font=LAYOUT['fonts']['title'], anchor="w", justify="left")
        breakpoint_label.pack(side=tk.TOP,fill=tk.X, padx=TITLE_INDENT, pady=4)

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
            check = tk.Checkbutton(self._run_control_frame, text=text, variable=var, bg=self._bg_color,
                                    font=LAYOUT['fonts']['default'], anchor="w", justify="left")
            check.pack(side=tk.TOP, fill=tk.X, padx=ITEM_INDENT, pady=0)
            self._run_control_options.append((check, var, key))

    def _init_buttons(self):
        """
        Below status, right half of the panel.
        """
        self._add_spacer(20,frame=self._button_frame)
        # "Clear breakpoints" button:
        self._clear_button = tk.Button(self._button_frame, text="Clear Breakpoints",
                                    font=LAYOUT['fonts']['buttons'],bg=self._bg_color,
                                    command=self._clear_breakpoints)
        self._clear_button.pack(side=tk.TOP, fill=tk.X, padx=4, pady=10)

        # "Go/Stop" button
        self._go_button = tk.Button(self._button_frame, text="Go", 
                                    font=LAYOUT['fonts']['buttons'],bg=self._bg_color,
                                    command=self._go_stop, padx=10)
        self._go_button.pack(side=tk.TOP, padx=4, pady=10)

    def _clear_breakpoints(self):
        pass
    def _go_stop(self):
        pass

class TestApp(object):
    """
    Stand-in for demo app to test the SelectionPanel.
    """

    def __init__(self):
        from layout import WIN_SIZE
        from policy_eval import PolicyEvalDemoAlg
        from selection_panel import SelectionPanel
        from game_base import Mark
        from reinforcement_base import Environment
        from baseline_players import HeuristicPlayer
        self.win_size = WIN_SIZE
        self._fullscreen = False
        self.root = tk.Tk()
        self.root.geometry(f"{self.win_size[0]}x{self.win_size[1]}")
        self.root.title("Selection Panel Test")
        self._env = Environment(opponent_policy=HeuristicPlayer(mark=Mark.O, n_rules=2))
        self._alg = PolicyEvalDemoAlg(self, self._env)
        self.status_control_panel = StatusControlPanel(self, self._alg ,[PolicyEvalDemoAlg], bbox_rel=LAYOUT['frames']['control'])
        self.selection_panel = SelectionPanel(self, [PolicyEvalDemoAlg], bbox_rel=LAYOUT['frames']['selection'])


    def run(self):
        self.root.mainloop()

    def save_state(self):
        logging.info("Save state button pressed.")

    def load_state(self):
        logging.info("Load state button pressed.")

    def reset_state(self):
        logging.info("Reset state button pressed.")

    def toggle_fullscreen(self):
        logging.info("Toggle fullscreen button pressed.")

    def toggle_fullscreen(self):

        self._fullscreen = not self._fullscreen
        global geom
        if self._fullscreen:
            geom = self.root.geometry()
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.overrideredirect(True)
            self.root.geometry('%dx%d+0+0' % (w, h))

        else:
            self.root.overrideredirect(False)
            self.root.geometry(geom)

    def set_opponent(self, n_rules):
        logging.info(f"Setting opponent to {n_rules} rules.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = TestApp()
    app.run()
