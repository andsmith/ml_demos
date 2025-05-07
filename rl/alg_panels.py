"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""

import tkinter as tk
import numpy as np
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, FRAME_TITLES, WIN_SIZE
import logging

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
        super().__init__(app, bbox_rel)

    def _init_widgets(self):
        print("WDIDGETS")
        self._init_title()
        self._init_status_frame()  # top half of the panel
        #self._init_run_control_frame() # bottom left half of the panel
        #self._init_button_frame() # bottom right half of the panel

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        self._title = tk.Label(self._frame, text=FRAME_TITLES['control'],
                               font=LAYOUT['fonts']['panel_title'],
                               bg=self._bg_color, fg=self._text_color)
        self._title.pack(pady=5)

        # Add dark line below the title:
        self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._line_color)
        self._title_line.pack(side=tk.TOP)
        self._add_spacer()
    
    def _init_status_frame(self):
        """
        Goes at the top of the status panel, one line per status message.
        """
        test_status = self._alg.get_status()
        print(test_status)
        self._status_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._status_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
        
        self._status_labels = []  # list of status labels, aligned on the left
        for i, (text, font) in enumerate(test_status):
            label = tk.Label(self._status_frame, text=text, bg=self._bg_color, font=font, anchor="w", justify="left")
            label.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
            self._status_labels.append(label)

    def refresh_status(self):
        """
        Refresh the status labels with the current status of the algorithm.
        """
        test_status = self._alg.get_status()
        for i, (text, font) in enumerate(test_status):
            self._status_labels[i].config(text=text, font=font)
        # TODO: Add/shrink labels as needed.

    def set_text(self, text):
        """
        Set the text of the status label.
        :param text: The text to set.
        """
        self._label.config(text=text)   

    def _on_resize(self, event):
        return super()._on_resize(event)

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
