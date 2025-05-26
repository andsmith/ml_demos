"""
Algorithm selection, state load/save/reset, fullscreen, start game.
Needs a title, buttons, and radio buttons for the algorithms.
"""

import logging
import tkinter as tk
import numpy as np
from gui_base import Panel
from layout import LAYOUT, TITLE_INDENT
from colors import COLOR_BG, COLOR_LINES, COLOR_TEXT, COLOR_URGENT
from util import tk_color_from_rgb


class SelectionPanel(Panel):
    """
    Display list of algorithms in a radio-box on the left side of the panel.
    Display the following buttons:
        * Save state  (Call self.app method)
        * Load state  (Call self.app method)
        * Reset state  (call self.app method) (toggle button color after selection changes, but do nothing until reset)
        * Fullscreen / Window (toggle button text)
    """

    def __init__(self, app, alg_types, bbox_rel,margin_rel=0.0):
        """
        :param app: The main application object.
        :param alg_types: List of algorithm types to display.
        """
        self._alg_types = alg_types
        self._bbox_rel = bbox_rel
        self._algs_by_name = {alg_type.get_name(): alg_type for alg_type in alg_types}
        self.cur_alg_name = alg_types[0].get_name()  # selected algorithm type
        self._pending_alg_name = None  # algorithm to be started after "RESET" button is pressed

        self._color_lines = tk_color_from_rgb(COLOR_LINES)
        self._color_bg = tk_color_from_rgb(COLOR_BG)
        self._color_text = tk_color_from_rgb(COLOR_TEXT)
        self._color_urgent = tk_color_from_rgb(COLOR_URGENT)
        self.opp_n_rules = 2  # number of rules for heuristic opponent(0-6)
        self._pending_opp_n_rules = None

        super().__init__(app=app, bbox_rel=bbox_rel, margin_rel=margin_rel)

    def _init_widgets(self):
        self._init_title()
        self._init_selection_buttons()
        self._init_demo_buttons()

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        self._title = tk.Label(self._frame, text=(self._algs_by_name[self.cur_alg_name].get_str()),
                               font=LAYOUT['fonts']['panel_title'],
                               bg=self._color_bg, fg=self._color_text)
        self._title.pack(pady=5, padx=TITLE_INDENT, anchor=tk.W)
        self._add_spacer()

    def _init_selection_buttons(self):
        """
        Create the radio buttons for the algorithm selection.
        """
        self._selection_buttons = {}
        for alg_ind, alg_type in enumerate(self._alg_types):
            alg_name, alg_str = alg_type.get_name(), alg_type.get_str()
            avail = not alg_type.is_stub()
            print("Algorithm %s available: %s" % (alg_name, avail))
            self._selection_buttons[alg_name] = tk.Radiobutton(self._frame, text=alg_str,
                                                               variable=self._pending_alg_name,
                                                               value=alg_name,
                                                               command=lambda m=alg_name: self._on_selection_change(m),
                                                               state=tk.NORMAL if avail else tk.DISABLED,
                                                               font=LAYOUT['fonts']['menu'],
                                                               bg=self._color_bg,
                                                               )
            self._selection_buttons[alg_name].pack(anchor=tk.W, padx=15)

    def _init_demo_buttons(self):
        """
        Create the demo buttons for saving, loading, and resetting the state.
        """
        bottom_frame = tk.Frame(self._frame, bg=self._color_bg)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # make a grid with 3 rows for the reset buttons, slider and state buttons
        bottom_frame.grid_rowconfigure(0, weight=1)  # reset button | reset message
        bottom_frame.grid_rowconfigure(1, weight=1)  # slider
        bottom_frame.grid_rowconfigure(2, weight=1)  # (Save) (load) (Fullscreen)
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)

        # Add the reset button directly below the radio buttons:
        self._reset_button = tk.Button(bottom_frame, text="Reset State", command=self._proc_reset,
                                       font=LAYOUT['fonts']['buttons'],
                                       bg=self._color_bg, fg=self._color_text)
        self._reset_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W + tk.E)

        # Add message below the button but hide it
        reset_msg_frame = tk.Frame(bottom_frame, bg=self._color_bg)
        reset_msg_frame.grid(row=0, column=1, columnspan=2)
        self._reset_msg = tk.Label(reset_msg_frame, text="RESET to apply",
                                   font=LAYOUT['fonts']['default'],
                                   bg=self._color_bg, fg=self._color_urgent)
        self._reset_msg.pack(side=tk.LEFT, padx=5, pady=5)
        self._reset_msg.pack_forget()  # hide the message

        self._opponent_slider = tk.Scale(bottom_frame, from_=0, to=6, orient=tk.HORIZONTAL, showvalue=0,
                                         label="Opponent: Heuristic(%i)" % self.opp_n_rules,
                                         bg=self._color_bg, fg=self._color_text, command=self._update_opp_n_rules,
                                         font=LAYOUT['fonts']['menu'], length=250)
        self._opponent_slider.set(self.opp_n_rules)
        self._opponent_slider.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=10)

        # put the save load, fullscreen buttons centered in the bottom frame:
        self._save_button = tk.Button(bottom_frame, text="Save", command=self.app.save_state,
                                      font=LAYOUT['fonts']['buttons'],
                                      bg=self._color_bg, fg=self._color_text)
        self._save_button.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W + tk.E)

        self._load_button = tk.Button(bottom_frame, text="Load", command=self.app.load_state,
                                      font=LAYOUT['fonts']['buttons'],
                                      bg=self._color_bg, fg=self._color_text)
        self._load_button.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W + tk.E)

        self._fullscreen_button = tk.Button(bottom_frame, text="Fullscreen", command=self.app.toggle_fullscreen,
                                            font=LAYOUT['fonts']['buttons'],
                                            bg=self._color_bg, fg=self._color_text)
        self._fullscreen_button.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W + tk.E)

    def _update_opp_n_rules(self, value):
        """
        Update the number of rules for the heuristic opponent.
        :param value: The new number of rules.
        """
        pending_opp_n_rules = int(value)

        if pending_opp_n_rules != self.opp_n_rules:
            self._pending_opp_n_rules = pending_opp_n_rules
            # set message to show the new value
            self._reset_msg.config(text="RESET to apply")
            self._reset_msg.pack()
        else:
            self._pending_opp_n_rules = None
            self._reset_msg.pack_forget()
        self._opponent_slider.config(label="Opponent: HeuristicPlayer(%i)" % pending_opp_n_rules)

    def set_selection(self, name):
        """
        Set the selected algorithm by name.
        :param name: The name of the algorithm to select.
        """
        if name in self._selection_buttons:
            self._selection_buttons[name].select()
            self.cur_alg_name = name
            self._pending_alg_name = None
            self.refresh_title()
        else:
            raise ValueError(f"Algorithm {name} not found in selection buttons.")

    def _on_selection_change(self, new_selection):
        """
        Button callback. 
        if the pending algorithm is different, change the reset button color.
        Otherwise set it back to normal.
        """
        self._pending_alg_name = new_selection if new_selection != self.cur_alg_name else None
        print(f"New pending selected algorithm: {self._pending_alg_name}.(Current: {self.cur_alg_name})")
        if self._pending_alg_name is not None:
            self._reset_msg.config(text="RESET to apply")
            self._reset_msg.pack()  # show the message
        else:
            self._reset_msg.pack_forget()

    def refresh_title(self):
        self._title.config(text=(self._algs_by_name[self.cur_alg_name].get_str()))

    def _proc_reset(self):
        """
        Process the reset button click.
        If the pending algorithm is different from the current one, set it to the current one.
        Otherwise, do nothing.
        """
        # reset agent state
        if self._pending_alg_name is not None:
            logging.info("Resetting selected demo to %s", self._pending_alg_name)
            self.cur_alg_name = self._pending_alg_name
            self._pending_alg_name = None
            self._reset_msg.pack_forget()  # hide the message
            # set the algorithm in the app
            self.refresh_title()
            self.app.change_alg(self.cur_alg_name)
        self.app.reset_state()

        # reset opponent state
        if self._pending_opp_n_rules is not None:
            logging.info("Resetting opponent to %i rules", self._pending_opp_n_rules)
            self.opp_n_rules = self._pending_opp_n_rules
            self._pending_opp_n_rules = None
            self._reset_msg.pack_forget()
            self.app.set_opponent(self.opp_n_rules)

    def _on_resize(self, event):
        return super()._on_resize(event)


class TestApp(object):
    """
    Stand-in for demo app to test the SelectionPanel.
    """

    def __init__(self):
        from layout import WIN_SIZE
        from policy_eval import PolicyEvalDemoAlg, InPlacePEDemoAlg
        from dynamic_prog import DynamicProgDemoAlg, InPlaceDPDemoAlg
        from q_learning import QLearningDemoAlg
        from policy_grad import PolicyGradientsDemoAlg

        ALGORITHMS = [PolicyEvalDemoAlg, InPlacePEDemoAlg, DynamicProgDemoAlg, InPlaceDPDemoAlg,
                      QLearningDemoAlg, PolicyGradientsDemoAlg]

        self.win_size = WIN_SIZE
        self._fullscreen = False
        self.root = tk.Tk()
        self.root.geometry(f"{self.win_size[0]}x{self.win_size[1]}")
        self.root.title("Selection Panel Test")

        self.selection_panel = SelectionPanel(self, ALGORITHMS, LAYOUT['frames']['selection'])
        self.selection_panel.set_selection(name=ALGORITHMS[0].get_name())

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
