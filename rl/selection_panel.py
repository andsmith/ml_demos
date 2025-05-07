"""
Algorithm selection, state load/save/reset, fullscreen, start game.
Needs a title, buttons, and radio buttons for the algorithms.
"""

import logging
import tkinter as tk
import numpy as np
from gui_base import Panel
from layout import LAYOUT, FRAME_TITLES
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

    def __init__(self, app, alg_types, bbox_rel):
        """
        :param app: The main application object.
        :param alg_types: List of algorithm types to display.
        """
        super().__init__(app=app, bbox_rel=bbox_rel)
        self._alg_types = alg_types
        self._bbox_rel = bbox_rel
        self._cur_alg_name = alg_types[0].get_name()  # selected algorithm type
        self._pending_alg_name = None  # algorithm to be started after "RESET" button is pressed

        self._color_lines = tk_color_from_rgb(COLOR_LINES)
        self._color_bg = tk_color_from_rgb(COLOR_BG)
        self._color_text = tk_color_from_rgb(COLOR_TEXT)
        self._color_urgent = tk_color_from_rgb(COLOR_URGENT)

    def init(self):
        self._init_title()
        self._init_selection_buttons()
        self._init_demo_buttons()

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        self._title = tk.Label(self._frame, text=FRAME_TITLES['selection'],
                               font=LAYOUT['fonts']['panel_title'],
                               bg=self._color_bg, fg=self._color_text)
        self._title.pack(pady=5)

        # Add dark line below the title:
        self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._color_lines)
        self._title_line.pack(side=tk.TOP)

    def _init_selection_buttons(self):
        """
        Create the radio buttons for the algorithm selection.
        """
        self._selection_buttons = {}
        for alg_ind, alg_type in enumerate(self._alg_types):
            alg_name, alg_str = alg_type.get_name(), alg_type.get_str()
            self._selection_buttons[alg_name] = tk.Radiobutton(self._frame, text=alg_str,
                                                               variable=self._cur_alg_name,
                                                               value=alg_name,
                                                               command=lambda m=alg_name: self._on_selection_change(m),
                                                               state=tk.NORMAL,  # set to disabled if unimplemented
                                                               font=LAYOUT['fonts']['menu'],
                                                               bg=self._color_bg,
                                                               )
            self._selection_buttons[alg_name].pack(anchor=tk.W, padx=25)

    
            

    def _init_demo_buttons(self):
        """
        Create the demo buttons for saving, loading, and resetting the state.
        """
        # Add dark line below the radio buttons:
        button_line = tk.Frame(self._frame, height=2, width=150, bg=self._color_lines)
        button_line.pack(side=tk.TOP, pady=8)
        #self._add_spacer()

        # Add the reset button directly below the radio buttons:
        self._reset_button = tk.Button(self._frame, text="Reset State", command=self.app.toggle_fullscreen,
                                       font=LAYOUT['fonts']['buttons'],
                                       bg=self._color_bg, fg=self._color_text)
        self._reset_button.pack(side=tk.TOP, pady=5, padx=25)

        # Add message below the button but hide it
        self._reset_msg = tk.Label(self._frame, text="[reset to switch algorithms]",
                                   font=LAYOUT['fonts']['default'],    
                                      bg=self._color_bg, fg=self._color_urgent)
        self._reset_msg.pack(side=tk.TOP, pady=5, padx=25)
        self._reset_msg.lower()  # hide the message

        bottom_frame = tk.Frame(self._frame, bg=self._color_bg)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.Y, padx=5, pady=5)
        
        # put the save load, fullscreen buttons centered in the bottom frame:
        self._save_button = tk.Button(bottom_frame, text="Save State", command=self.app.save_state,
                                      font=LAYOUT['fonts']['buttons'],
                                      bg=self._color_bg, fg=self._color_text)
        self._save_button.pack(side=tk.LEFT, padx=5, pady=5)

        self._load_button = tk.Button(bottom_frame, text="Load State", command=self.app.load_state,
                                        font=LAYOUT['fonts']['buttons'],
                                        bg=self._color_bg, fg=self._color_text)
        self._load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self._fullscreen_button = tk.Button(bottom_frame, text="Fullscreen", command=self.app.toggle_fullscreen,
                                        font=LAYOUT['fonts']['buttons'],
                                        bg=self._color_bg, fg=self._color_text)
        self._fullscreen_button.pack(side=tk.LEFT, padx=5, pady=5)

    def set_selection(self, name):
        """
        Set the selected algorithm by name.
        :param name: The name of the algorithm to select.
        """
        if name in self._selection_buttons:
            self._selection_buttons[name].select()
        else:
            raise ValueError(f"Algorithm {name} not found in selection buttons.")

    def _on_selection_change(self, new_selection):
        """
        Button callback. 
        if the pending algorithm is different, change the reset button color.
        Otherwise set it back to normal.
        """
        self._pending_alg_name = new_selection if new_selection != self._cur_alg_name else None
        print(f"New pending selection: {self._pending_alg_name}.")

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

        ALGORITHMS = [PolicyEvalDemoAlg, InPlacePEDemoAlg, DynamicProgDemoAlg, InPlaceDPDemoAlg]

        self.win_size = WIN_SIZE
        self.root = tk.Tk()
        self.root.geometry(f"{self.win_size[0]}x{self.win_size[1]}")
        self.root.title("Selection Panel Test")

        self.selection_panel = SelectionPanel(self, ALGORITHMS, LAYOUT['frames']['selection'])
        self.selection_panel.init()
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = TestApp()
    app.run()
