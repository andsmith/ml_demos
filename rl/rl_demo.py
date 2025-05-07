"""
Entry point for the RL demo app.

This file:  
    - Set up the TK window.
    - Start algorithm selection panel.
    - Create the algorithm object, let it know where it's 3 panels go:
        - status and control, 
        - step visualization, 
        - state/value/update visualization).
    - Initializes all panels (panels manage their own settings/options/tabs)
    - Start the app. 
"""
import logging
from tkinter import Tk, Frame, Label
from policy_eval import PolicyEvalDemoAlg, InPlacePEDemoAlg
from dynamic_prog import DynamicProgDemoAlg, InPlaceDPDemoAlg
from layout import LAYOUT, WIN_SIZE
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from selection_panel import SelectionPanel

# Will display in this order:
ALGORITHMS = [PolicyEvalDemoAlg, InPlacePEDemoAlg, DynamicProgDemoAlg, InPlaceDPDemoAlg]


class RLDemoApp(object):
    def __init__(self):
        self._init_tk()
        self._init_selection()  # This function will also set the current algorithm.
        self._init_algorithm()  # And this starts its frames.
        self._alg = None # current DemoAlg object

    def _init_tk(self):
        self.root = Tk()
        self.root.title("Reinforcement Learning Demo")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")

    def _init_selection(self):
        self._selector = SelectionPanel(self, ALGORITHMS)
        self._selector.set_selection(name=ALGORITHMS[0].name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
