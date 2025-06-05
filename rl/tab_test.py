import tkinter as tk
from tkinter import ttk
from state_tab_content import FullStateContentPage
from policy_eval import PolicyEvalDemoAlg
from colors import COLOR_SCHEME
import numpy as np
import logging
from PIL import Image, ImageTk
from util import tk_color_from_rgb
from baseline_players import HeuristicPlayer
from reinforcement_base import Environment
from tic_tac_toe import Mark,Game
from mouse_state_manager import MouseBoxManager
from layout import LAYOUT
from alg_panels import TabPanel


class FakeApp(object):
    """
    stand-in for rl_demo.RLDemoApp

    """
    def __init__(self,size):
        self.size=size
        self.root=None
        self._init_tk()


        self.player = Mark.X
        
        self.opp_policy = HeuristicPlayer(mark=Mark.O, n_rules=2)
        self.env = Environment(self.opp_policy,self.player)
        terminals, nonterminals = self.env.get_terminal_states(), self.env.get_nonterminal_states()
        self.all_states = {state:None for state in terminals + nonterminals}
        self.selected = []

    def toggle_selected_state(self, state_id):
        if state_id not in self.all_states:
            raise ValueError(f"State {state_id} not found in all_states.")
        if state_id in self.selected:
            self.selected.remove(state_id)
        else:
            self.selected.append(state_id)
        print(f"Toggled selection for state {state_id}, now selected: {state_id in self.selected}, Total selected: {len(self.selected)}")   
    def _init_tk(self):
        """
        Init Tkinter and create the main window.
        """
        self.root = tk.Tk()
        self.root.title("Fake App for Tab Tester")
        self.root.geometry(f"{self.size[0]}x{self.size[1]}")
        self.root.configure(bg=tk_color_from_rgb(COLOR_SCHEME['bg']))   
        self.root.resizable(True, True)

    def start(self):
        self.root.mainloop()


class TabPanelTest(object):
    """
    Create an TabPanel object in its own frame (the whole window).
    Simulate the main app, without the other panels.
    """
    def __init__(self, app, alg):
        self._app = app
        self._alg = alg
        self._img_size = app.size

        bbox = {'x_rel':(0, 1), 'y_rel':(0, 1)}  # Full window
        self._tab_panel = TabPanel(app, alg, bbox, margin_rel=LAYOUT['margin_rel'])


    def _change_algorithm(self, alg):
        tab_content = alg.get_tabs()
        if tab_content is not None:
            self.set_tabs(tab_content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    img_size = (1200, 970)
    box_size = 20
    app = FakeApp(img_size)  # Stand-in for the RLDemoApp
    pi_seed = HeuristicPlayer(mark=Mark.X, n_rules=1)
    alg = PolicyEvalDemoAlg(app,pi_seed)
    
    tester= TabPanelTest(app, alg)
    app.start()
    logging.info("Tab Tester finshed.")
