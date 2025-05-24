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
from tkinter import Tk, Frame, Label, filedialog
from policy_eval import PolicyEvalDemoAlg, InPlacePEDemoAlg
from dynamic_prog import DynamicProgDemoAlg, InPlaceDPDemoAlg
from q_learning import QLearningDemoAlg
from policy_grad import PolicyGradientsDemoAlg
from layout import LAYOUT, WIN_SIZE
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from selection_panel import SelectionPanel
from reinforcement_base import Environment
from tic_tac_toe import Game
from game_base import Mark, TERMINAL_REWARDS, get_reward
from alg_panels import StatePanel, VisualizationPanel
from status_ctrl_panel import StatusControlPanel
import time
import pickle
from threading import Event
from baseline_players import HeuristicPlayer
# Will display in this order:
from test_panels import TestDemoAlg
from util import tk_color_from_rgb
#from loop_timing.loop_profiler import LoopPerfTimer as LPT

ALGORITHMS = [TestDemoAlg,PolicyEvalDemoAlg, InPlacePEDemoAlg, DynamicProgDemoAlg, InPlaceDPDemoAlg,
              QLearningDemoAlg, PolicyGradientsDemoAlg]

AGENT_MARK = Mark.X  # The agent's mark in the game.
OPPONENT_MARK = Mark.O  # The opponent's mark in the game.

FPS = 15


class RLDemoApp(object):
    def __init__(self):

        self._init_tk()
        self._init_selection()  # This function will also create the current algorithm.

        # Selection panel defines opponent player so we can make the enviornment now
        self._fullscreen = False
        self.shutdown = False  # set to True to signal the app to exit
        self.paused = True
        self._advance_event = Event()  # Event to signal the algorithm to advance.

        self._init_ctrl_point = None  # Start paused here.
        # For running continuous modes:
        self._timing_info = {'t_last_refresh': time.perf_counter(),  # for refreshing the window
                             't_last_print': time.perf_counter(),  # for printing the FPS
                             'print_interval_sec': 1.0,
                             'n_frames': 0,
                             'fps': 0.0}
        self._ticks_skipped = 0
        self._init_alg_panels()

    def _init_tk(self):
        self.root = Tk()
        self.root.configure(bg=tk_color_from_rgb(COLOR_LINES))
        self.root.title("Reinforcement Learning Demo")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")
        # Set background color to black.


    def _init_selection(self):
        self._selection_panel = SelectionPanel(self, ALGORITHMS, LAYOUT['frames']['selection'],margin_rel = LAYOUT['margin_rel'])
        self._selection_panel.set_selection(self._selection_panel.cur_alg_name)  # necessary?

    def _init_alg_panels(self):
        self._opp_policy = HeuristicPlayer(mark=OPPONENT_MARK, n_rules=self._selection_panel.opp_n_rules)
        self._env = Environment(self._opp_policy, AGENT_MARK)
        
        alg_ind = self._get_alg_ind(self._selection_panel.cur_alg_name)
        self._alg = ALGORITHMS[alg_ind](self, self._env)
        self._init_ctrl_point = self._alg.get_init_control()  # first control point

        # Create panels that need to know about the algorithm:
        self._status_control_panel = StatusControlPanel(self, self._alg, LAYOUT['frames']['control'],margin_rel = LAYOUT['margin_rel'])
        self._state_panel = StatePanel(self, self._alg, LAYOUT['frames']['state-tabs'],margin_rel = LAYOUT['margin_rel'])
        self._visualization_panel = VisualizationPanel(self, self._alg, LAYOUT['frames']['step-visualization'],margin_rel = LAYOUT['margin_rel'])

    def start_alg(self):
        self._alg.start(self._advance_event)


    def change_alg(self, alg_name):
        # Stop the current algorithm.
        logging.info("Stopping current algorithm...")

        self._alg.stop()
        
        # Get and start the new algorithm.
        logging.info("Starting new algorithm: %s", alg_name)
        alg_ind = self._get_alg_ind(alg_name)
        self._alg = ALGORITHMS[alg_ind](self, self._env)

        self.start_alg()
        #import ipdb ; ipdb.set_trace()
        # Update the panels with the new algorithm.
        self._status_control_panel.change_algorithm(self._alg)  # checks all control points (all stops on)
        self._state_panel.change_algorithm(self._alg)
        self._visualization_panel.change_algorithm(self._alg)

        # Refresh panels' images with new algorithm.
        self._init_ctrl_point = self._alg.get_init_control()  # first control point
        self.paused = True
        self._state_panel.refresh_images(is_paused=self.paused)
        self._visualization_panel.refresh_images(is_paused=self.paused, control_point=self._init_ctrl_point)
        #self._status_control_panel.refresh_status()

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

    def _get_alg_ind(self, name):
        """
        Get the index of the algorithm in the ALGORITHMS list.
        :param name: The name of the algorithm.
        :return: The index of the algorithm in the ALGORITHMS list.
        """
        for i, alg in enumerate(ALGORITHMS):
            if alg.get_name() == name:
                return i
        raise ValueError(f"Algorithm {name} not found in ALGORITHMS list.")

    def save_state(self):
        """
        Open a save dialog, call the algorithm's save_state function.
        """
        filename = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if filename:
            self._alg.save_state(filename, clobber=True)
            logging.info(f"State saved to {filename}")

    def load_state(self):
        """
        Open a load dialog, call the algorithm's load_state function.
        """
        filename = filedialog.askopenfilename(defaultextension=".pkl",
                                              filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if filename:
            with open(filename, 'rb') as f:
                alg_name = pickle.load(f)
            alg_ind = self._get_alg_ind(alg_name)
            self._alg = ALGORITHMS[alg_ind](self, self._env)
            self._init_ctrl_point = self._alg.get_init_control()  
            self._alg.load_state(filename)
            logging.info(f"State loaded from {filename}")
            # new algorithm might be a different type, so inform the selection panel:
            print("Changing selector to loaded type: ", alg_name)
            self._selection_panel.set_selection(name=alg_name)

    def set_opponent(self, n_rules):
        """
        Set the opponent policy to a new heuristic player with the given number of rules.
        NOTE: This only gets called as part of a the init or a reset.
        :param n_rules: The number of rules for the opponent policy.
        """
        self._opp_policy = HeuristicPlayer(mark=OPPONENT_MARK, n_rules=n_rules)
        self._env.set_opp_policy(self._opp_policy)

    def reset_state(self):
        logging.info("Resetting demo state.")
        self._alg.reset_state()
        self._status_control_panel.refresh_status()
        self._state_panel.refresh_images(is_paused=self.paused)
        self._visualization_panel.refresh_images(is_paused=self.paused, control_point=self._init_ctrl_point)

    def start(self):
        logging.info("Starting algorithm.")
        self._alg.start(self._advance_event)
        logging.info("Starting RL Demo App")
        self.root.mainloop()
        logging.info("Exiting RL Demo App")

    def get_aspect(self):
        """
        Get the aspect ratio of the window.
        :return: The aspect ratio of the window.
        """
        width, height = self.root.winfo_width(), self.root.winfo_height()
        return width / height

    #@LPT.time_function
    def tick(self, is_paused, control_point):
        """
        Called every tick to update the algorithm and the panels.
        :param is_paused: Whether the algorithm is paused or not.
        """
        if is_paused:
            self.paused = True
            self._status_control_panel.refresh_status()
            self._state_panel.refresh_images(is_paused=True)
            self._visualization_panel.refresh_images(is_paused=True, control_point=control_point)
        else:
            self.paused = False
            now = time.perf_counter()
            elapsed = now - self._timing_info['t_last_refresh']

            if elapsed > 1.0 / FPS:
                self._status_control_panel.refresh_status()
                self._state_panel.refresh_images(is_paused=False)
                self._visualization_panel.refresh_images(is_paused=False, control_point=control_point)
                self._timing_info['n_frames'] += 1
                self._timing_info['t_last_refresh'] = now

            else:
                self._ticks_skipped += 1
            elapsed = now - self._timing_info['t_last_print']
            if elapsed >= self._timing_info['print_interval_sec']:
                self._timing_info['fps'] = self._timing_info['n_frames'] / elapsed
                logging.info(
                    f"N-frames:  {self._timing_info['n_frames']}, FPS: {self._timing_info['fps']:.2f}, ticks skipped: {self._ticks_skipped}")
                self._timing_info['n_frames'] = 0
                self._ticks_skipped
                self._timing_info['t_last_print'] = now


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    demo = RLDemoApp()
    demo.start()
