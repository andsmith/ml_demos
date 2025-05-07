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
import pickle
# Will display in this order:
ALGORITHMS = [PolicyEvalDemoAlg, InPlacePEDemoAlg, DynamicProgDemoAlg, InPlaceDPDemoAlg,
                      QLearningDemoAlg, PolicyGradientsDemoAlg]



class RLDemoApp(object):
    def __init__(self):
        self._init_tk()
        self._init_selection()  # This function will also set the current algorithm.
        alg_ind = self._get_alg_ind(self._selector.cur_alg_name)
        self._alg = ALGORITHMS[alg_ind](self)
        self._fullscreen = False
        self.shutdown = False  # set to True to signal the app to exit

        self._pending_clears = []  # call these functions when a button is pressed (clear status msgs, etc.)

    def _init_tk(self):
        self.root = Tk()
        self.root.title("Reinforcement Learning Demo")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")

    def _init_selection(self):
        self._selector = SelectionPanel(self, ALGORITHMS, LAYOUT['frames']['selection'])
        self._selector.set_selection(name=ALGORITHMS[0].get_name())

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
            self._alg = ALGORITHMS[alg_ind](self)
            self._alg.load_state(filename)
            logging.info(f"State loaded from {filename}")
            # new algorithm might be a different type, so inform the selection panel:
            print("Changing selector to loaded type: ", alg_name)
            self._selector.set_selection(name=alg_name)

    def change_alg(self, alg_name):  
        alg_ind = self._get_alg_ind(alg_name)
        self._alg = ALGORITHMS[alg_ind](self)

    def reset_state(self):
        logging.info("Resetting demo state.")
        self._alg.reset_state()

    def start(self):
        logging.info("Starting RL Demo App")
        self.root.mainloop()
        logging.info("Exiting RL Demo App")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    demo = RLDemoApp()
    demo.start()
