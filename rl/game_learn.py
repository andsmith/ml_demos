"""
Tic-Tac-Toe Reinforcement Learning demo app.

Main window is three pannels, two large images side-by-side, and a flat control panel at the bottom.
Two image panels at the top are:
  - value function V_t(S) at time t, and
  - current delta-V_t(s), pending/recent updates to make V_{t+1}(s).
The image of a function of s, a game state is reprsented as a game-state tree with single colored
boxes instead of game state images, the color of a "state" indicating the value.
"""
import matplotlib.pyplot as plt
import numpy as np
from drawing import GameStateArtist
from tic_tac_toe import Game, get_game_tree_cached
from node_placement import FixedCellBoxOrganizer
from game_base import Result, Mark
from reinforcement_base import Environment
from abc import ABC, abstractmethod
from policies import ValueFuncPolicy
import logging
from baseline_players import HeuristicPlayer
from collections import OrderedDict
from gui_components import RLDemoWindow
from threading import Thread, Lock, Event

# r is known only for these states in adavnce:
TERMINAL_REWARDS = {
    Result.X_WIN: 1.0,
    Result.O_WIN: -1.0,
    Result.DRAW: -.666,
}

WIN_SIZE = (1920, 990)


import pickle
class PolicyImprovementDemo(ABC):
    """
    Abstract class for Policy Improvment, i.e. iteration between:
      - value function estimation and
      - policy optimization.  

    """

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, gamma=0.9):
        """
        :param seed_policy:  The first value function will be for this policy.
        :param opponent_policy:  The policy of the opponent.  Will be used to define the "environment" for the agent.
        :param player:  The player for the agent.  The opponent is the other player.
        """
        self._iter=0
        self._player = player
        self._max_iter = 1000
        self._n_updated =0
        self._epoch =0 
        self._pending_pause = False

        self._seed_p = seed_policy
        self._opponent_p = opponent_policy
        self._gamma = gamma  # discount factor for future rewards.

        # P.I. initialization:
        #self._env = Environment(opponent_policy, player)
        #with open('env_x.pkl', 'wb') as f:
        #    pickle.dump(self._env, f)
        with open('env_x.pkl', 'rb') as f:
            self._env = pickle.load(f)
        self.children = self._env.get_children()

        self._pi = seed_policy
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        self._v_terminal = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self.running_tournament = False
        self._v = None  # dict from state to value function V_t (s)
        self._v_new = None  # dict from state to new value function V_{t+1} (s)

        self.reset()  

        self._shutdown = False
        # App  & graphics:
        
        self._action_signal = Event()  # learner will wait for this, action button will set it.
        self._init_gui()



    def reset(self):
        logging.info("Learn app reset.")
        # initial value function
        self._v = {state: 0.0 for state in self.updatable_states}
        self._v.update(self._v_terminal)
        self._v_new = {state: 0.0 for state in self.updatable_states}
        self._v_new.update(self._v_terminal)

        self._iter = 0
        self._pending_pause = False




    @abstractmethod
    def optimize_value_function(self):
        """
        Update the value function for the current policy.
        :return:  None
        """
        pass

    def _resume(self):
        print("RESUMING")
        self._action_signal.set()

    def _pause(self):
        if not self._shutdown:
            self._gui.refresh_text_labels()
            self._action_signal.wait()
            self._action_signal.clear()

    def _maybe_pause(self, stage, info=None):
        """
        Caller has just finished something.  Do we need to wait for user to click the action buttion?
        If so, wait for the event.
        NOTE:  If more speed options are added in subclasses, they will neeed to handle them by overriding this method.
        TODO:  Fix this.
        """
        if self._pending_pause:
            self._pause()
            self._pending_pause = False
            return

        elif stage == 'state-update':
            if self._gui.cur_speed_option == 'state-update':
                logging.info("Pausing for state update...")
                self._pause()

            # TODO:  "Breakpoints" for specifc state updates in here

        elif stage == 'pi-round':
            if self._gui.cur_speed_option == 'pi-round':
                logging.info("Pausing for policy update...")
                self._pause()

    def _learn_loop(self):
        """
        Run "policy iteration," iterative policy evaluation. 

        """
        self._iter = 0
        while not self._shutdown:
            # 1. Update the value function for the current policy.
            self.optimize_value_function()
            self._maybe_pause('pi-round')
            if self._shutdown:
                break

            # 2. Update policy & check for convergence:
            if self.optimize_policy():
                # TODO: What do do when converged?
                logging.info("Policy converged.")

            self._maybe_pause('pi-round')

            self._iter += 1

    def optimize_policy(self, v_tol=1e-6):
        """
        1. New policy is based on current value function.

        2. In all cases:
            Check to see if policy updates are stable:
                Now: if the value function has converged (i.e. no changes).
                TODO: if the same action(s) have highest probability in all states.
        returns: True if converged
        """

        pi_new = ValueFuncPolicy(self._v)

        # Check for policy stability (recommended action list has same distribution)

        def action_dist_eq(actions_1, actions_2, tolerance=1e-6):
            """
            Check if two action distributions are equal.
            :param actions_1: list of (action, probability) tuples.
            :param actions_2: list of (action, probability) tuples.
            :return: True if equal, False otherwise.
            """
            if len(actions_1) != len(actions_2):
                return False
            for a1, a2 in zip(actions_1, actions_2):
                if (a1[0] != a2[0]):
                    raise Exception("distribution lists should be in same order...")
                if np.abs(a1[1] - a2[1]) > tolerance:
                    return False
            return True

        n_diff = 0
        for state in self.updatable_states:
            actions_old = self._pi.recommend_action(state)
            actions_new = pi_new.recommend_action(state)
            if not action_dist_eq(actions_old, actions_new):
                n_diff += 1
                self._pi.update_policy(state, actions_new)
        logging.info(f"Policy Estimation, iteration {self._iter} updated {n_diff} states .")
        return n_diff == 0  # True if no updates were made.

    def _init_gui(self):
        speed_options = self._get_speed_options()
        self._gui = RLDemoWindow(WIN_SIZE, self, speed_options, player_mark=self._player)

    def _get_speed_options(self):
        """
        Return an ordered dict of speed setting options for the radio buttons.
        keys are the names of the speed settings.
        values are a list of {'text', 'callback'} dicts for every state the app can be in for that setting where:
            text is the label on the action button,
            callback is the function to call when the button is pressed
        Each time the button is pressed, it will cycle through the list for that setting.  
        """
        options = OrderedDict()
        options['state-update'] = [{'text': 'Step state', 'callback': self._resume}]
        options['pi-round'] = [{'text': 'Step PI: V(s)', 'callback':  self._resume},
                               {'text': 'Step PI: pi(s)', 'callback':  self._resume},]
        options['continuous'] = [{'text': 'Start continuous', 'callback':  self._resume},
                                 {'text': 'Pause continuous', 'callback': lambda: self._pause},]
        return options

    def toggle_tournament(self):
        self.running_tournament = not self.running_tournament
        print("STUB:  Changed tournament run-status:  ", self.running_tournament)
        # TODO: start/stop tournament with current policy

    def start_app(self):
        """
        Start the learning loop in its own thread.
        Start the GUI (doesn't return until the GUI is closed).
        """
        self._learning_thread = Thread(target=self._learn_loop)
        self._learning_thread.start()
        self._gui.start()
        logging.info("GUI closed, waiting for learning thread to finish...")
        self._shutdown = True  # set shutdown flag to stop the learning loop.
        # set event signal in case gui is waiting for user to click a button:
        self._action_signal.set()
        self._learning_thread.join()  # wait for the learning thread to finish before exiting.
        logging.info("Learning thread finished.")


class PolicyEvaluationPIDemo(PolicyImprovementDemo):
    """
    Policy Improvement using Policy Evaluation as the value function estimator at each step.
    """
    _EXTRA_PAUSE_POINTS = ['epoch']  # Pause after every epoch of Policy Evaluation updates (all states updated once).

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, in_place=False):
        self._gui = None
        self._delta_v_tol = 1e-6  # tolerance for delta v(s) convergence.
        self._delta_v_max = 0  # running max for each epoch (epoch ends if it's < delta_v_tol)

        self._v_converged = False
        super().__init__(seed_policy, opponent_policy, player, in_place)

    def reset(self):
        super().reset()
        self._epoch = 0
        self._v_converged = False
        self._n_updated = 0
        self._delta_v_max = 0.0  # max delta v(s) for this epoch
        if self._gui is not None:
            self._gui.refresh_text_labels()

    def _get_speed_options(self):
        old_options = super()._get_speed_options()
        epoch_button_seq = [{'text': 'Step Epoch', 'callback':  self._resume}]
        options = OrderedDict()
        for i, (k, v) in enumerate(old_options.items()):
            if i == 1:
                options['epoch-update'] = epoch_button_seq
            options[k] = v
        return options

    def _maybe_pause(self, stage, info=None):
        super()._maybe_pause(stage, info)
        if stage == 'epoch-update':
            if self._gui.cur_speed_option == 'epoch-update':
                logging.info("Pausing for epoch update...")
                self._pause()

    def get_values(self):
        return self._v, self._v_new

    def get_status(self):
        status = OrderedDict()

        status['title'] = "Policy Evaluation / Improvement"
        status['phase'] = "Policy Evaluation"
        status['iteration'] = self._iter
        status['epoch'] = self._epoch
        status['states processed'] = "%i of %i" % (self._n_updated, len(self.updatable_states))
        status['delta-v(s)'] = "nmax delta %.3e (<> %.3e)." % (self._delta_v_max, self._delta_v_tol)
        status['flag'] = "V(s) Converged after %i epochs!" % self._epoch if self._v_converged else ""
        return status

    def optimize_value_function(self):
        self._epoch = 0
        self._v_converged = False
        self._maybe_pause('state-update')  # don't start running
        while not self._v_converged:
            self._v_new = self._v_terminal.copy()
            self._n_updated = 0
            for iter, state in enumerate(self.updatable_states):
                self._n_updated += 1
                self._v_new[state] = np.random.rand()

                #self._gui.add_update(SingleStateUpdate(state, new_value = self._v_new[state]))

                self._maybe_pause('state-update', info=state)
                if self._shutdown:
                    return
            self._epoch += 1
            # check for convergence:
            if np.random.rand() < 0.1:
                self._v_converged = True

            self._maybe_pause('epoch-update')
            if self._shutdown:
                return

    def _stub_optimize_value_function(self):
        """
        "Iterative policy evalutation, for estimating V ~ v_pi(s) for all states s and policy pi(s)."
        Barto & Sutton, 2020, p. 75.
        """

        while True:
            delta = 0
            self._v_new = self._v_terminal.copy()
            for state in self.updatable_states:
                # Calculate the value of the state using the current policy.
                # get action distribution for current policy:
                actions = self._pi.recommend_action(state)
                exp_val_new = 0.0
                for action, prob_action in actions:
                    new_states = self._env.get_next_state_dist(state, action)
                    discount_reward = 0
                    for new_state, new_state_prob in new_states():
                        discount_reward += new_state_prob * self._v[new_state] * self._gamma
                    exp_val_new += prob_action * discount_reward
                self._v_new[state] = exp_val_new

    def _init_visualizations(self):
        # create layout of value function visualization using hot-cold heatmap and
        # positions from the FixedCellBoxOrganizer.

        # START HERE
        space_size = 2
        artists = GameStateArtist(space_size=space_size)
        heat_colors = plt.colormaps['hot']
        # create a color map for the values:
        all_values = [self._v.values()]
        value_range = max(all_values), min(all_values)
        norm = plt.Normalize(vmin=value_range[1], vmax=value_range[0])
        cmap = plt.get_cmap(heat_colors, 256)
        # create a color map for the values:
        layers = []
        colors = {}
        for state in self._v:
            color = cmap(norm(self._v[state]))
            colors[state] = color
            layers.append(state)
        # create a color map for the values:


def run_app():
    # Example usage:
    seed_policy = HeuristicPlayer(n_rules=6, mark=Mark.X)  # Replace with your seed policy
    opponent_policy = HeuristicPlayer(n_rules=6, mark=Mark.X)  # Replace with your opponent policy
    player = Mark.X  # Replace with your player mark

    demo = PolicyEvaluationPIDemo(seed_policy, opponent_policy, player)
    demo.start_app()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_app()
