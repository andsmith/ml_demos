"""
Tic-Tac-Toe Reinforcement Learning demo app.

Main window is three pannels, two large images side-by-side, and a flat control panel at the bottom.
Two image panels at the top are:
  - value function V_t(S) at time t, and
  - current delta-V_t(s), pending/recent updates to make V_{t+1}(s).
The image of a function of s, a game state is reprsented as a game-state tree with single colored
boxes instead of game state images, the color of a "state" indicating the value.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from drawing import GameStateArtist
from tic_tac_toe import Game, get_game_tree_cached
from node_placement import FixedCellBoxOrganizer
from game_base import Result, Mark, TERMINAL_REWARDS, get_reward
from reinforcement_base import Environment, PIPhases
from abc import ABC, abstractmethod
from policy_optim import ValueFuncPolicy
import logging
from baseline_players import HeuristicPlayer
from collections import OrderedDict
from gui_components import RLDemoWindow
from threading import Thread, Lock, Event
from enum import IntEnum
from step_visualizer import StateUpdateStep, EpochStep, PIStep, ContinuousStep

WIN_SIZE = (1920, 990)


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
        self._iter = 0  # of policy imporovement (PE/PO) cycles
        self._max_iter = 100  # of policy improvement.
        self._player = player
        self._n_updated = 0  # states updated this epoch
        self._convergence_iter = None

        # app state:
        self._phase = PIPhases.POLICY_EVAL  # current phase of the algorithm
        self._pending_pause = False
        self.running_tournament = False
        self.running_continuous = False
        self._converged = False

        self._seed_p = seed_policy
        self._opponent_p = opponent_policy
        self._gamma = gamma  # discount factor for future rewards.

        # P.I. initialization:
        self._env = Environment(opponent_policy, player)
        with open('env_x.pkl', 'wb') as f:
            pickle.dump(self._env, f)
        # with open('env_x.pkl', 'rb') as f:
        #    self._env = pickle.load(f)
        self.children = self._env.children

        self._pi = seed_policy
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        self._v_terminal = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
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
        :return:  list of step updates (StateUpdateStep) for the GUI to display.
        """
        pass

    def _resume(self):
        print("RESUMING")
        self._action_signal.set()

    def _pause(self, vis_update=None):

        if vis_update is not None:
            self._gui.annotate_frame(vis_update)

        if not self._shutdown:
            print("PAUSING!")
            self._gui.refresh_text_labels()
            self._action_signal.wait()
            self._action_signal.clear()

    def _maybe_pause(self, stage, vis_updates):
        """
        Caller has just finished something.  Do we need to display something and wait for user to click the action buttion?
        If so, update the GUI then pause.

        NOTE:  If more speed options are added in subclasses, they will neeed to handle them by overriding this method.
        TODO:  Fix this.
        """
        if self._pending_pause:
            self._pause(vis_updates)
            self._pending_pause = False
            return

        elif stage == 'state-update':
            if self._gui.cur_speed_option == 'state-update':
                logging.info("Pausing for state update...")
                self._pause(vis_updates)

            # TODO:  "Breakpoints" for specifc state updates in here

        elif stage == 'pi-round':
            if self._gui.cur_speed_option == 'pi-round':
                logging.info("Pausing for policy update...")
                self._pause(vis_updates)
        else:
            print("________running continuously__________NO PAUSE!")

    def _learn_loop(self):
        """
        Run "policy improvement," iterative policy evaluation / policy optimization. 

        """
        self._iter = 0

        while not self._shutdown:

            # 1. Update the value function for the current policy.
            self._phase = PIPhases.POLICY_EVAL
            state_updates = self.optimize_value_function()

            if self._shutdown:
                break

            vis = PIStep(self, self._gui, PIPhases.POLICY_EVAL, info={'state_updates': state_updates})
            self._maybe_pause('pi-round', vis)

            if self._shutdown:
                break

            # 2. Update policy & check for convergence:
            self._phase = PIPhases.POLICY_OPT
            new_policy, self._converged = self.optimize_policy()
            
            if self._shutdown:
                break
            
            vis = PIStep(self, self._gui, PIPhases.POLICY_OPT, info={'old':  self._pi, 'new': new_policy})
            self._maybe_pause('pi-round', vis)

            self._iter += 1

    def optimize_policy(self):
        """
        1. New policy is based on current value function.

        2. In all cases:
            Check to see if policy updates are stable:
                Now: if the value function has converged (i.e. no changes).
                TODO: if the same action(s) have highest probability in all states.
        returns: True if converged
        """

        pi_new = ValueFuncPolicy(self._v)
        converged = pi_new.equals(self._pi)
        return pi_new, converged

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
        # let the gui call start_learn_loop after it knows where everything goes
        self._gui.start()
        logging.info("GUI closed, waiting for learning thread to finish...")
        self._shutdown = True  # set shutdown flag to stop the learning loop.
        # set event signal in case gui is waiting for user to click a button:
        self._action_signal.set()
        self._learning_thread.join()  # wait for the learning thread to finish before exiting.
        logging.info("Learning thread finished.")

    def start_learn_loop(self):
        self._learning_thread = Thread(target=self._learn_loop)
        self._learning_thread.start()



class PolicyEvaluationPIDemo(PolicyImprovementDemo):
    """
    Policy Improvement using Policy Evaluation as the value function estimator at each step.
    """
    _EXTRA_PAUSE_POINTS = ['epoch']  # Pause after every epoch of Policy Evaluation updates (all states updated once).

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, in_place=False):
        self._gui = None
        self._epoch = 0  # passes through all states

        self._delta_v_tol = 1e-6  # tolerance for delta v(s) convergence.
        self._delta_v_max = 0  # running max for each epoch (epoch ends if it's < delta_v_tol)

        self._v_converged = False
        super().__init__(seed_policy, opponent_policy, player, in_place)
        self._state_update_order = None

    def _set_state_update_order(self):
        """
        Sort by x-coordinate from gui's box positions so they update from left to right, top to bottom.
        """
        positions = self._gui.get_box_positions()
        # import ipdb; ipdb.set_trace()
        self._state_update_order = sorted(self.updatable_states, key=lambda s: positions[s]['x'][0])

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

    def _maybe_pause(self, stage, vis_updates):
        super()._maybe_pause(stage, vis_updates)
        if stage == 'epoch-update':
            if self._gui.cur_speed_option == 'epoch-update':
                logging.info("Pausing for epoch update...")
                self._pause(vis_updates)

    def get_values(self):
        return self._v, self._v_new

    def get_status(self):
        status = OrderedDict()
        print("Getting status...")

        status['title'] = "Policy Improvement Demo"
        status['PI Phase'] = "Policy Evaluation" if self._phase == PIPhases.POLICY_EVAL else "Policy Optimization"
        status['PI Iteration'] = self._iter
        status['PI Convergence'] = "YES (%i iter)" % (self._convergence_iter,) if self._converged else "no"
        status['PE Epoch'] = self._epoch
        status['States processed'] = "%i of %i" % (self._n_updated, len(self.updatable_states))
        status['Max delta-v(s)'] = "%.2e (max %.2e)." % (self._delta_v_max, self._delta_v_tol)
        status['flag'] = "V(s) Converged after %i epochs!" % self._epoch if self._v_converged else ""
        return status

    def optimize_value_function(self):
        """
        "Iterative policy evalutation, for estimating V ~ v_pi(s) for all states s and policy pi(s)."
        Barto & Sutton, 2020, p. 75.
        """
        logging.info("Starting Policy Evaluation, (PI round %i)" % self._iter)
        self._epoch = 0

        if self._state_update_order is None:
            self._set_state_update_order()
        while True:
            logging.info("Starting epoch %i" % self._epoch)
            self._v_new = self._v_terminal.copy()
            self._delta_v_max = 0.0  # max delta v(s) for this epoch
            state_updates = []  # list of state updates for the GUI to display.
            for iter, state in enumerate(self._state_update_order):
                #logging.info("Updating state:\n%s" % state)

                def get_reward_term_and_next_states(action):
                    """
                    The reward term is the bracketed expression in V_{pi'}(s) update eqn. on page 79 of Sutton & Barto, the 
                      reward for taking the action plus the discounted value of the next state.
                    """
                    next_state = state.clone_and_move(action, self._player)
                    next_result = next_state.check_endstate()
                    if next_result in [self._env.winning_result, self._env.losing_result, self._env.draw_result]:
                        # if the next state is a terminal, we got a reward so just return that.
                        return TERMINAL_REWARDS[next_result], [(next_state, 1.0)]
                    else:
                        # opponent's turn, get the probability distribution of next states s'
                        opp_move_dist = self._env.pi_opp.recommend_action(next_state)
                        # take the weighted sum over every action the opponent might make of v(s')
                        reward_term = 0.0
                        next_states = []
                        for opp_action, prob in opp_move_dist:
                            if prob==0:
                                # TODO: Don't include zero probability actions in the next state distribution.
                                continue
                            next_next_state = next_state.clone_and_move(opp_action, self._env.opponent)
                            next_states.append((next_next_state, prob))
                            next_next_result = next_next_state.check_endstate()
                            if next_next_result in [self._env.winning_result, self._env.losing_result, self._env.draw_result]:
                                reward_term += prob * TERMINAL_REWARDS[next_next_result]
                            else:
                                reward_term += prob * self._v[next_next_state] * self._gamma
                    return reward_term, next_states

                actions = state.get_actions()
                reward_terms, next_states = [],[]

                for action in actions:
                    reward_term, next_state_dist = get_reward_term_and_next_states(action)
                    reward_terms.append(reward_term)
                    next_states.append(next_state_dist)

                self._v_new[state] = np.max(reward_terms)

                # Handle visualizer updates
                vis = StateUpdateStep(self, self._gui, state, actions, next_states, reward_terms,
                                      self._v[state], self._v_new[state])
                state_updates.append(vis)

                self._n_updated = iter + 1
                self._maybe_pause('state-update', vis)
                if self._shutdown:
                    return

            self._epoch += 1
            # check for convergence:
            if np.random.rand() < 0.1:
                self._v_converged = True

            vis = EpochStep(self, self._gui, 'epoch', state_updates)
            self._maybe_pause('epoch-update', vis)

            if self._shutdown:
                return

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
    seed_policy = HeuristicPlayer(n_rules=2, mark=Mark.X)  # Replace with your seed policy
    opponent_policy = HeuristicPlayer(n_rules=6, mark=Mark.X)  # Replace with your opponent policy
    player = Mark.X  # Replace with your player mark

    demo = PolicyEvaluationPIDemo(seed_policy, opponent_policy, player)
    demo.start_app()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_app()
