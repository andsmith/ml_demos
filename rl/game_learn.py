"""
Tic-Tac-Toe Reinforcement Learning demo app.

Main window is three pannels, two large images side-by-side, and a flat control panel at the bottom.
Two image panels at the top are:
  - value function V_t(S) at time t, and
  - current delta-V_t(s), pending/recent updates to make V_{t+1}(s).
The image of a function of s, a game state is reprsented as a game-state tree with single colored
boxes instead of game state images, the color of a "state" indicating the value.
"""
from loop_timing.loop_profiler import LoopPerfTimer as LPT
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
import time
from step_visualizer import StateUpdateStep, EpochStep, PIStep, ContinuousStep
from value_panel import sort_states_into_layers
WIN_SIZE = (1920, 990)

FPS = 0.50


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
        logging.info("Initializing Policy Improvement demo...")
        self._player = player

        self._seed_p = seed_policy
        self._opponent_p = opponent_policy
        self._gamma = gamma  # discount factor for future rewards.
        self._learning_thread = None
        # P.I. initialization:
        self._env = Environment(opponent_policy, player)
        with open('env_x.pkl', 'wb') as f:
            pickle.dump(self._env, f)
        # with open('env_x.pkl', 'rb') as f:
        #    self._env = pickle.load(f)
        self.children = self._env.children

        self.pi = seed_policy
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        self._v_terminal = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self._v = None  # dict from state to value function V_t (s)
        self._v_new = None  # dict from state to new value function V_{t+1} (s)

        self.reset()

        self.shutdown = False
        # App  & graphics:

        self._action_signal = Event()  # learner will wait for this, action button will set it.
        self._init_gui()
        self._states_by_layer = None  # for passing to visualizer  (init later)

    def reset(self):
        logging.info("Learn app reset.")
        # initial value function parts:
        self._v = {state: 0.0 for state in self.updatable_states}
        self._v.update(self._v_terminal)
        self._v_new = {state: 0.0 for state in self.updatable_states}
        self._v_new.update(self._v_terminal)

        # App state:
        self._convergence_iter = None
        self._n_updated = 0  # states updated this epoch
        self._iter = 0  # of policy imporovement (PE/PO) cycles
        self._max_iter = 100  # of policy improvement.
        self._pending_pause = False
        self._converged = False
        self._convergence_iter = None
        self._phase = PIPhases.VALUE_F_OPT  # current phase of the algorithm
        self._n_updated = 0  # states updated this epoch
        self.started = False
        self.running_tournament = False
        self.running_continuous = False

        # Timing
        self._t_last_updated = time.perf_counter()
        self._t_last_txt_updated = time.perf_counter()

    @abstractmethod
    def optimize_value_function(self):
        """
        Update the value function for the current policy.
        :return:  list of step updates (StateUpdateStep) for the GUI to display.
        """
        pass

    def _resume(self):
        # print("RESUMING")
        if not self.started:
            self.start_learn_loop()
            self.started = True
        print("###################### SET RESUM EVENT")
        self._action_signal.set()

    def _pause(self, vis_update):
        """
        Draw stuff to window before waiting for action signal.
        """

        if vis_update is not None:
            self._gui.annotate_frame(vis_update)  # update state/value/update images in GUI for display
            step_viz_img = vis_update.draw_step_viz()  # Make the step image
            self._gui.update_step_viz_image(step_viz_img)  # update step image in GUI
        else:
            # raise Exception("Paused without a ")
            self._gui.refresh_text_labels()

            pass

        if not self.shutdown:
            # print("PAUSING!")
            self._gui.refresh_text_labels()
            self._action_signal.wait()
            print("---------------------- CLEARING RESUME EVENT (WILL PAUS ON NEXT WAIT)")
            self._action_signal.clear()

        if vis_update is not None:
            vis_update.post_viz_cleanup()

    @LPT.time_function
    def maybe_pause(self, stage, vis_update, cont_update=True):
        """
        Caller has just finished something. 
        If we need to pause, do so and create temporary images for the GUI.

        :param stage:  The stage of the algorithm that just finished (e.g. 'state-update', 'pi-round', 'epoch-update').
        :param vis_update:  The visualization object to use for the pause.
        :param cont_update:  If True, update the screen even if we didn't pause (only false if called by someone overriding this method).
        :return:  True if we paused, False otherwise.

        NOTE:  If more speed options are added in subclasses:
          1. override this method, call super().maybe_pause(...,cont_update=False)
        """

        if stage == 'state-update':
            if self._gui.cur_speed_option == 'state-update':
                logging.info("Pausing for state update...")
                self._pause(vis_update)
                return True

        elif stage == 'pi-round':
            if self._gui.cur_speed_option == 'pi-round':
                logging.info("Pausing for policy update...")
                self._pause(vis_update)
                return True

        if cont_update:
            self._didnt_pause()

        return False

    @LPT.time_function
    def _didnt_pause(self):
        # Not paused, need to update screen anyway
        now = time.perf_counter()

        if now - self._t_last_updated > 1.0 / FPS:
            self._t_last_updated = now
            self._gui.update_step_viz_image()  # clear step vis for continuous mode.
            self._gui.refresh_text_labels()  # update text labels in GUI for display
            self._gui.refresh_continuous()
            # make sure action signal is clear, so we don't block the next action:
            if self._action_signal.is_set():
                self._action_signal.clear()
        elif now - self._t_last_txt_updated > .5:
            self._gui.refresh_text_labels()  # update text labels faster
            self._t_last_txt_updated = now

    def _init_layers(self):
        # set self._states_by_layer, from the gui
        sl = self._gui.states_by_layer
        self._states_by_layer = [[sl[lay][ind]['id'] for ind in range(len(sl[lay]))] for lay in range(6)]

    def _learn_loop(self):
        """
        Run "policy improvement," iterative policy evaluation / policy optimization. 

        """
        self._iter = 0

        self.maybe_pause('state-update', None)

        self._init_layers()
        while not self.shutdown:

            # 1. Update the value function for the current policy.
            self._phase = PIPhases.VALUE_F_OPT
            self.optimize_value_function()

            if self.shutdown:
                break

            # vis = PIStep(self, self._gui, PIPhases.VALUE_F_OPT, info={'state_updates': state_updates})
            # self.maybe_pause('pi-round', vis)

            if self.shutdown:
                break

            # 2. Update policy & check for convergence:
            self._phase = PIPhases.POLICY_OPT
            #import pickle
            #with open('value_func.pkl', 'wb') as f:
            #    pickle.dump(self._v, f)

            import ipdb
            ipdb.set_trace()
            new_policy, self._converged = self.optimize_policy()

            if self.shutdown:
                break

            vis = PIStep(self, self._gui, PIPhases.POLICY_OPT, info={'old':  self.pi, 'new': new_policy})
            self.maybe_pause('pi-round', vis)

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
        pi_new = ValueFuncPolicy(self, self._v, self._env, gamma=self._gamma, player=self._player)
        converged = pi_new.equals(self.pi)
        return pi_new, converged

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
        # clear action event to start paused
        self._action_signal.clear()

        self._gui.start()

        logging.info("GUI closed, waiting for learning thread to finish...")
        # set shutdown flag to stop the learning loop
        self.shutdown = True

        # set event signal in case gui is waiting for user to click a button:
        print("###################### SET RESUME EVENT ")
        self._action_signal.set()
        if self._learning_thread is not None:
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

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, gamma=0.9):
        self._gui = None
        self._epoch = 0  # passes through all states

        self._delta_v_tol = 1e-6  # tolerance for delta v(s) convergence.
        self._delta_v_max = 0  # running max for each epoch (epoch ends if it's < delta_v_tol)
        self.v_converged = False  # True if the value function converged (i.e. no changes).

        super().__init__(seed_policy, opponent_policy, player, gamma=gamma)

        self._state_update_order = None

    def _set_state_update_order(self):
        """
        Sort by x-coordinate from gui's box positions so they update from left to right, top to bottom.
        """
        positions = self._gui.get_box_positions()
        self._state_update_order = sorted(self.updatable_states, key=lambda s: positions[s]['x'][0])

    def reset(self):
        super().reset()
        self._epoch = 0
        self.v_converged = False
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

    @LPT.time_function
    def maybe_pause(self, stage, vis_update, cont_update=True):
        if super().maybe_pause(stage, vis_update, cont_update=False):
            return True

        elif stage == 'epoch-update':

            if self._gui.cur_speed_option == 'epoch-update':
                logging.info("Pausing for epoch update...")
                self._pause(vis_update)

        if cont_update:
            self._didnt_pause()

        return False
    def get_gui(self):
        return self._gui
    def get_values(self):
        return self._v, self._v_new

    def get_status(self):
        status = OrderedDict()

        status['title'] = "Policy Improvement Demo"
        status['PI Phase'] = "Policy Evaluation" if self._phase == PIPhases.VALUE_F_OPT else "Policy Optimization"
        status['PI Iteration'] = self._iter + 1
        status['PI Convergence'] = "YES (%i iter)" % (self._convergence_iter,) if self._converged else "no"
        status['PE Epoch'] = self._epoch + 1
        status['States processed'] = "%i of %i" % (self._n_updated, len(self.updatable_states))
        status['Max delta V(s)'] = "%.3f (max %.1e)." % (self._delta_v_max, self._delta_v_tol)
        status['flag'] = "V(s) Converged after %i epochs!" % self._epoch if self.v_converged else ""
        return status

    def _init_empty_v(self):
        v_empty = self._v_terminal.copy()
        return v_empty

    def optimize_value_function(self):
        """
        "Iterative policy evalutation, for estimating V ~ v_pi(s) for all states s and policy pi(s)."
        Barto & Sutton, 2020, p. 75.
        """
        logging.info("Starting Policy Evaluation, (PI round %i)" % self._iter)
        self._epoch = 0

        if self._state_update_order is None:
            self._set_state_update_order()

        self._v_new = self._init_empty_v()
        verbose = False

        LPT.reset(enable=False, burn_in=100, display_after=30, save_filename=None)

        while not self.shutdown and not self.v_converged:
            logging.info("Starting epoch %i" % self._epoch)
            self._delta_v_max = 0.0  # max delta v(s) for this epoch

            for iter, state in enumerate(self._state_update_order):

                LPT.mark_loop_start()
                # logging.info("Updating state:\n%s" % state)
                #if state == Game.from_strs(["XOO", "X  ", "   "]):
                #    verbose=True
                #    import ipdb; ipdb.set_trace()

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
                        opp_move_dist = self._env.opp_move_dist(state, action)
                        # take the weighted sum over every action the opponent might make of v(s')
                        reward_term = 0.0
                        next_states = []
                        for opp_action, prob in opp_move_dist:
                            if prob == 0:
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

                actions = self.pi.recommend_action(state)
                reward_terms, next_states = [], []

                bare_actions = [action for action, act_prob in actions]
                for action, act_prob in actions:
                    reward_term, next_state_dist = get_reward_term_and_next_states(action)
                    reward_term *= act_prob  # scale by the probability of taking this action.
                    if verbose:
                        print("Action: ", action, "Reward term: ", reward_term)
                        for next_state, prob in next_state_dist:
                            print("\tNext state:\n", next_state, "\nProbability: ", prob, "\n")

                    reward_terms.append(reward_term*act_prob)
                    next_states.append(next_state_dist)

                self._v_new[state] = np.max(reward_terms)

                delta_v = np.abs(self._v_new[state] - self._v[state])
                self._delta_v_max = max(self._delta_v_max, delta_v)

                # Handle visualizer updates
                vis = StateUpdateStep(self, self._gui, state, bare_actions, next_states, reward_terms,
                                      self._v, self._v_new[state])

                self._n_updated = iter + 1
                self.maybe_pause('state-update', vis)
                if self.shutdown:
                    return
                verbose = False

            self._epoch += 1

            # check for convergence:
            if self._delta_v_max < self._delta_v_tol:
                self.v_converged = True
                self._phase = PIPhases.POLICY_OPT  # change this now so the status msg is correct.

            # import ipdb; ipdb.set_trace()

            # now swap the new and old value functions:
            v_old = self._v
            self._v = self._v_new
            self._v_new = self._init_empty_v()

            # and update the gui
            vis = EpochStep(self, self._gui, v_old, self._v, self._epoch, self._states_by_layer)
            self.maybe_pause('epoch-update', vis)
            self._gui.build_images()
            self._gui.refresh_text_labels()
            self._gui.refresh_images()

            # and clear the action signal so we don't block the next action:
            if self._action_signal.is_set():
                self._action_signal.clear()


def run_app():
    # Example usage:
    seed_policy = HeuristicPlayer(n_rules=2, mark=Mark.X)  # Replace with your seed policy
    opponent_policy = HeuristicPlayer(n_rules=6, mark=Mark.O)  # Replace with your opponent policy
    player = Mark.X  # Replace with your player mark

    demo = PolicyEvaluationPIDemo(seed_policy, opponent_policy, player)
    demo.start_app()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_app()
