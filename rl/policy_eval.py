from rl_alg_base import DemoAlg
from enum import IntEnum
import pickle
from util import get_clobber_free_filename
import logging
import layout
from game_base import Mark, TERMINAL_REWARDS, get_reward
from collections import OrderedDict
import numpy as np
from colors import COLOR_SCHEME
import cv2
from drawing import GameStateArtist
import time
from state_key import StateKey
from color_key import SelfAdjustingColorKey, ProbabilityColorKey, get_good_cmap
from state_embedding import StateEmbedding, StateEmbeddingKey
from state_tab_content import FullStateContentPage, ValueFunctionContentPage
from gui_base import Key
from tic_tac_toe import Game
# TODO: import results viz tab
import matplotlib.pyplot as plt


class PIPhases(IntEnum):
    POLICY_EVAL = 0
    POLICY_OPTIM = 1


class PolicyEvalDemoAlg(DemoAlg):

    def __init__(self, app, pi_seed, gamma=0.9):  # ,alg_params):

        # TODO:   DemoAlg.get_options() shoudl return a dict of option types,
        # then the alg_params dict should match it here.
        """
        Initialize the algorithm with the app, GUI, environment, and discount factor.

        :param app: The application object.
        :param gui: The GUI object.
        :param env: The environment object.
        :param  alg_params:  Algorithm parameters dict:
            {'pi_seed': Policy to estimate first value function.
            'gamma': Discount factor for future rewards (default is 0.9).}

        :param gamma: The discount factor (default is 0.9).
        """
        self.app = app
        self.env = app.env
        self._embedding = self._make_embedding()
        self._state_update_order = None  # set after box_placer is defined
        self._gamma = gamma
        self._viz_img_size = None
        self._delta_v_tol = 1e-6
        self.pi_seed = pi_seed
        self.updatable_states = app.env.get_nonterminal_states()
        self.terminal_states = app.env.get_terminal_states()

        self.reset_state()
        super().__init__(app=app)

        logging.info("PolicyEvalDemoAlg initialized.")

    def resize(self, panel, new_size):
        super().resize(panel, new_size)
        if panel == 'state-tabs':
            self._embedding.set_size(new_size)
            self._state_update_order = self._get_state_update_order()
            for tab in self._tabs.values():
                tab['tab_content'].resize(new_size)
        elif panel != 'step-visualization':
            raise ValueError("Unknown panel for resizing: %s" % panel)

    def _get_state_update_order(self):
        """
        Get the order of states to update.
        :return: A list of states in the order they should be updated.
        """
        positions = self._embedding.box_placer.box_positions
        update_order = sorted(self.updatable_states, key=lambda s: positions[s]['x'][0])
        return update_order

    def reset_state(self):

        # start by learning v(s) for this policy:
        self.policy = self.pi_seed

        # Current state being updated, etc.
        self.state = None   # for both phases

        # Policy Improvement (PI) state:
        self.pi_iter = 0
        self.pi_phase = PIPhases.POLICY_EVAL
        self.pi_converged = False
        self.pi_convergence_iter = None  # iteration when policy converged (don't update)

        # Policy evaluation state (part of a PI iteration):
        self.pe_iter = 0  # epoch
        self.pe_converged = False
        self.next_state_ind = 0  # index of the next state to evaluate
        self.max_delta_vs = 0.0

        # Policy optimization state:
        self.n_pi_changes = 0  # number of states with changed policy actions
        # index of the next state to optimize (in self._state_update_order)
        self.next_state_ind = 0

        # Value function & update tables
        # initial values for terminal states (should be zero, but we're using the terminal reward):
        self.values = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self.values.update({state: 0.0 for state in self.updatable_states})
        self.next_values = {state: None for state in self.updatable_states}

    def get_status(self):
        font_default = layout.LAYOUT['fonts']['status']
        font_bold = layout.LAYOUT['fonts']['status_bold']

        if self.pi_phase == PIPhases.POLICY_EVAL:
            status = [("PI Phase: Policy Evaluation", font_bold),
                      ("PI Iteration: {}".format(self.pi_iter) + ("(converged)"if self.pi_converged else ""), font_default),
                      ("PE Epoch: {}".format(self.pe_iter), font_default),
                      ("PE State: {} of {}".format(self.next_state_ind, len(self.updatable_states)), font_default),
                      ("Max Delta Vs: %.6f " % (self.max_delta_vs, ), font_default)]
            if self.pe_converged:
                status += [("V(s) converged, iter %i" % (self.pe_iter,), font_bold)]
            else:
                status += [("", font_default),]
        else:
            status = [("PI Phase: Policy Optimization", font_bold),
                      ("PI Iteration: {}".format(self.pi_iter) + ("(converged)"if self.pi_converged else ""), font_default),
                      ("PO State: {} of {}".format(self.next_state_ind, len(self.updatable_states)), font_default),
                      ("N policy changes: {}".format(self.n_pi_changes), font_default)]
        if self.pi_converged:
            status += [("Pi(s) converged, iter %i" % (self.pi_convergence_iter,), font_bold)]
        else:
            status += [("", font_default),]

        status += [('N Stop States: %i' % len(self.app.selected), font_default),]

        return status

    @staticmethod
    def get_name():
        return 'pi-pe'

    @staticmethod
    def get_str():
        return "(PI) Policy Evaluation"

    @staticmethod
    def is_stub():
        return False

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['state-update'] = "state update"
        rco['epoch-update'] = "epoch update"
        rco['policy-update'] = "policy update"
        rco['stops'] = "selected states"
        return rco

    def _make_embedding(self):
        """
        Need the key size to determine the embedding size.
        """
        # Only include keys that go on embedding-based images.
        key_h = 90
        self._key_sizes = OrderedDict((('state', {'height': key_h-5, 'width': key_h-5}),
                                       ('embedding', {'height': key_h, 'width': 200}),
                                       ('color', {'height': key_h, 'width': 150})))
        self._x_offsets, self._key_sizes, self._total_key_size = self._calc_key_placement(self._key_sizes)
        embedding = StateEmbedding(self.app.env, key_size=self._total_key_size)
        return embedding

    def _make_tabs(self):

        state_key = StateKey(size=self._key_sizes['state'],
                             x_offset=self._x_offsets['state'])
        
        value_color_key = SelfAdjustingColorKey(size=self._key_sizes['color'],
                                                x_offset=self._x_offsets['color'],
                                                cmap=get_good_cmap())
        
        updates_color_key = SelfAdjustingColorKey(size=self._key_sizes['color'],
                                                x_offset=self._x_offsets['color'],
                                                cmap=get_good_cmap(), value_range=(-self._delta_v_tol, self._delta_v_tol))
        
        embedding_key = StateEmbeddingKey(size=self._key_sizes['embedding'],
                                          x_offset=self._x_offsets['embedding'])

        tabs = OrderedDict((('state', {'disp_text': "States",
                                       'tab_content': FullStateContentPage(self, self._embedding,
                                                                           keys=OrderedDict((('state', state_key),
                                                                                             ('embedding', embedding_key))))}),
                            ('values', {'disp_text': "Values",
                                        'tab_content': ValueFunctionContentPage(self, self._embedding, self.values,
                                                                                 self._embedding.updatable_states,
                                                                                keys=OrderedDict((('state', state_key),
                                                                                                 ('values', value_color_key))))}),
                                                                                                   
        
                            ('updates', {'disp_text': "Updates",
                                         'tab_content': ValueFunctionContentPage(self, self._embedding, self.next_values,
                                                                                 self._embedding.updatable_states,
                                                                                 keys=OrderedDict((('state', state_key),
                                                                                                   ('values', updates_color_key))))})
                                                                                                   
        ))                                                                                                                                                                           
        
        return tabs

    def get_tabs(self):
        return self._tabs

    def load_state(self, filename):
        """
        Load the state from a file.
        :param filename: The name of the file to load the state from.
        """
        with open(filename, 'rb') as f:
            self.check_file_type(f)  # check the first item is the algorithm name
            data = pickle.load(f)
        self.pi_iter = data['pi_iter']
        self.pi_phase = data['pi_phase']
        self.pe_iter = data['pe_iter']
        self.next_state_ind = data['next_state_ind']
        self.values = data['values']
        self.next_values = data['next_values']
        self.policy = data['policy']
        logging.info("")

    def save_state(self, state_file, clobber=False):
        state_filename = get_clobber_free_filename(state_file, clobber=clobber)
        with open(state_filename, 'wb') as f:
            self.mark_file_type(f)
            pickle.dump({
                'pi_iter': self.pi_iter,
                'pi_phase': self.pi_phase,
                'pe_iter': self.pe_iter,
                'next_state_ind': self.next_state_ind,
                'values': self.values,
                'next_values': self.next_values,
                'policy': self.policy
            }, f)
        logging.info(f"Saved state to {state_filename}")

    def start(self, advance_event):
        super().start(advance_event)


    def _learn_loop(self):
        self.pi_convergence_iter = -1

        self.state = Game.from_strs(["   ", "   ", "   "])  # Set a default state for the algorithm

        self._maybe_pause('state-update')  # start paused, before any learning

        while not self._shutdown:
            self.pi_iter += 1
            # optimize value function for given policy
            self.pi_phase = PIPhases.POLICY_EVAL
            if self._optimize_value_function():
                logging.info("Policy Evaluation terminated early.")
                break
            # optimize policy
            self.pi_phase = PIPhases.POLICY_OPTIM

            if self._maybe_pause('policy-update'):
                break

            finished, self.n_pi_changes = self._optimize_policy()
            if finished:
                logging.info("Policy Optimization terminated early.")
                break

            # check for convergence
            if self.n_pi_changes == 0 or self.pi_iter > 2:  # For testing
                self.pi_convergence_iter = self.pi_iter
                self.pi_converged = True
                logging.info("Policy converged after %i iterations." % self.pi_convergence_iter)
                self.app.set_control_point('state-update')  # stop running

            if self._maybe_pause('policy-update'):
                break

        logging.info("Policy Evaluation loop finished (%s)." % self.pi_phase.name)

    def stop_loop(self):
        self._shutdown = True


    def _reset_tabs(self, phase):
        logging.info("----------------------------------Resetting tabs for phase: %s" % phase.name)
        if phase == PIPhases.POLICY_EVAL:
            if 'values' in self._tabs:
                self._tabs['values']['tab_content'].reset_values(0.0)
            if 'updates' in self._tabs:
                self._tabs['updates']['tab_content'].reset_values(None)

    def _update_single_value(self, state, new_val):
        # Set S'(state) = new_val
        old_val = self.values[state]
        delta = new_val - old_val
        self.next_values[state] = new_val
        self._tabs['updates']['tab_content'].set_value(state, delta)

    def _update_values(self):
        # Copy S'(s) to S(s), reset S'(s) to None
        self.values = self.next_values
        self.next_values = {state: None for state in self.updatable_states}


    def _optimize_value_function(self):
        """
        Policy Evaluation:  Find the value function V(s) for the current policy Pi(s).

        :returns: shutdown status
        """
        self._reset_tabs(PIPhases.POLICY_EVAL)

        state_update_order = self._state_update_order  # should exist by now...

        self.pe_iter = 0
        self.pe_converged = False
        self.pe_convergence_iter = None

        # Keep running even if converged, should not change values.
        while not self._shutdown:

            if 'updates' in self._tabs:
                self._tabs['updates']['tab_content'].reset_values(None)

            # reset updates tab so blank spaces show before filling in values (because this is model-based RL).
            # for state in self.updatable_states:
            #    self._img_mgr.set_state_val(state, 'updates', 0.0)

            # One epoch:
            self.next_state_ind = 0
            while not self._shutdown and self.next_state_ind < len(state_update_order):
                
                self.state = state_update_order[self.next_state_ind]
                
                new_val = self._optimize_state_value(self.state)
                
                self._update_single_value(self.state, new_val)

                if self._maybe_pause('state-update'):
                    return self._shutdown
                self.next_state_ind += 1

            if self._shutdown:
                return True

            # for state in state_update_order:
            #    self._img_mgr.set_state_val(state, 'values', self.next_values[state])

            self.pe_converged = self._check_value_function_convergence()
            if self.pe_converged:
                logging.info("Value function converged after epoch %i." % (self.pe_iter,))
                self.pe_convergence_iter = self.pi_iter
                self.next_state_ind = 0

            self.state = None  # clear so nothing is highlighted.
            if self._maybe_pause('epoch-update'):
                return self._shutdown

            self._update_values()

            if self.pe_converged:
                return self._shutdown

            self.pe_iter += 1

        return self._shutdown

    def _optimize_state_value(self, state):
        logging.info("Optimizing value for state:\n\n%s" % state)
        old_val = self.values[state]
        delta = np.random.randn() * 0.1  # Simulate some value change
        new_val = old_val + delta
        time.sleep(0.00001)  # Simulate some processing time
        return new_val

    def _check_value_function_convergence(self):
        """
        Check if the value function has converged.
        :returns: True if converged, False otherwise.
        """
        if self.pe_iter == 1:
            return True  # For testing
        max_delta = 0.0
        for state in self.updatable_states:
            delta = abs(self.values[state] - self.next_values[state])
            if delta > max_delta:
                max_delta = delta
            if delta > self._delta_v_tol:
                return False
        return True

    def _optimize_state_policy(self, state):
        """
        Find the best policy (action with highest expected return.)
        """
        actions = state.get_actions()

        # STUB
        new_action = np.random.choice(actions)  # Randomly choose an action for now
        time.sleep(0.00001)

        return new_action
    

    def _optimize_policy(self):
        logging.info("Optimizing policy for iteration %i" % self.pi_iter)

        # Reset 'updates' tab, will be used in policy optimization as binary
        # binary, marking which states have a new action, Pi(s)=a.
        # self._img_mgr.set_range('updates', (0.0, 1.0))
        # self._img_mgr.reset_values(tabs=('updates'))
        # for state in self.updatable_states:  # show blank squares before filling in values
        #    self._img_mgr.set_state_val(state, 'updates', 0.0)

        self.n_pi_changes = 0
        self.next_state_ind = 0

        def update(state, new_action):
            old_action_dist = self.policy[state]

            changed = False
            if changed:
                self.n_pi_changes += 1

        while not self._shutdown and self.next_state_ind < len(self.updatable_states):
            self.state = self._state_update_order[self.next_state_ind]
            new_action = self._optimize_state_policy(self.state)
            update(self.state, new_action)
            if self._maybe_pause('state-update'):
                logging.info("---------Policy optimization early shutdown.")
                return self._shutdown, self.n_pi_changes

            self.next_state_ind += 1

        self.state = None  # clear so nothing is highlighted.
        return self._shutdown, self.n_pi_changes
    '''
    def get_state_image(self, size, tab_name, is_paused):
        self._img_mgr.set_size(size)
        if self.state is not None:
            self._img_mgr.set_current_state(self.state)
        else:
            self._img_mgr.set_current_state()
        img = self._img_mgr.get_tab_img(tab=tab_name, annotated=is_paused)
        return img
    '''

    def get_viz_image(self, size, control_point, is_paused):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']

        artist = GameStateArtist(40)
        if self.state is not None:
            icon = artist.get_image(self.state)
            w, h = icon.shape[1], icon.shape[0]
            x, y = 20, 600
            img[y:y+h, x:x+w] = icon

        text = "Phase: %s" % (self.pi_phase.name,)
        cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_SCHEME['text'], 1, cv2.LINE_AA)
        text = "ctrl-pt: %s" % (control_point,)
        cv2.putText(img, text, (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_SCHEME['text'], 1, cv2.LINE_AA)
        text = "paused: %s" % (is_paused,)
        cv2.putText(img, text, (10, 300), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_SCHEME['text'], 1, cv2.LINE_AA)
        text = "%f" % (np.random.randn(),)
        cv2.putText(img, text, (10, 400), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_SCHEME['text'], 1, cv2.LINE_AA)
        return img


class InPlacePEDemoAlg(PolicyEvalDemoAlg):
    def __init__(self, app, env, gamma=0.9):
        """
        Initialize the algorithm with the app, GUI, environment, and discount factor.

        :param app: The application object.
        :param gui: The GUI object.
        :param env: The environment object.
        :param gamma: The discount factor (default is 0.9).
        """

        super().__init__(app=app, env=env)
        self.gamma = gamma

    @staticmethod
    def get_name():
        return 'pi-pe-inplace'

    def get_status(self):
        font_default = layout.LAYOUT['fonts']['status']
        font_bold = layout.LAYOUT['fonts']['status_bold']

        status = super().get_status()
        status[0] = ("PI(In-Pl.) Phase: Policy Evaluation", font_bold)
        status[0] = ("PI(In-Place) Phase: Policy Evaluation", font_bold)
        return status

    @staticmethod
    def get_str():
        return "(PI) In-Place Policy Eval."

    def load_state(self, filename):
        """
        Same as iterative, but no next_values.
        """
        with open(filename, 'rb') as f:
            self.check_file_type(f)
            data = pickle.load(f)
        self.pi_iter = data['pi_iter']
        self.pi_phase = data['pi_phase']
        self.pe_iter = data['pe_iter']
        self.next_state_ind = data['next_state_ind']
        self.values = data['values']
        self.policy = data['policy']
        logging.info("")

    def save_state(self, state_file, clobber=False):
        state_filename = get_clobber_free_filename(state_file, clobber=clobber)
        with open(state_filename, 'wb') as f:
            self.mark_file_type(f)
            pickle.dump({
                'pi_iter': self.pi_iter,
                'pi_phase': self.pi_phase,
                'pe_iter': self.pe_iter,
                'next_state_ind': self.next_state_ind,
                'values': self.values,
                'policy': self.policy
            }, f)
        logging.info(f"Saved state to {state_filename}")
