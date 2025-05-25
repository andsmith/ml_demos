from rl_alg_base import DemoAlg
from enum import IntEnum
import pickle
from util import get_clobber_free_filename
import logging
import layout
from game_base import Mark, TERMINAL_REWARDS, get_reward
from collections import OrderedDict
import numpy as np
from state_image_manager import PolicyEvalSIM
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
import cv2
from drawing import GameStateArtist
import time


class PIPhases(IntEnum):
    POLICY_EVAL = 0
    POLICY_OPTIM = 1


class PolicyEvalDemoAlg(DemoAlg):

    def __init__(self, app, env,  gamma=0.9):
        """
        Initialize the algorithm with the app, GUI, environment, and discount factor.

        :param app: The application object.
        :param gui: The GUI object.
        :param env: The environment object.
        :param gamma: The discount factor (default is 0.9).
        """
        super().__init__(app=app, env=env)
        self._colors = {'bg': COLOR_BG,
                        'lines': COLOR_LINES,
                        'text': COLOR_TEXT}

        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()

        self._delta_v_tol = 1e-6
        self.pi_seed = None
        self.gamma = gamma
        self.reset_state()

        logging.info("PolicyEvalDemoAlg initialized.")

    def _make_state_image_manager(self):
        return PolicyEvalSIM(self._env)

    def reset_state(self):
        # start by learning v(s) for this policy:
        self.policy = self.pi_seed

        self.state = None
        # Policy Improvement (PI) state:
        self.pi_iter = 0
        self.pi_phase = PIPhases.POLICY_EVAL
        self.pi_converged = False

        # Policy evaluation state (part of a PI iteration):
        self.pe_iter = 0  # epoch
        self.pe_converged = False
        self.next_state_ind = 0  # index of the next state to evaluate
        self.max_delta_vs = 0.0

        # Policy optimization state:
        self.n_changes = 0

        # initial values for terminal states (should be zero, but we're using the terminal reward):
        self.values = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self.values.update({state: 0.0 for state in self.updatable_states})

        # Reset the image manager with the initial values.
        self._img_mgr.reset_values()

        for state, value in self.values.items():
            self._img_mgr.set_state_val(state, 'values', value)
        for state in self.updatable_states:
            self._img_mgr.set_state_val(state, 'updates', 0.0)

        self.next_values = {}

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
                      ("N policy changes: {}".format(self.n_changes), font_default)]
        if self.pi_converged:
            status += [("Pi(s) converged, iter %i" % (self.pi_convergence_iter,), font_bold)]
        else:
            status += [("", font_default),]

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

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['state-update'] = "state update"
        rco['epoch-update'] = "epoch update"
        rco['policy-update'] = "policy update"
        return rco

    def _learn_loop(self):
        self.pi_convergence_iter = -1
        self._maybe_pause('state-update')  # start paused, before any learning

        while not self._shutdown:

            self.pi_iter += 1
            # optimize value function for given policy
            self.pi_phase = PIPhases.POLICY_EVAL
            if self._optimize_value_function():
                logging.info("Policy Evaluation terminated early.")
                break

            if self._maybe_pause('policy-update'):
                break

            # optimize policy
            self.pi_phase = PIPhases.POLICY_OPTIM
            finished, self.n_changes = self._optimize_policy()
            if finished:
                logging.info("Policy Optimization terminated early.")
                break

            # check for convergence
            if self.n_changes == 0 or self.pi_iter > 2:  # For testing
                self.pi_convergence_iter = self.pi_iter
                self.pi_converged = True
                logging.info("Policy converged after %i iterations." % self.pi_convergence_iter)
                self.app.set_control_point('state-update')  # stop running

            if self._maybe_pause('policy-update'):
                break

        logging.info("Policy Evaluation loop finished (%s)." % self.pi_phase.name)

    def _optimize_value_function(self):
        """
        :returns: shutdown status
        """
        self._img_mgr.set_range('updates', (-1.0, 1.0))
        self._img_mgr.reset_values(tabs=('updates'))

        def update(state, value):
            old_val = self.values[state]
            delta = value - old_val
            self.next_values[state] = value
            self._img_mgr.set_state_val(state, 'updates', delta)

        state_update_order = self._img_mgr.get_state_update_order()

        self.pe_iter = 0
        self.pe_converged = False
        self.pe_convergence_iter = None

        # Don't use for-loops, in case user clicks reset pe_iter, next_state_ind need to start again at zero.
        while not self._shutdown:

            # reset epoch state
            self.next_state_ind = 0
            for state in self.updatable_states:
                self._img_mgr.set_state_val(state, 'updates', 0.0)

            while not self._shutdown and self.next_state_ind < len(state_update_order):

                # TODO:  Fill in here.
                self.state = state_update_order[self.next_state_ind]
                delta = np.random.randn() * 0.1  # Simulate some value change
                old_val = self.values[self.state]
                new_val = old_val + delta
                update(self.state, new_val)

                if self._maybe_pause('state-update'):
                    return self._shutdown

                self.next_state_ind += 1
                time.sleep(0.000001)

            if self._shutdown:
                return True

            for state in state_update_order:
                self._img_mgr.set_state_val(state, 'values', self.next_values[state])

            self.pe_converged = self._check_value_function_convergence()
            if self.pe_converged:
                logging.info("Value function converged after epoch %i." % (self.pe_iter,))
                self.pe_convergence_iter = self.pi_iter
                self.next_state_ind = 0

                # self.pi_phase = PIPhases.POLICY_OPTIM # change now so it displays in status during epoch update

            if self._maybe_pause('epoch-update'):
                return self._shutdown

            self._img_mgr.reset_values(tabs=('updates',))
            self.values = self.next_values
            self.next_values = {}

            if self.pe_converged:
                return self._shutdown

            self.pe_iter += 1

        return self._shutdown

    def _check_value_function_convergence(self):
        """
        Check if the value function has converged.
        :returns: True if converged, False otherwise.
        """
        if self.pe_iter == 1:
            return True  # For testing
        for state in self.updatable_states:
            delta = abs(self.values[state] - self.next_values[state])
            if delta > self.max_delta_vs:
                self.max_delta_vs = delta
            if delta > self._delta_v_tol:
                return False
        return True

    def _optimize_policy(self):
        logging.info("Optimizing policy for iteration %i" % self.pi_iter)
        self.next_state_ind = 0

        # binary, marking which states have a new best action
        self._img_mgr.set_range('updates', (0.0, 1.0))
        self._img_mgr.reset_values(tabs=('updates'))

        def update(state, new_action):

            changed = False if self.pi_iter > 2 else np.random.rand() < 0.333  # for testing, simulate some changes
            if changed:
                self.n_changes += 1
                self._img_mgr.set_state_val(state, 'updates', 1.0)
            else:
                self._img_mgr.set_state_val(state, 'updates', 0.0)

        state_update_order = self._img_mgr.get_state_update_order()

        while not self._shutdown and self.next_state_ind < len(self.updatable_states):
            # TODO:  Fill in here.
            self.state = state_update_order[self.next_state_ind]

            # Simulate some policy change
            update(self.state, None)

            if self._maybe_pause('state-update'):
                logging.info("---------Policy optimization early shutdown.")
                return self._shutdown, self.n_changes

            time.sleep(0.00001)
            self.next_state_ind += 1

        return self._shutdown, self.n_changes

    def get_state_image(self, size, tab_name, is_paused):
        self._img_mgr.set_size(size)
        img = self._img_mgr.get_tab_img(tab=tab_name, annotated=is_paused)
        return img

    def get_viz_image(self, size, control_point, is_paused):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._colors['bg']

        artist = GameStateArtist(40)
        if self.state is not None:
            icon = artist.get_image(self.state)
            w, h = icon.shape[1], icon.shape[0]
            x, y = 20, 600
            img[y:y+h, x:x+w] = icon

        text = "Phase: %s" % (self.pi_phase.name,)
        cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "ctrl-pt: %s" % (control_point,)
        cv2.putText(img, text, (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "paused: %s" % (is_paused,)
        cv2.putText(img, text, (10, 300), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "%f" % (np.random.randn(),)
        cv2.putText(img, text, (10, 400), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        return img

    @staticmethod
    def get_state_tab_info():
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        st = OrderedDict((('states', 'Game states  '),
                          ('values', 'Values: V(s)'),
                          ('updates', "Updates: delta V(s)"),))
        return st


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
