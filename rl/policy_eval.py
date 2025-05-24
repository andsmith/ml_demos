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
        super().__init__(app=app)
        self._colors = {'bg': COLOR_BG,
                        'lines': COLOR_LINES,
                        'text': COLOR_TEXT}
        self._env = env
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        self._v_terminal = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self._img_mgr = PolicyEvalSIM(self._env)
        self._delta_v_tol = 1e-6
        self.pi_seed = None
        self.gamma = gamma
        self.reset_state()
        logging.info("PolicyEvalDemoAlg initialized.")

    def _init_images(self, env):
        self._img_mgr = PolicyEvalSIM(env)
        for state in self._img_mgr.all_states:
            self._img_mgr.set_state_val(state, 'values', np.random.rand()*2 - 1)
            self._img_mgr.set_state_val(state, 'updates', np.random.rand()*2 - 1)
        self._tabs = self._img_mgr.tabs

    def reset_state(self):
        print("Class: %s resetting state" % self.__class__.__name__)
        # start by learning v(s) for this policy:
        self.policy = self.pi_seed

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

        # tables to update:
        self.values = {}
        self.next_values = {}

    def get_status(self):
        font_default = layout.LAYOUT['fonts']['status']
        font_bold = layout.LAYOUT['fonts']['status_bold']

        if self.pi_phase == PIPhases.POLICY_EVAL:
            status = [("PI Phase: Policy Evaluation (PE)", font_bold),
                      ("PI Iteration: {}".format(self.pi_iter) + ("(converged)"if self.pi_converged else ""), font_default),
                      ("PE Epoch: {}".format(self.pe_iter), font_default),
                      ("PE State: {} of {}".format(self.next_state_ind, len(self.updatable_states)), font_default),
                      ("Max Delta Vs: %.3f (max %.1e)" % (self.max_delta_vs, self._delta_v_tol), font_default)]
            if self.pe_converged:
                status += [(" Converged in %i iterations" % (self._pe_iter+1,), font_bold)]
            else:
                status += [("", font_default),]
        else:
            status += [("PI Phase: Policy Optimization (PO)", font_bold),
                       ("PI Iteration: {}".format(self.pi_iter) + ("(converged)"if self.pi_converged else ""), font_default),
                       ("", font_default),
                       ("PO State: {} of {}".format(self.next_state_ind+1, len(self.updatable_states)), font_default),
                       ("N policy changes: {}".format(self.policy), font_default)]

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
        self._maybe_pause('state-update')
        
        while not self._shutdown:

            self.pe_iter += 1
            # optimize value function for given policy
            self.pi_phase = PIPhases.POLICY_EVAL
            if self._optimize_value_function():
                return

            # optimize policy & check for convergence
            self._phase = PIPhases.POLICY_OPTIM
            if self._optimize_policy():
                return

            if self.pi_converged:
                self.pi_convergence_iter = self.pe_iter
                logging.info("Policy converged after %i iterations." % self.pi_convergence_iter)

            if self._maybe_pause('policy-update'):
                print("Paused by object %s, with type %s" % (self, type(self._pause_obj)))
                return

    def _optimize_value_function(self):
        self.pe_iter=0

        # Don't use for loops, in case user clicks reset.
        while not self._shutdown and self.pe_iter < 4:
            self.next_state_ind = 0
            while not self._shutdown and self.next_state_ind < len(self.updatable_states):
                # TODO:  Fill in here.

                self._maybe_pause('state-update') 
                self.next_state_ind += 1

            self._maybe_pause('epoch-update')         
            self.pe_iter += 1

        return self._shutdown

    def _optimize_policy(self):
        self.next_state_ind = 0
        while not self._shutdown and self.next_state_ind < len(self.updatable_states):
            self.next_state_ind += 1
            # TODO:  Fill in here.
            self._maybe_pause('state-update')

    def get_state_image(self, size, tab_name, is_paused):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._colors['bg']
        text = "Policy Evaluation State Image:  %s, paused = %s" % (tab_name, is_paused)
        cv2.putText(img, text, (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "%f" % (np.random.randn(),)
        cv2.putText(img, text, (30, 200), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        return img

    def get_viz_image(self, size, control_point, is_paused):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        img[:] = self._colors['bg']

        text = "Policy Eval step viz"
        cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "control point:  %s" % (control_point,)
        cv2.putText(img, text, (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, self._colors['text'], 1, cv2.LINE_AA)
        text = "paused = %s" % (is_paused,)
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
