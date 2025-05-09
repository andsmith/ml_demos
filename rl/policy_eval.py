from rl_alg_base import DemoAlg
from enum import IntEnum
import pickle
from util import get_clobber_free_filename
import logging
import layout
from game_base import Mark, TERMINAL_REWARDS, get_reward
from collections import OrderedDict

class PolicyImprovementPhases(IntEnum):
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
        self._env = env
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        self._v_terminal = {state: TERMINAL_REWARDS[state.check_endstate()] for state in self.terminal_states}
        self._delta_v_tol = 1e-6
        self.pi_seed = None
        self.gamma = gamma
        self.reset_state()
        logging.info("PolicyEvalDemoAlg initialized.")

    def reset_state(self):

        # start by learning v(s) for this policy:
        self.policy = self.pi_seed

        # Policy Improvement (PI) state:
        self.pi_iter = 0
        self.pi_phase = PolicyImprovementPhases.POLICY_EVAL
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

        if self.pi_phase == PolicyImprovementPhases.POLICY_EVAL:
            status = [("PI Phase: Policy Evaluation (PE)", font_bold),
                      ("PI Iteration: {}".format(self.pi_iter) + ("(converged)"if self.pi_converged else ""), font_default),
                      ("PE Epoch: {}".format(self.pe_iter), font_default),
                      ("PE State: {} of {}".format(self.next_state_ind+1, len(self.updatable_states)), font_default),
                      ("Max Delta Vs: %.3f (max %.1e)"%(self.max_delta_vs, self._delta_v_tol), font_default)]
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
    
    def start(self):
        pass
    

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

    @staticmethod
    def get_str():
        return "(PI) In-Place Policy Evaluation"
    
    def reset_state(self):
        pass

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
