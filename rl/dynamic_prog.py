"""
Calculate the value function for a given policy using dynamic programming.
Use the HeuristicPolicy as the policy to learn.
"""
from rl_alg_base import DemoAlg


class DynamicProgDemoAlg(DemoAlg):
    @staticmethod
    def get_name():
        return 'dp'

    @staticmethod
    def get_str():
        return "(PI) Dynamic Programming"


    def load_state(self, state_file):
        pass

    def save_state(self, state_file):
        pass

    def _reset_state(self):
        pass

    @staticmethod
    def is_stub():
        return True


class InPlaceDPDemoAlg(DynamicProgDemoAlg):
    @staticmethod
    def get_name():
        return 'dp-inplace'

    @staticmethod
    def get_str():
        return "(PI) In-Place Dynamic Programming"


    @staticmethod
    def is_stub():
        return True
