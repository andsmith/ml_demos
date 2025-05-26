from rl_alg_base import DemoAlg


class QLearningDemoAlg(DemoAlg):
    @staticmethod
    def get_name():
        return 'q'

    @staticmethod
    def get_str():
        return "Q-learning"

    def load_state(self, state_file):
        pass

    def save_state(self, state_file):
        pass

    def _reset_state(self):
        pass

    @staticmethod
    def is_stub():
        return True
