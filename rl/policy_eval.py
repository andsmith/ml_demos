from rl_alg_base import DemoAlg


class PolicyEvalDemoAlg(DemoAlg):

    def __init__(self, app, gamma=0.9):
        """
        Initialize the algorithm with the app, GUI, environment, and discount factor.

        :param app: The application object.
        :param gui: The GUI object.
        :param env: The environment object.
        :param gamma: The discount factor (default is 0.9).
        """
        super().__init__(app=app)
        self.gamma = gamma

    @staticmethod
    def get_name():
        return 'pi-pe'

    @staticmethod
    def get_str():
        return "(PI) Policy Evaluation"

    def _init_frames(self):
        pass
    @staticmethod
    def is_stub():
        return False

class InPlacePEDemoAlg(PolicyEvalDemoAlg):
    def __init__(self, app, gamma=0.9):
        """
        Initialize the algorithm with the app, GUI, environment, and discount factor.

        :param app: The application object.
        :param gui: The GUI object.
        :param env: The environment object.
        :param gamma: The discount factor (default is 0.9).
        """
        super().__init__(app=app)
        self.gamma = gamma

    @staticmethod
    def get_name():
        return 'pi-pe-inplace'

    @staticmethod
    def get_str():
        return "(PI) In-Place Policy Evaluation"
