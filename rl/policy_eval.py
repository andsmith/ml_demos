from rl_alg_base import DemoAlg


class PolicyEvalDemoAlg(DemoAlg):
    @staticmethod
    def get_name():
        return 'pi-pe'

    @staticmethod
    def get_str():
        return "(PI) Policy Evaluation"

    def _init_frames(self):
        pass


class InPlacePEDemoAlg(PolicyEvalDemoAlg):
    @staticmethod
    def get_name():
        return 'pi-pe-inplace'

    @staticmethod
    def get_str():
        return "(PI) In-Place Policy Evaluation"

    def _init_frames(self):
        pass
