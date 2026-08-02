"""
Dynamic-programming demo algorithms.

Both variants reuse the PolicyEvalDemoAlg machinery (tabs, run-control,
policy-improvement phase) but replace the policy-expectation backup with the
optimality (max) backup:

    V(s) = max_a E[ R + gamma * V(s') | s, a ]

so they compute the optimal value function directly instead of evaluating the
current policy.  They differ in update order:

- DynamicProgDemoAlg: classic backward induction.  States are swept from most
  marks to fewest (children before parents) with in-place commits, so every
  backup reads already-final successor values and V*(s) is exact after a
  single pass.  A second pass shows max |dV| = 0, and greedy policy
  improvement then converges immediately.

- InPlaceDPDemoAlg: asynchronous value iteration.  Same max backup, in-place
  commits, but in the default left-to-right screen order — demonstrating that
  in-place updates converge without the "right" ordering, just in more passes.
"""
import numpy as np

from policy_eval import PolicyEvalDemoAlg


class DynamicProgDemoAlg(PolicyEvalDemoAlg):
    """
    Backward-induction dynamic programming: optimality backups, children
    before parents, committed in place.
    """

    def __init__(self, app, env, pi_seed=None, gamma=0.9):
        super().__init__(app=app, env=env, pi_seed=pi_seed, gamma=gamma)
        self._in_place = True

    def _optimize_state_value(self, state):
        """
        Optimality (max) backup: V(s) = max_a E[ R + gamma*V(s') | s, a ].
        """
        return max(self._expected_return(state, action) for action in state.get_actions())

    def _get_state_update_order(self):
        """
        Deepest states (most marks) first, so every backup reads final values;
        ties broken by screen x-position for a readable sweep.
        """
        positions = self._embedding.box_placer.box_positions
        return sorted(self.updatable_states,
                      key=lambda s: (-np.count_nonzero(np.asarray(s.state)), positions[s]['x'][0]))

    @staticmethod
    def get_name():
        return 'dp'

    @staticmethod
    def get_str():
        return "(PI) Dynamic Programming"

    @staticmethod
    def is_stub():
        return False


class InPlaceDPDemoAlg(DynamicProgDemoAlg):
    """
    Asynchronous value iteration: optimality backups committed in place, in
    the default (screen-order) sweep instead of backward induction.
    """

    def _get_state_update_order(self):
        # Default screen order (see PolicyEvalDemoAlg), not backward induction.
        return PolicyEvalDemoAlg._get_state_update_order(self)

    @staticmethod
    def get_name():
        return 'dp-inplace'

    @staticmethod
    def get_str():
        return "(PI) In-Place Dynamic Prog."

    @staticmethod
    def is_stub():
        return False
