"""
Tic-Tac-Toe Reinforcement Learning demo app.

Main window is three pannels, two large images side-by-side, and a flat control panel at the bottom.
Two image panels at the top are:
  - value function V_t(S) at time t, and
  - current delta-V_t(s), pending/recent updates to make V_{t+1}(s).
The image of a function of s, a game state is reprsented as a game-state tree with single colored
boxes instead of game state images, the color of a "state" indicating the value.
"""
import matplotlib.pyplot as plt
import numpy as np
from drawing import GameStateArtist
from tic_tac_toe import Game, get_game_tree_cached
from node_placement import FixedCellBoxOrganizer
from game_base import Result, Mark
from reinforcement_base import Environment
from abc import ABC, abstractmethod
from policies import ValueFuncPolicy
import logging
from baseline_players import HeuristicPlayer


# r is known only for these states in adavnce:
TERMINAL_REWARDS = {
    Result.X_WIN: 1.0,
    Result.O_WIN: -1.0,
    Result.DRAW: -1.0,
}


class PolicyImprovementDemo(ABC):
    """
    Abstract class for Policy Improvment, i.e. iteration between:
      - value function estimation and
      - policy optimization.  

    """

    PAUSE_POINTS = ['state',  # Pause while every v(s) is updated for a state s.
                    'pi_round',  # pause after v(s) converges, before each policy optimization step.
                    None  # run continuously.
                    ]

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, in_place=False):
        """
        :param seed_policy:  The first value function will be for this policy.
        :param opponent_policy:  The policy of the opponent.  Will be used to define the "environment" for the agent.
        :param player:  The player for the agent.  The opponent is the other player.
        :param in_place:  If True, the agent will update the value function in place.  Otherwise, it will create a new value function, update every epoch.
        """
        self._in_place = in_place
        if in_place:
            raise NotImplementedError("In-place updates not implemented yet.")
        self._player = player
        self._max_iter = 1000

        # P.I. initialization:
        self._env = Environment(opponent_policy, player)

        self._pi = seed_policy
        self._updatable_states = self._env.get_nonterminal_states()
        self._terminal_states = self._env.get_terminal_states()

        # initial value function
        self._v_terminal = [TERMINAL_REWARDS[state.check_endstate()] for state in self._terminal_states]
        self._v = {state: 0.0 for state in self._updatable_states}
        self._v.update(self._v_terminal)
        self._delta_v = {state: 0.0 for state in self._updatable_states}
        self._v_prev = self._v.copy()
        self._iter = 0

        # App  & graphics:
        self._pause_points = PolicyImprovementDemo.PAUSE_POINTS
        self._pause_point = 'state'  # initially pause after every state update.
        space_size = 5  # for all images?
        self._game_artist = GameStateArtist(self._env.get_game(), space_size=space_size)
        self._fig = plt.figure(figsize=(12, 8))
        self._fig.canvas.set_window_title("Policy Evaluation")
        self._state_images = self._make_state_images()  # dict: state- > image

        self._init_visualizations()


    @abstractmethod
    def optimize_value_function(self):
        """
        Update the value function for the current policy.
        :return:  None
        """
        pass

    def optimize_policy(self, v_tol=1e-6):
        """
        1. New policy is based on current value function.

        2. In all cases:
            Check to see if policy updates are stable:
                Now: if the value function has converged (i.e. no changes).
                TODO: if the same action(s) have highest probability in all states.
        returns: True if converged
        """
        # 1. Update the policy based on the current value function.
        if  self._in_place:
            raise NotImplementedError("In-place updates not implemented yet.")
        # if not self._in_place:
        pi_new = ValueFuncPolicy(self._v)
        
        # Check for policy stability (recommended action list has same distribution)

        def action_dist_eq(actions_1, actions_2, tolerance=1e-6):
            """
            Check if two action distributions are equal.
            :param actions_1: list of (action, probability) tuples.
            :param actions_2: list of (action, probability) tuples.
            :return: True if equal, False otherwise.
            """
            if len(actions_1) != len(actions_2):
                return False
            for a1, a2 in zip(actions_1, actions_2):
                if (a1[0]!=a2[0]):
                    raise Exception("distribution lists should be in same order...")
                if np.abs(a1[1] - a2[1]) > tolerance:
                    return False
            return True

        n_diff = 0
        for state in self._updatable_states:
            actions_old = self._pi.recommend_action(state)
            actions_new = pi_new.recommend_action(state)
            if not action_dist_eq(actions_old, actions_new):
                n_diff += 1
                self._pi.update_policy(state, actions_new)
        logging.info(f"Policy Estimation, iteration {self._iter} updated {n_diff} states .")
        return n_diff == 0  # True if no updates were made.

    def iterate(self):
        """
        Run "policy iteration," iterative policy evaluation. 
        """
        for self._iter in range(self._max_iter):
            # 1. Update the value function for the current policy.
            self.optimize_value_function()

            # 2. Check for convergence:
            if self.optimize_policy():
                break

            # 3. Pause if needed:
            if self._pause_point is not None:
                plt.pause(0.01)

        return self._v_new
    @abstractmethod
    def _init_visualizations(self):
        pass

class PolicyEvaluationPIDemo(PolicyImprovementDemo):
    """
    Policy Improvement using Policy Evaluation as the value function estimator at each step.
    """
    _EXTRA_PAUSE_POINTS = ['epoch']  # Pause after every epoch of Policy Evaluation updates (all states updated once).

    def __init__(self, seed_policy, opponent_policy, player=Mark.X, in_place=False):
        super().__init__(seed_policy, opponent_policy, player, in_place)
        self._pause_points.insert(1, 'epoch')  # add extra pause point for epoch.

    def optimize_value_function(self):
        """
        "Iterative policy evalutation, for estimating V ~ v_pi(s) for all states s and policy pi(s)."
        Barto & Sutton, 2020, p. 75.
        """
        while True:
            delta = 0
            self._v_new = self._v_terminal.copy()
            for state in self._updatable_states:
                # Calculate the value of the state using the current policy.
                # get action distribution for current policy:
                actions = self._pi.recommend_action(state)
                exp_val_new = 0.0
                for action, prob_action in actions:
                    new_states = self._env.get_next_state_dist(state, action)
                    discount_reward = 0
                    for new_state, new_state_prob in new_states():
                        discount_reward += new_state_prob * self._v[new_state] * self._gamma
                    exp_val_new += prob_action * discount_reward
                self._v_new[state] = exp_val_new

    def _init_visualizations(self):
        # create layout of value function visualization using hot-cold heatmap and
        # positions from the FixedCellBoxOrganizer.
        LAYOUT = {'win_size': (1920, 1080)}
        SPACE_SIZES = [9, 6, 5, 3, 3, 2, 3, 2, 3, 4]  # only used if displaying the full tree, else attempted autosize

        # START HERE
        space_size= 2
        artists = GameStateArtist(space_size = space_size)
        heat_colors = plt.colormaps['hot']
        # create a color map for the values:
        all_values = [self._v.values()]
        value_range = max(all_values), min(all_values)
        norm = plt.Normalize(vmin=value_range[1], vmax=value_range[0])
        cmap = plt.get_cmap(heat_colors, 256)
        # create a color map for the values:
        layers = []
        colors = {}
        for state in self._v:
            color = cmap(norm(self._v[state]))
            colors[state] = color
            layers.append(state)
        # create a color map for the values:

        



def run_app():
    # Example usage:
    seed_policy = HeuristicPlayer(n_rules=6, mark=Mark.X)  # Replace with your seed policy
    opponent_policy = HeuristicPlayer(n_rules=6, mark=Mark.X)  # Replace with your opponent policy
    player = Mark.X  # Replace with your player mark

    demo = PolicyEvaluationPIDemo(seed_policy, opponent_policy, player)
    demo.iterate()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    run_app()
