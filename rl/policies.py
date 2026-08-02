import numpy as np
from abc import ABC, abstractmethod

from tic_tac_toe import Game
from game_base import Mark, Result, get_reward, OTHER_GUY
import re

class Policy(ABC):
    """
    Abstract class for a policy function, anything that can play the game,
       and RL agent, a MiniMax player, a human player UI, etc.

    A policy function takes a game state and returns a probability 
    distribution over ALL possible actions.
    """

    def __init__(self, player):
        """
        :param player:  one of Mark.X or Mark.O  Policy optimizes actions for this player.
        """
        if player != Mark.X and player != Mark.O:
            raise ValueError("mark must be Mark.X or Mark.O")
        self.player = player
        self.opponent = Mark.X if player == Mark.O else Mark.O

        self.winning_result = Result.X_WIN if player == Mark.X else Result.O_WIN
        self.losing_result = Result.O_WIN if player == Mark.X else Result.X_WIN
        self.draw_result = Result.DRAW

    @abstractmethod
    def __str__(self):
        """
        Return a string representation of the policy.
        """
        pass

    @abstractmethod
    def recommend_action(self, state):
        """
        Return actions to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples where the sum of all probabilities is 1.0.
          (NOTE:  the list should be complete, i.e. all possible actions should be included)
        """
        pass

    def compare(self, other, states, count=False, deterministic=True, prob_rel_tol=1e-5):
        """
        For all updatable states, is the distribution of recommended actions the same?
        :param other:  another Policy object to compare with.
        :param states:  list of Game states to compare the policies on.
        :param count:  if True, return the number of different actions, otherwise return True/False.
        :param deterministic:  if True, only compare the best action from each policy's distribution.
        :param prob_rel_tol:  relative tolerance for the probabilities of the best actions.
        """
        n_diff=0
        for state in states:
            
            a_dist_1 = self.recommend_action(state)
            a_dist_2 = other.recommend_action(state)
            if not deterministic and len(a_dist_1) != len(a_dist_2):
                n_diff +=1
                if not count:
                    return False
            best_ind_1 = np.argmax([a[1] for a in a_dist_1])
            best_ind_2 = np.argmax([a[1] for a in a_dist_2])
            if a_dist_1[best_ind_1][0] != a_dist_2[best_ind_2][0]:
                n_diff +=1
                if not count:
                    return False
            prob_1 = a_dist_1[best_ind_1][1]
            prob_2 = a_dist_2[best_ind_2][1]
            different_max_probs = (prob_1==0.0 and prob_2 != 0.0) or (prob_1 != 0.0 and prob_2 == 0.0)\
                or (abs(prob_1 - prob_2) > prob_rel_tol * max(prob_1, prob_2))
            if not deterministic and different_max_probs:
                n_diff += 1
                if not count:
                    return False
                
        if not count:
            return True
        return n_diff

    def take_action(self, game_state, deterministic=False):
        # just return the sampled action
        distribution, best = self.recommend_and_take_action(game_state, deterministic)
        return distribution[best][0]
    
    def recommend_and_take_action(self, game_state, deterministic=False):
        """
        Get action distribution, sample it.
        """
        recommendations = self.recommend_action(game_state)
        probs = np.array([prob for _, prob in recommendations])
        actions = np.array([action for action, _ in recommendations])
        action_inds = np.arange(len(actions))
        if deterministic:
            # only sample from actions w/prob. equal to best action's.
            best_prob = np.max(probs)
            probs[probs < best_prob] = 0.0
            probs = probs / np.sum(probs)
        action_ind = np.random.choice(action_inds, p=probs)
        return recommendations, action_ind


    def __str__(self):
        return self.__class__.__name__ + "(%s)" % self.player.name


class TabularPolicy(Policy):
    """
    Policy defined by an explicit state -> [(action, prob), ...] table,
    e.g. the result of a policy-improvement sweep.
    """

    def __init__(self, pi, player):
        """
        :param pi: dict mapping Game states to [(action, prob), ...] distributions.
        :param player: Mark.X or Mark.O
        """
        super().__init__(player=player)
        self._pi = pi

    def __str__(self):
        return "TabularPolicy(%s)" % self.player.name

    def recommend_action(self, state):
        return self._pi[state]


class InvPolicy(Policy):
    """
    Inverse policy, i.e. the opponent's policy.
    """
    def __init__(self, opp_policy):
        self._opp_pi = opp_policy
        self.player = OTHER_GUY[opp_policy.player]
        self.opponent = opp_policy.player
        self.winning_result = opp_policy.losing_result
        self.losing_result = opp_policy.winning_result
        self.draw_result = opp_policy.draw_result

    def __str__(self):
        orig_str = str(self._opp_pi)
        if 'X' in orig_str[-3:].upper():
            return orig_str[:-3]+"(O)"
        elif 'O' in orig_str[-3:].upper():
            return   orig_str[:-3]+"(X)"
        else:
            raise ValueError("Unknown player in opponent policy string: %s" % orig_str)
        
    def recommend_action(self, state):
        state = state.invert()
        act_dist = self._opp_pi.recommend_action(state)
        return act_dist