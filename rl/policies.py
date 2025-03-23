import numpy as np
from abc import ABC, abstractmethod

from tic_tac_toe import Mark, Result

class Policy(ABC):
    """
    Abstract class for a policy function, anything that can play the game,
       and RL agent, a MiniMax player, a human player UI, etc.

    A policy function takes a game state and returns a probability 
    distribution over the possible actions.
    """

    def __init__(self, player, deterministic=False):
        """
        :param player:  one of Mark.X or Mark.O  Policy optimizes actions for this player.
        :param deterministic: If True, the policy is deterministic, 
            i.e. it returns a single action with probability 1.0.
            If False, will return all actions having non-zero probability.
        """
        self._deterministic = deterministic
        if player != Mark.X and player != Mark.O:
            raise ValueError("mark must be Mark.X or Mark.O")
        self.player = player
        self.opponent = Mark.X if player == Mark.O else Mark.O

        self.winning_result = Result.X_WIN if player == Mark.X else Result.O_WIN

    @abstractmethod
    def recommend_action(self, state):
        """
        Return an action to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples.
        """
        pass

    def __str__(self):
        return self.__class__.__name__ + "(%s)" % self.player.name


class ValuePolicy(Policy):
    """
    A policy function that is parameterized by a value function
        V(s) = value of state s.

    The action recommended by the policy function is the action that
    maximizes the value function of the resulting state, i.e.
        Policy(s) = argmax_a V(T(s,a)), where
        T(s,a) = state resulting from taking action a in state s.
    If more than one next state has the maximal values, all are returned with 
    uniform probability, or one is selected arbitrarily if deterministic.
    """

    def __init__(self, deterministic=False):
        """
        :param value_function: A function that takes a state and returns a value.
        :param deterministic: If True, the policy is deterministic, will alawys return a single action.
        """
        super().__init__(deterministic)
        self._v = self._get_value_func()  # hash of game state to value

    def recommend_action(self, state):
        """
        Return an action to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples.
        """
        actions = state.get_actions()
        if self._deterministic:
            action = max(actions, key=lambda a: self._v(self._transition(state, a)))
            return [(action, 1.0)]
        else:
            values = {a: self._v(self._transition(state, a)) for a in actions}
            max_value = max(values.values())
            actions = [(a, 1.0) for a, v in values.items() if v == max_value]
            p = 1.0 / len(actions)
            return [(a, p) for a in actions]
