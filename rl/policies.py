import numpy as np
from abc import ABC, abstractmethod

from tic_tac_toe import Mark, Result, Game

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
        self.losing_result = Result.O_WIN if player == Mark.X else Result.X_WIN
        self.draw_result = Result.DRAW

    @abstractmethod
    def recommend_action(self, state):
        """
        Return an action to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples where the sum of all probabilities is 1.0.
        """
        pass

    def __str__(self):
        return self.__class__.__name__ + "(%s)" % self.player.name
    
    def get_policy(self):
        """
        :returns:dict mapping all possible (non-terminal) game states to the recommended action (or action distribution)
            dict[state] = [(action, probability), ...]
        """
        states = Game.enumerate_states(self.player)
        return {state: self.recommend_action(state) for state in states}




class ValueFuncPolicy(Policy):
    """
    A policy function that is parameterized by a value function:
        V(s) = value of state s (expected discounted reward for being in state s if we follow the policy)

        
    This class can be used as the value function itself by calling DPValueFunc.value(state) and also as a
      Policy object acting so the next state has highest value according to v(), by calling the inherited
      method ValueFuncPolicy.recommend_action(state).


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
        :param deterministic:  return one or many actions
           If True, the policy is deterministic, will alawys return the most valuable action with p=1.
           If False, it will return all n actions having the same highest value with uniform probability p=1/n.
        """
        super().__init__(deterministic)
        self._v = self._make_value_func()  # hash of game state to value

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
            actions = [a for a, v in values.items() if v == max_value]
            p = 1.0 / len(actions)
            return [(a, p) for a in actions]
        
    def value(self, state):
        return self._v(state)

    @abstractmethod
    def _make_value_func(self):
        """
        Create the value function to use for this policy
        :return:  dict(key=game_state, value=value)
            game_state: Game object
            value: float
        """
        pass