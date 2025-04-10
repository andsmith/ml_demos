import numpy as np
from abc import ABC, abstractmethod

from tic_tac_toe import Mark, Result, Game


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
    def recommend_action(self, state):
        """
        Return actions to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples where the sum of all probabilities is 1.0.
          (NOTE:  the list should be complete, i.e. all possible actions should be included)
        """
        pass


    def take_action(self, game_state, epsilon=0):
        recommendations = self.recommend_action(game_state)
        probs = np.array([prob for _, prob in recommendations])
        actions = np.array([action for action, _ in recommendations])
        action_inds = np.arange(len(actions))
        # sample from the distribution
        action_ind = np.random.choice(action_inds, p=probs)
        return tuple(actions[action_ind])

    def __str__(self):
        return self.__class__.__name__ + "(%s)" % self.player.name


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

    def __init__(self, v, player=Mark.X):
        """
        :param player: Mark.X or Mark.O  The player for whom the policy is optimized.
        :return: None
        """
        super().__init__(player=player)  # default player is X
        self._v = v  # hash of Game (state) to value

    def recommend_action(self, state):
        """
        Return an action to take given the current state.

        :param state: A game state.
        :return: list of (action, probability) tuples.
        """
        actions = state.get_actions()
        action_values = []
        for action in actions:
            next_state = state.clone_and_move(action, self.player)
            if next_state not in self._v:
                print("Warning:  state %s not in value function for player %s." % (next_state, self.player.name))
            value = self._v[next_state]
            action_values.append((action, value))

        best_action = max(action_values, key=lambda x: x[1])[0]
        return best_action

    def value(self, state):
        return self._v[state]
