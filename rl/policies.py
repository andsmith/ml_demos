import numpy as np
from abc import ABC, abstractmethod

from tic_tac_toe import Game
from game_base import Mark, Result, TERMINAL_REWARDS, get_reward

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

