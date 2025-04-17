"""
Base classes and functions for RL demos.
"""
from tic_tac_toe import Game, get_game_tree_cached, GameTree
from game_base import Mark, Result
from enum import IntEnum
import numpy as np
import logging


class PIPhases(IntEnum):
    POLICY_EVAL = 0
    POLICY_OPT = 1

class Environment(object):
    """
    The Environment class encapsulates the rules of Tic Tac Toe as well as the 
    policies of the opponent our RL agent is learning to play against.

    Each timestep, the agent takes an action (makes a move) based on the current state.
    The environment responds with a new state (the game board after the move AND the opponent's move distribution),
    and a reward.

    Two-player zero sum games are more properly modeled as an Alternating Markov Game. 
    This class shoehorns the game into a Markov Decision Process (MDP) by assuming the opponent is a stochastic policy.
    """

    def __init__(self, opponent_policy, player_mark=Mark.X):
        """
        :param opponent_policy:  policy of the opponent, Policy object initialized with the opposite player mark.  Will be 
            used to define the "environment" for the agent.
        :param player_mark: The player for the agent.  The opponent is the other player.
        """
        self.player = player_mark
        self.opponent = GameTree.opponent(player_mark)
        self.pi_opp = opponent_policy
        self.tree = get_game_tree_cached(player=player_mark, verbose=True)
        self.terminal, self.children, self.parents, self.initial = self.tree.get_game_tree()

    def _state_valid(self, state):
        """
        Check if it's really agent's turn.
        """
        return np.sum(state.state == self.player) <= np.sum(state.state == self.opponent)

    def get_terminal_states(self):
        return [state for state in self.terminal if self.terminal[state] is not None]

    def get_nonterminal_states(self):
        # states player can see in a game (when it's their turn)
        return [state for state in self.terminal if (self.terminal[state] is None and self._state_valid(state))]
