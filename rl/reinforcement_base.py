"""
Base classes and functions for RL demos.
"""
from tic_tac_toe import Game, get_game_tree_cached
from game_base import Mark, Result

import numpy as np
import logging



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
    def __init__(self, opponent_policy, player_mark = Mark.X):
        self._opp = opponent_policy
        self._tree = get_game_tree_cached(player=player_mark)
        self._p_next_state = self._calc_state_transitions()

    def _calc_state_transitions(self):
        """
        For every state our OPPONENT can be confronted with (i.e. even number of moves made or one more player move),
        calculate the probability of each possible next state by evaluating the opponent's policy.
        """
        