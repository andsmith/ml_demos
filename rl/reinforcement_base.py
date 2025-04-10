"""
Base classes and functions for RL demos.
"""
from tic_tac_toe import Game, get_game_tree_cached, GameTree
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

    def __init__(self, opponent_policy, player_mark=Mark.X):
        """
        :param opponent_policy:  policy of the opponent, Policy object initialized with the opposite player mark.  Will be 
            used to define the "environment" for the agent.
        :param player_mark: The player for the agent.  The opponent is the other player.
        """
        self._player = player_mark
        self._opponent = GameTree.opponent(player_mark)
        self._opp = opponent_policy
        self._tree = get_game_tree_cached(player=player_mark)
        self._terminal, self._children, self._parents, self._initial = self._tree.get_game_tree()
        self._p_next_state = self._calc_state_transitions()

    def get_children(self):
        return self._children

    def _calc_state_transitions(self):
        """
        The state transitions from the agent's perspective are the probabilities of the opponents moves
        for every move the agent can make.

        For every state our OPPONENT can be confronted with (i.e. even number of moves made or one more player move),
        calculate the probability of each possible next state by evaluating the opponent's policy.
        """
        p_sp_g_sa = {}  # for every state, for every action the agent can take,
        # what is the distribution of next states for the agent (after opponent makes a move).
        logging.info("Calculating opponent's state transitions...")

        nonterm = self.get_nonterminal_states()
        for s_i, state in enumerate(nonterm):
            if s_i % 100 == 0:
                logging.info("\tprocessing state %d of %d nonterminals." % (s_i, len(nonterm)))
            agent_actions = state.get_actions()
            for agent_action in agent_actions:
                # get the next state after the agent makes a move
                inter_state = state.clone_and_move(agent_action, self._player)  # after agent, before opponent
                next_state_dist = self._opp.recommend_action(inter_state)

                # store the distribution of next states for this state and action
                p_sp_g_sa[(state, agent_action)] = {agent_action: next_state_dist}

        return p_sp_g_sa

    def _state_valid(self, state):
        """
        Check if the given state is valid for self._player.
        """
        return np.sum(state.state == self._player) <= np.sum(state.state == self._opponent)

    def p_next_states(self, state, action):
        return self._p_next_state[(state, action)]

    def get_terminal_states(self):
        return [state for state in self._terminal if self._terminal[state] is not None]

    def get_nonterminal_states(self):
        return [state for state in self._terminal if (self._terminal[state] is None and self._state_valid(state))]
