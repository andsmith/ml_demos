"""
Classes for Reinforcement Learning policy and value functions for Tic Tac Toe.

WLOG, let the policy's player be X.


"""
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
import re

class Result(IntEnum):
    X_WIN = 1
    O_WIN = 2
    DRAW = 3

class Mark(IntEnum):
    EMPTY = 0
    X = 1  # also denotes the player
    O = 2



class Game(object):
    """
    Represent the state of a game of Tic Tac Toe.
    """

    def __init__(self, state=None):
        if state is None:
            self.state = np.zeros((3, 3), dtype=int) + Mark.EMPTY
        else:
            self.state = state

    def __str__(self):
        chars = {Mark.EMPTY: " ", Mark.X: "X", Mark.O: "O"}
        s = "\n-----\n".join("|".join(chars[i] for i in row) for row in self.state)
        return s
    
    def indent(self, tabs):
        return re.sub(r'^', '\t'*tabs, str(self), flags=re.MULTILINE)

    def __hash__(self):
        return hash(tuple(self.state.flatten()))

    def copy(self):
        return Game(self.state.copy())

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    @staticmethod
    def get_all_actions():
        return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    def get_actions(self):
        free_rows, free_cols = np.where(self.state == Mark.EMPTY)
        return list(zip(free_rows, free_cols))

    def move(self, mark, action):
        """
        :param mark: Mark.X or Mark.O
        :param action: (row, col) tuple
        """
        if self.state[action] != Mark.EMPTY:
            raise ValueError("Invalid move")
        self.state[action] = mark

    def check_terminal(self):
        """
        Is this a terminal game state?
        :returns: Mark.X if X wins, Mark.O if O wins, Mark.EMPTY if draw, None otherwise.
        """
        # Check rows
        for row in self.state:
            if np.all(row == Mark.X):
                return Result.X_WIN
            if np.all(row == Mark.O):
                return Result.O_WIN
        # Check columns
        for col in self.state.T:
            if np.all(col == Mark.X):
                return Result.X_WIN
            if np.all(col == Mark.O):
                return Result.O_WIN
        # Check diagonals
        if np.all(np.diag(self.state) == Mark.X) or np.all(np.diag(np.fliplr(self.state)) == Mark.X):
            return Result.X_WIN
        if np.all(np.diag(self.state) == Mark.O) or np.all(np.diag(np.fliplr(self.state)) == Mark.O):
            return Result.O_WIN
        # Check for draw
        if Mark.EMPTY not in self.state:
            return Result.DRAW
        return None

    @staticmethod
    def get_game_tree(player):
        # Enumerate every state player=X or O could be confronted with, i.e. all states reachable from
        #   the empty to a terminal state. Note this is not all game boards, or even all reachable game states, 
        #   as the player will not be asked to make moves that skip the opponent's turn, etc.
        #
        # Consider both the player going first and second.
        #
        # Return three dicts:
        #  terminality: key = Game object (state), value = one of Mark.X, Mark.O, Mark.EMPTY (draw), or None (not terminal).
        #  child_states: key = Game object (state), value = list of (Game, action) tuples for all valid moves from the Key state (after 1 round, 1 player move then 1 opponent move).
        #  parent_states: key = Game object (state), value = list of (Game, action) tuples for all valid moves to the Key state (after 1 round).

        def _opponent(player):
            return Mark.X if player == Mark.O else Mark.O
        
        player_states = {}
        opponent_states = {}

        next_states = {}  # key = state, value = list of states that can be reached by player
        prev_states = {}  # key = state, value = list of states that can reach this state

        def _enumerate(state, current_player=Mark.X):
            """
            :param state: Game state
            :param player: who makes the next move, Mark.X or Mark.O?
            :returns: list of (child_state, action) tuples for all valid moves from state, or [] if terminal.
            """
            if state in player_states or state in opponent_states:
                return []
            print("Enumerating for player %s:" % current_player)
            print(state.indent(1))
            term = state.check_terminal()
            print("\tTerminality:", term)
            if current_player == player:
                player_states[state] = term
            else:
                opponent_states[state] = term

            if term is not None:
                return []
            
            actions = state.get_actions()
            children = []  # next game state (after player move)
            grand_children = []  # next valid RL state (after player & opponent move)
            for action in actions:
                new_state = state.copy()
                new_state.move(current_player, action)

                children.append((new_state, action))
                grand_children.extend(_enumerate(new_state, _opponent(current_player)))

            if current_player == player:
                print("\trecording %i successor states." % len(grand_children))
                next_states[state] = grand_children
                for gchild, action in grand_children:
                    g_childs_parents = prev_states.get(gchild, [])
                    g_childs_parents.append((state, action))
                    prev_states[gchild] = g_childs_parents
            print("\treturning with %i successor states." % len(children))
            return children
        
        _enumerate(Game(), Mark.X)
        _enumerate(Game(), Mark.O)

        return player_states, next_states, prev_states
    
def test_game():
    import ipdb; ipdb.set_trace()
    states, successors, predecssors = Game.get_game_tree(player=Mark.X)
    print(len(states), "states for player X")


if __name__ == "__main__":
    test_game()