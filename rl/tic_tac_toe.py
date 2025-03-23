"""
Classes for Reinforcement Learning policy and value functions for Tic Tac Toe.

WLOG, let the policy's player be X.


"""
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum

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
    def enumerate_states():
        # Enumerate every state reachable from the empty to a terminal state.
        # Return a dict of state: terminality, where state is a Game object and terminality is
        #   one of Mark.X, Mark.O, Mark.EMPTY (draw), or None (not terminal)..

        def _opponent(player):
            return Mark.X if player == Mark.O else Mark.O
        
        states = {}

        def _enumerate(state, player=Mark.X):
            """
            :param state: Game state
            :param player: who makes the next move, Mark.X or Mark.O?
            """
            if state in states:
                return
            term = state.check_terminal()
            states[state] = term
            if term is not None:
                return
            actions = state.get_actions()
            for action in actions:
                new_state = state.copy()
                new_state.move(player, action)
                _enumerate(new_state, _opponent(player))
        #import ipdb; ipdb.set_trace()
        _enumerate(Game(), Mark.X)
        _enumerate(Game(), Mark.O)

        return states
    
def test_game():
    states = Game.enumerate_states()
    print(len(states), "states")


if __name__ == "__main__":
    test_game()