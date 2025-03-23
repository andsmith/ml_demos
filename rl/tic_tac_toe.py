"""
Classes for Reinforcement Learning policy and value functions for Tic Tac Toe.

WLOG, let the policy's player be X.


"""
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
import re
import cv2
from util import get_annulus_polyline
import matplotlib.pyplot as plt
from drawing import MarkerArtist
from game_base import Mark, Result


class Game(object):
    """
    Represent the state of a game of Tic Tac Toe.
    """

    def __init__(self, state=None):
        if state is None:
            self.state = np.zeros((3, 3), dtype=int) + Mark.EMPTY
        else:
            self.state = state

        # for rendering:
        self._color_o = (255, 127, 14)  # matplotlib orange for O
        self._color_x = (31, 119, 180)  # matplotlib blue for X
        self._color_draw = (57, 255, 20)  # green for draw
        self._SHIFT = 5
        self._mark_artist = MarkerArtist(color_x=self._color_x, color_o=self._color_o, color_d=self._color_draw)
        self._SHIFT_BITS = 4
        self._SHIFT = 1 << self._SHIFT_BITS

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

        def _enumerate(state, current_player):
            """
            :param state: Game state
            :param player: who makes the next move, Mark.X or Mark.O?
            :returns: list of (child_state, action) tuples for all valid moves from state, or [] if terminal.
            """
            if state in player_states or state in opponent_states:
                return []
            # print("Enumerating for player %s:" % current_player)
            # print(state.indent(1))
            term = state.check_terminal()
            # print("\tTerminality:", term)
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
                # print("\trecording %i successor states." % len(grand_children))
                next_states[state] = grand_children
                for gchild, action in grand_children:
                    g_childs_parents = prev_states.get(gchild, [])
                    g_childs_parents.append((state, action))
                    prev_states[gchild] = g_childs_parents
            # print("\treturning with %i successor states." % len(children))
            return children

        _enumerate(Game(), Mark.X)  # player=X goes first
        _enumerate(Game(), Mark.O)  # player=O goes first

        return player_states, next_states, prev_states

    def get_img(self, space_size=11,
                bar_w_frac=.1,
                marker_padding_frac=.2,
                color_bg=(255, 255, 255),
                color_lines=(0, 0, 0),):
        """
        Return an image of the game board.
        :param space_size: size of each cell in pixels
        :param bar_w_frac: width of the lines as a fraction of space_size
        :param marker_padding_frac: padding around the marker as a fraction of space_size
        :param color_x: color of X markers
        :param color_o: color of O markers
        :param color_bg: color of the background
        :param color_lines: color of the lines
        """
        # Param to MarkerArtist.add_marker, what to draw for each Mark / Result
        artist_arg = {Mark.X: 'X', Mark.O: 'O', Result.X_WIN: 'X', Result.O_WIN: "O", Result.DRAW: "D"}
        marker_colors = {Mark.X: self._color_x, Mark.O: self._color_o, Mark.EMPTY: self._color_draw}
        space_size = space_size + 1 if space_size % 2 == 1 else space_size  # displays better with odd cell sizes
        line_width = max(1, int(space_size * bar_w_frac))

        side_len = space_size * 3 + line_width * 2
        img = np.zeros((side_len, side_len, 3), dtype=np.uint8)
        img[:, :] = color_bg

        # Draw grid
        for i in range(1, 3):
            start = i * space_size + (i - 1) * line_width
            img[start:start + line_width, :] = color_lines
            img[:, start:start + line_width] = color_lines
        print("Getting image for game state:")
        print(self.indent(1))

        # Draw marks
        for i in range(3):
            for j in range(3):
                if self.state[i, j] == Mark.EMPTY:
                    continue
                x_center = j * space_size + j * line_width + space_size / 2
                y_center = i * space_size + i * line_width + space_size / 2
                center = (x_center, y_center)
                m_char = artist_arg[self.state[i, j]]
                self._mark_artist.add_marker(img, center, space_size, m_char, pad_frac=marker_padding_frac)

        # Draw win/loss/draw
        term = self.check_terminal()
        if term is not None:
            # Draw marker over entire board
            img_side_len = img.shape[0]
            marker = artist_arg[term]
            img_center = (img_side_len//2, img_side_len//2)
            self._mark_artist.add_marker(img, img_center, img_side_len, marker,
                                         pad_frac=marker_padding_frac * 1.5)  # pad more for full board marker

        return img

    @staticmethod
    def from_strs(rows):
        """
        e.g: ["XOX",
              "OXO",
              "X  "]
        :param rows: list of 3 strings, each with 3 characters from {"X", "O", " "}
        """
        state = np.zeros((3, 3), dtype=int)
        for i, row in enumerate(rows):
            for j, c in enumerate(row):
                if c == "X":
                    state[i, j] = Mark.X
                elif c == "O":
                    state[i, j] = Mark.O
        return Game(state)


def test_game():

    states, successors, predecssors = Game.get_game_tree(player=Mark.X)
    print(len(states), "states for player X")


if __name__ == "__main__":
    test_game()
    print("Done.")
