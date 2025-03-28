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
from drawing import MarkerArtist, draw_win_line, get_size
from game_base import Mark, Result, WIN_MARKS, get_cell_center
import logging
import pickle
import os
from copy import deepcopy
from colors import COLOR_BG, COLOR_LINES, COLOR_DRAW, COLOR_X, COLOR_O

# Defaults to get_img

ARTIST = MarkerArtist()  # change default marker colors w/args here


class Game(object):
    """
    Tic-tac-toe game board.
    """

    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8) + Mark.EMPTY
        # for rendering:
        self._SHIFT = 5
        self._mark_artist = ARTIST
        self._SHIFT_BITS = 4
        self._SHIFT = 1 << self._SHIFT_BITS

    def __hash__(self):
        return hash(tuple(self.state.flatten().tolist()))

    def __eq__(self, other):
        return np.all(self.state == other.state)

    def get_actions(self):
        # return list of all (i,j) tuples where state[i,j] == Mark.EMPTY
        return [p for p in zip(*np.where(self.state == Mark.EMPTY))]
    
    def n_free(self):
        return np.sum(self.state == Mark.EMPTY)
    
    def n_marked(self):
        return np.sum(self.state != Mark.EMPTY)

    def clone_and_move(self, action, mark):
        new_board = Game()
        new_board.state = np.copy(self.state)
        new_board.state[action] = mark
        return new_board

    def _check_winner(self, mark):
        # check rows
        for i in range(3):
            if np.all(self.state[i] == mark):
                return True
        # check columns
        for j in range(3):
            if np.all(self.state[:, j] == mark):
                return True
        # check diagonals
        if np.all(np.diag(self.state) == mark) or np.all(np.diag(np.fliplr(self.state)) == mark):
            return True
        return False

    def check_endstate(self):
        if self._check_winner(Mark.X):
            return Result.X_WIN
        if self._check_winner(Mark.O):
            return Result.O_WIN
        if np.all(self.state != Mark.EMPTY):
            return Result.DRAW
        return None

    _CHARS = {Mark.EMPTY: ' ', Mark.X: 'X', Mark.O: 'O'}

    def __str__(self):
        return '\n-----\n'.join(['|'.join([Game._CHARS[mark] for mark in row]) for row in self.state])

    def indent(self, n_tabs=1):
        s = str(self)
        return '\n'.join(['\t'*n_tabs + line for line in s.split('\n')])

    @staticmethod
    def get_space_size(img_size, bar_w_frac=.15):
        """
        Attempt to predict a good cell size for a given image size
        (i.e. inverse of get_image_dims).
        """
        space_size = img_size
        dims = Game.get_image_dims(space_size, bar_w_frac=bar_w_frac)
        while dims['img_size'] > img_size:
            space_size -= 1
            dims = Game.get_image_dims(space_size, bar_w_frac=bar_w_frac)
        return space_size

    @staticmethod
    def get_image_dims(space_size, bar_w_frac=.15, marker_padding_frac=.4):
        """
        Get the dimensions dict for drawing a game state.

                +--u--u--+
                |  |  |  |
                +--+--+--+
                |  |  |  |
                +--+--+--+
                |  |  |  |
                +--l--l--+

        * The grid image is composed of 3x3 "cells" or [marker] "spaces."
        * The line width is expressed as a fraction of the cell side length.
        * The upper and lower attachment points are the 'u' and 'l' points, respectively.
        * The bounding box may or may not be drawn, but is included in the grid's side length.

        Therefore, the smallest possible grid size is 7x7 pixels, space_size=1.
        (The minimum bar width is 1 pixel, regardless of bar_w_frac.)

        :param space_size: int, width in pixels of a squares in the 3x3 game grid.
        :param bar_w_frac: float, width of the lines of the grid as a fraction of space_size.

        :returns: dict with
          'img_size': grid image's side length
          'line_t': line width for the 4 grid lines and the bounding box
          'upper': [(x1, y1), (x2, y2)] attachment points for the upper grid line (floats, non-integers if line_width is even)
          'lower': [(x1, y1), (x2, y2)] attachment points for the lower grid line
            'space_size': space_size
            'bar_w': grid line width
        """
        grid_line_width = max(1, int(space_size * bar_w_frac))
        img_side_len = space_size * 3 + grid_line_width * 4
        upper = [(space_size + 3 * grid_line_width / 2, grid_line_width),
                 (space_size * 2 + 5 * grid_line_width / 2, grid_line_width)]
        lower = [(upper[0][0], img_side_len-grid_line_width),
                 (upper[1][0], img_side_len-grid_line_width)]

        cell_x = (grid_line_width, grid_line_width + space_size)  # x range of first cell
        cell_y = (grid_line_width, grid_line_width + space_size)  # y range of first cell
        cell_x_offset = space_size + grid_line_width  # add one or two to cell_X to get the other cells
        cell_y_offset = space_size + grid_line_width  # add one or two to cell_Y to get the other cells

        # cell_span[row][col] = {'x': (x1, x2), 'y': (y1, y2)}
        cell_spans = [[{'x': (cell_x[0] + col * cell_x_offset, cell_x[0] + col * cell_x_offset + space_size),
                        'y': (cell_y[0] + row * cell_y_offset, cell_y[0] + row * cell_y_offset + space_size)}
                       for col in range(3)]
                      for row in range(3)]

        return {'img_size': img_side_len,
                'line_t': grid_line_width,
                'space_size': space_size,
                'bar_w': grid_line_width,
                'upper': upper,
                'lower': lower,
                'cells': cell_spans,
                'marker_padding_frac': marker_padding_frac}

    def get_img(self,
                dims,
                color_bg=COLOR_BG,
                color_lines=COLOR_LINES,
                draw_box=None):
        """
        Return an image of the game board & its dimension dictionary.

        :param dims: dict, output of get_image_dims
        :param color_bg: color of the background
        :param color_lines: color of the lines
        :param draw_box: draw a bounding box around the grid (with (non)terminal color)
            if None, only draw the box around terminal states

        """

        space_size = dims['space_size']
        line_t = dims['line_t']
        img_s = dims['img_size']

        # Create the image
        img = np.zeros((img_s, img_s, 3), dtype=np.uint8)
        img[:, :] = color_bg

        term = self.check_endstate()
        grid_line_color = color_lines
        box_color = {Result.DRAW: COLOR_DRAW,
                     Result.X_WIN: COLOR_X,
                     Result.O_WIN: COLOR_O}[term] if term is not None else color_lines

        drawing_box = (draw_box is None and term is not None) or draw_box

        # Draw grid lines
        for i in [1, 2]:
            line_color = grid_line_color if not (term is not None and term==Result.DRAW) else COLOR_DRAW
            z_0 = i * (space_size + line_t)
            z_1 = z_0 + line_t

            w0 = line_t
            w1 = img_s - line_t

            img[z_0:z_1, w0:w1] = line_color
            img[w0:w1, z_0:z_1] = line_color

        # Draw bounding box, if terminal, extra heavy.
        if drawing_box:
            line_color = box_color
            for i in [0, 3]:
                z_0 = i * (space_size + line_t)
                z_1 = z_0 + line_t
                img[z_0:z_1, :] = line_color
                img[:, z_0:z_1] = line_color

        # Draw markers
        for i in range(3):
            for j in range(3):
                if self.state[i, j] == Mark.EMPTY:
                    continue
                self._mark_artist.add_marker(img, dims, (i, j), self.state[i, j])

        size = get_size(space_size)

        # Draw win lines, connecting a row.
        if size=='tiny':  # Skip for tiny images
            return img
        win_lines = self.get_win_lines()
        for line in win_lines:
            draw_win_line(img, dims, line, box_color)

        # Finally, if it's in the middle category, draw an extra thick bounding box for clarity
        if False:#drawing_box and term is not None:
            line_color = box_color
            box_w = 6
            for i in [0, 3]:
                img[i:i+box_w, :] = line_color
                img[:, i:i+box_w] = line_color
                img[img_s-box_w:img_s, :] = line_color
                img[:, img_s-box_w:img_s] = line_color
                


        return img

    def get_win_lines(self):
        term = self.check_endstate()
        if term in [None, Result.DRAW]:
            return []
        winner_mark = WIN_MARKS[term]
        lines = []  # (i.e. {'orient': 'h','v','d'
        #                    'c1': (i1, j1),
        #                    'c2': (i2, j2)
        #                    }  where i* and j* are in [0, 1, 2])
        #             for every 3-in-a-row.
        for i in range(3):
            # check rows
            if np.all(self.state[i, :] == winner_mark):
                lines.append({'orient': 'h', 'c1': (i, 0), 'c2': (i, 2)})
            # check columns
            if np.all(self.state[:, i] == winner_mark):
                lines.append({'orient': 'v', 'c1': (0, i), 'c2': (2, i)})
        # check diagonals
        if np.all(np.diag(self.state) == winner_mark):
            lines.append({'orient': 'd', 'c1': (0, 0), 'c2': (2, 2)})
        if np.all(np.diag(np.fliplr(self.state)) == winner_mark):
            lines.append({'orient': 'd', 'c1': (0, 2), 'c2': (2, 0)})
        return lines

    @staticmethod
    def from_strs(rows):
        """
        e.g: ["XOX",
              "OXO",
              "X  "]
        :param rows: list of 3 strings, each with 3 characters from {"X", "O", " "}
        """
        g = Game()
        state = np.zeros((3, 3), dtype=int)
        for i, row in enumerate(rows):
            for j, c in enumerate(row):
                if c == "X":
                    state[i, j] = Mark.X
                elif c == "O":
                    state[i, j] = Mark.O
        g.state = state
        return g


class GameTree(object):
    """
    Play every possible game.
    """

    def __init__(self, player, verbose=False):
        self._player = player

        # state (Game): (None or one of the Result values).  Check here to if a state has been seen before.
        self._terminal = {}
        # state: {child_state: (action, player) for each of state's child states}, w/the player & actions that led to them.
        self._children = {}
        self._parents = {}  # state: [state].  List of states that can lead to this state.
        self._initial = []  # List of initial states.
        self._verbose = verbose
        self._build_tree()

    @staticmethod
    def opponent(player):
        return Mark.X if player == Mark.O else Mark.O

    def _build_tree(self):
        initial = Game()
        self._initial = [initial]
        self._terminal[initial] = initial.check_endstate()
        self._children[initial] = {}
        self._parents[initial] = []

        def _initial_printout(state):
            if self._verbose:
                print("\n---------------------------------------------------------------\n")
                print("Collecting states for player %s, starting from initial state:\n" % self._player.name)
                print(state.indent(1))
                print("\n\tRandom (p<.0001) state updates:\n")

        # dict(first_player = {result_type: count})
        self._game_outcomes = {Mark.X: {Result.X_WIN: 0, Result.O_WIN: 0, Result.DRAW: 0},
                               Mark.O: {Result.X_WIN: 0, Result.O_WIN: 0, Result.DRAW: 0}}

        # Player makes first move:
        _initial_printout(initial)
        self._build_tree_recursive(initial,
                                   current_player=self._player,
                                   initial_player=self._player)
        if self._verbose:
            print("\n")
            self.print_win_losses()

        # Opponent makes first move:
        self._init_child_states = {}
        # keys: state after each possible first move of the opponent
        # values: (action, Mark.[opponent])
        # i.e. the edge labels for the children of the initial state where the opponent goes first.

        for i in range(3):
            for j in range(3):
                action = (i, j)
                state = initial.clone_and_move(action, GameTree.opponent(self._player))
                self._initial.append(state)
                self._terminal[state] = state.check_endstate()
                self._children[state] = {}
                self._parents[state] = [initial]
                _initial_printout(state)
                self._init_child_states[state] = (action, GameTree.opponent(self._player))
                self._build_tree_recursive(state,
                                           current_player=self._player,
                                           initial_player=GameTree.opponent(self._player))
                if self._verbose:
                    print("\n")
                    self.print_win_losses()

    def _build_tree_recursive(self, state, current_player, initial_player):
        if self._terminal[state] is not None:
            self._game_outcomes[initial_player][self._terminal[state]] += 1
            return

        if self._verbose and np.random.rand() < .0001:
            n_games = sum([sum([self._game_outcomes[p][r] for r in [Result.X_WIN, Result.O_WIN, Result.DRAW]])
                          for p in [Mark.X, Mark.O]])
            print("\t\tgames finished:  %i\t\tunique states: %i" % (n_games, len(self._terminal)))

        for action in state.get_actions():
            child = state.clone_and_move(action, current_player)
            self._children[state][child] = (action, current_player)
            if child not in self._terminal:
                self._terminal[child] = child.check_endstate()
                self._children[child] = {}
                self._parents[child] = []
            self._parents[child].append(state)
            self._build_tree_recursive(child, 3 - current_player, initial_player)

    def get_game_tree(self, generic=False):
        """
        Return the full game tree.
        If Generic, remove the opponent-moves-first states from self._initial, add the single initial state to their parents, them
        to its children.
        """
        if generic:
            initial = [self._initial[0]]
            children = deepcopy(self._children)
            children[initial[0]].update(self._init_child_states)
        else:
            initial = self._initial
            children = self._children

        return self._terminal, children, self._parents, initial

    def print_win_losses(self):
        print("Total unique states: ", len(self._terminal))
        print("\tterminal, X-wins: ", len([state for state in self._terminal if self._terminal[state] == Result.X_WIN]))
        print("\tterminal, O-wins: ", len([state for state in self._terminal if self._terminal[state] == Result.O_WIN]))
        print("\tterminal, draw: ", len([state for state in self._terminal if self._terminal[state] == Result.DRAW]))

    def get_outcome_counts(self):
        return self._game_outcomes


def get_game_tree_cached(player, verbose=False):
    filename = f"game_tree_{player.name}.pkl"
    if os.path.exists(filename):
        print("Loading game tree from cache file: ", filename)
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print("\tloaded game tree from cache file: %s" % filename)
        return data
    else:
        print("Cache file not found: ", filename)
        print("\tgenerating game tree...")
        tree = GameTree(player, verbose=verbose)
        print("\n\n==========================================================")
        print("Saving game tree to cache file: ", filename)
        with open(filename, "wb") as f:
            pickle.dump(tree, f)
        print("\tsaved game tree to cache file: ", filename)
        return tree


def test_game_tree():
    player = Mark.X
    opponent = GameTree.opponent(player)
    tree = get_game_tree_cached(player, verbose=True)  # GameTree(player, verbose=True)
    print('==========================================================')
    tree.print_win_losses()
    terminal, children, parents, initial = tree.get_game_tree()
    print('==========================================================')
    results = tree.get_outcome_counts()
    x_wins = results[Mark.X][Result.X_WIN] + results[Mark.O][Result.X_WIN]
    o_wins = results[Mark.X][Result.O_WIN] + results[Mark.O][Result.O_WIN]
    draws = results[Mark.X][Result.DRAW] + results[Mark.O][Result.DRAW]
    print("Games played: %i\n" % (x_wins + o_wins + draws))
    print("\t%s goes first:" % player.name)
    print("\t\tX wins: " + str(results[player][Result.X_WIN]))
    print("\t\tO wins: " + str(results[player][Result.O_WIN]))
    print("\t\tDraws: %s\n" % str(results[player][Result.DRAW]))
    print("\t%s goes first:" % opponent.name)
    print("\t\tX wins: " + str(results[opponent][Result.X_WIN]))
    print("\t\tO wins: %s" % str(results[opponent][Result.O_WIN]))
    print("\t\tDraws: " + str(results[opponent][Result.DRAW]))
    print("\n\tTotals:")
    print("\t\tX wins: ", x_wins)
    print("\t\tO wins: ", o_wins)
    print("\t\tDraws: %i\n" % draws)

    # Show all 32 draw states:
    print("Creating Draw States image...")
    filename = "draw_states_all.png"

    cell_size = 20
    cell_dims = Game.get_image_dims(cell_size)
    grid_pad = int(cell_dims['line_t'] * 1.5)
    draw_states = [state for state in terminal if terminal[state] == Result.DRAW]
    draw_state_img_size = 4*(cell_dims['img_size'] + 2 * grid_pad)
    pad = 4  # on all sides

    draw_state_imgs = [state.get_img(cell_dims,
                                     color_bg=COLOR_BG,
                                     color_lines=COLOR_LINES,
                                     draw_box=False) for state in draw_states]

    img_size = draw_state_imgs[0].shape[:2][::-1]
    draw_img = np.zeros((draw_state_img_size, 2*draw_state_img_size, 3), dtype=np.uint8)
    draw_img[:] = COLOR_BG
    img_num = 0
    for i in range(4):
        for j in range(8):
            # if img_num >= len(draw_state_imgs):
            #    break

            img_x = j * (img_size[0] + 2 * grid_pad) + grid_pad
            img_y = i * (img_size[1] + 2 * grid_pad) + grid_pad

            draw_img[img_y:img_y + img_size[1], img_x:img_x + img_size[0]] = draw_state_imgs[img_num]
            img_num += 1
    cv2.imwrite(filename, draw_img[:, :, ::-1])
    print("\tsaved to: ", filename)
    cv2.imshow("Draw states", draw_img[:, :, ::-1])
    cv2.waitKey(0)


def make_image():
    game = Game.from_strs(["X  ",
                           "   ",
                           "   "])
    print(game)

    dims = game.get_image_dims(17)
    img = game.get_img(dims)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test_game_tree()
