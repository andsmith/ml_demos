"""
Classes for Reinforcement Learning policy and value functions for Tic Tac Toe.

WLOG, let the policy's player be X.


"""
import numpy as np
from util import get_annulus_polyline
import matplotlib.pyplot as plt
from game_base import Mark, Result
import logging
import pickle
import os
from copy import deepcopy

# Defaults to get_img


class Game(object):
    """
    Tic-tac-toe game board.
    """

    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8) + Mark.EMPTY

    def __hash__(self):
        return hash(tuple(self.state.flatten().tolist()))

    def __eq__(self, other):
        return np.all(self.state == other.state)
    
    def greater_than(self, other):
        # Compare two game states lexicographically, for embedding.
        return np.lexsort((self.state.flatten(),)) > np.lexsort((other.state.flatten(),))

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

    from colors import COLOR_LINES, COLOR_BG
    from drawing import GameStateArtist

    artist = GameStateArtist(cell_size)
    cell_dims = artist.dims
    print(cell_dims)
    grid_pad = int(cell_dims['line_t'] * 1.5)
    draw_states = [state for state in terminal if terminal[state] == Result.DRAW]
    draw_state_img_size = 4*(cell_dims['img_size'] + 2 * grid_pad)
    pad = 4  # on all sides
    draw_state_imgs = [artist.get_image(state, draw_box=False) for state in draw_states]

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
    import cv2
    cv2.imwrite(filename, draw_img[:, :, ::-1])
    print("\tsaved to: ", filename)
    cv2.imshow("Draw states", draw_img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    test_game_tree()
