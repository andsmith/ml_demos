from game_base import Mark, Result
from tic_tac_toe import get_game_tree_cached, Game
import numpy as np
import logging

import cv2

'''
class InteractiveGameGraph(object):
    """
    The game graph shows the all possible (reinforcement, not game) states and transition on a single image.

    As RL algorithms are applied, the value function v(s) of each state can be visualized. 
    [FUTURE:  Mouseover to show transition probabilities of each action according to the policy.]

    This will be visually incomprehensiblem, so the user can click to select/deselect states.  
    When a state is selected, all states leading back to an initial state are also selected.  
    Selected states are shown blown up, superimposed over the full graph.  
    Edges between selected states are also made more visible.

    Selected states have a "status" area underneath them, which shows the value function, policy, etc.

    LAYOUT:
        1. The top row will show the 10 possible initial states, the next four rows show the RL agent's possible states after
           1, 2, 3, 4 rounds of play (1 agent move and one opponent move, unless the first move ends the game).
        2. The second row will show the 720 states after the first round, in 10 groups below each group's initial state. 

    """

    def __init__(self, size=(1900, 880), player=Mark.X):
        self._grid_size_range = 19, 100  # (smallest, largest)
        self._size = size
        self._bkg_color = (255, 255, 255)
        self._line_color = (0, 0, 0)
        self._player = player
        self._terminal, self._children, self._parents, self._initial = Game.get_game_tree(self._player)


def _get_layer_num(game_state):
    # TODO:  Make every time it's x's turn aligned.
    return np.sum(game_state.state != Mark.EMPTY)

''''''
class GameGraph(object):
    """
    Arrange game states to show the graph structure of the game.

    States will be in layers, all states in a layer will have the same number of marks made so far.
    States with a single parent will be approximately under their parent.
    States with multiple parents will be placed in the middle of their parents.

    Lines will connect states to their successors.

    """
    _DIMS = {'layer_space_px': 2,
             'min_space_size': 6,  # a "space" is 1/3 of a grid cell.
             'max_space_size': 12,  # min layer_size = (max_space_size + 2 * layer_space_px)
             'grid_padding_frac': 0.05,   # multiply by grid_size to get padding on either side of a grid.
             'bar_width': 8}

    def __init__(self, size=(1900, 880), player=Mark.X):
        self._player = player
        self._tree = get_game_tree_cached(self._player)
        self._states, self._children, self._parents, self._initial = self._tree.get_game_tree()
        self._size = size
        # sort states by layers, determine layer (vertical) spacing.
        self._layer_numbers = {s: _get_layer_num(s) for s in self._states}
        self._layers = [[s for s in self._states if self._layer_numbers[s] == i] for i in range(10)]
        l_sizes = [len(l) for l in self._layers]
        print("Layer sizes:")
        print("\n\t".join([f"{i}: {s}" for i, s in enumerate(l_sizes)]))

        self._layer_spacing = self._calc_layer_spacing()
        import pprint
        pprint.pprint(self._layer_spacing)

        # within each layer determine grid sizing

        # then which grid cell each state will be placed in, depeding on where it's parents are.

    def _calc_layer_spacing(self):
        """
        Find vertical spacing of the layer boundaries.
        1 Determine the most maximum-sized grids that could fit on one row, the vertical space is the minimum row size.
        2 Determine how many rows have the minimum or fewer, these are the minimum-sized rows.
        3 Determine the extra space, divide it among the rest of the rows proportionally to the number of states in each row.
        """
        grid_padding_fraction = self._DIMS['grid_padding_frac']
        min_space_size = self._DIMS['min_space_size']
        max_space_size = self._DIMS['max_space_size']
        bar_width = self._DIMS['bar_width']
        layer_sp = self._DIMS['layer_space_px']

        max_side_len, _ = Game.get_grid_dims(max_space_size)
        max_grid_pad = int(max_side_len * grid_padding_fraction)

        # 1. Find the number of max-sized grids that can fit on one row horizontally,
        n_grids = np.floor((self._size[0] - 2 * layer_sp) // (max_side_len + 2 * max_grid_pad)).astype(int)
        # then the height of this row is the minimum (vertical) space:
        tiny_row_h = max_side_len + 2 * max_grid_pad + 2 * layer_sp
        if tiny_row_h * len(self._layers) > self._size[1]:
            raise ValueError(f"Too many layers to fit in the vertical space: {tiny_row_h * len(self._layers)} > {self._size[1]}")
        print("Tiny row_height:", tiny_row_h)
        import ipdb
        ipdb.set_trace()
        # 2. Find the number of rows that have the minimum or fewer grids:
        tiny_layers = [i for i, l in enumerate(self._layers) if len(l) <= n_grids]
        n_big_layers = len(self._layers) - len(tiny_layers)

        # 3. Find the remaining space, divide it among the remaining rows:
        remaining_states = np.array([len(l) for l in self._layers])
        remaining_states[tiny_layers] = 0
        used_v_space = len(tiny_layers) * tiny_row_h + (len(tiny_layers) - 1) * bar_width if len(tiny_layers) > 0 else 0
        remaining_v_space = self._size[1] - used_v_space
        rel_heights = remaining_states / np.sum(remaining_states)
        slack_space = remaining_v_space - n_big_layers * tiny_row_h  # only divide up area in excess of tiny_row_h
        non_tiny_heights = slack_space * rel_heights + tiny_row_h  # Add divided proportion to tiny_row_h

        # 4. Put it all together
        tiny_row_h = max_side_len + 2 * max_grid_pad + 2 * layer_sp
        y = 0
        v_pos = []
        for l_ind in range(len(self._layers)):
            row_dims = {}
            if l_ind in tiny_layers:
                row_dims['space_size'] = max_space_size
                row_dims['grid_pad'] = max_grid_pad
                row_dims['side_len'] = max_side_len
                row_dims['layer_y'] = y, y + tiny_row_h
                y += tiny_row_h
                if l_ind != len(self._layers) - 1:
                    row_dims['bar_y'] = y, y + bar_width
                    y += bar_width
            else:
                row_dims['layer_y'] = y, y + non_tiny_heights[l_ind]
                y += non_tiny_heights[l_ind]
                if l_ind != len(self._layers) - 1:
                    row_dims['bar_y'] = y, y + bar_width
                    y += bar_width
                # other keys filled in later
            v_pos.append(row_dims)
        return v_pos


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    gg = GameGraph()
    # img = gg.build_graph()
    # cv2.imshow('game graph', img)
    # cv2.waitKey(0)
'''