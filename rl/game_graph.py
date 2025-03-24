from game_base import Mark, Result
from tic_tac_toe import Game
import numpy as np
import logging

import cv2


class GameGraph(object):
    """
    Arrange game states to show the graph structure.

    The top row will show each possible initial state.
    The second row will show the states that can be reached from the initial states, and so on.
    Lines will connect states to their successors.
    """

    def __init__(self, player=Mark.X):
        self._player = player
        self._terminal, self._children, self._parents, self._initial = Game.get_game_tree(self._player)

        self._bkg_color = (255, 255, 255)
        self._line_color = (0, 0, 0)

    def _is_terminal(self, state):
        if state in self._initial:
            return False
        return self._terminal[state] is not None

    def _get_layers(self, depth):
        """
        Round 0 at the top, etc.
        :param depth: number of rounds (1 means only initial states)
        :return: list of lists, each list contains the states of a round
           dict(state: round_num/layer)
        """
        # Breadth first search, organize by layers
        # import ipdb; ipdb.set_trace()
        states_layers = {state: 0 for state in self._initial}
        layers = [self._initial]
        for l in range(depth-1):
            next_layer = []
            for state in layers[-1]:
                for (child_state, _) in self._children[state]:
                    if child_state not in states_layers:
                        states_layers[child_state] = l+1
                        next_layer.append(child_state)
            layers.append(next_layer)

        return layers, states_layers

    def _get_positions(self, graph_layers, all_states, img_size):
        """
        Space out the initial states in the top row.
        Below each state, in the next row, place the successor states in rows then columns.
        If a successor state is already placed, don't place it again.
        Attempt to make each layer the right size for the number of states. (so every state/grid can be the same size)
        :returns: dict(state: (x, y)) (upper-left corner of the state grid's position in the image)
        """
        pos = {}
        layer_padding = 10
        layer_bar_w = 4
        w, h = img_size
        max_grid_size = 100

        def _get_dims(grid_size, grid_padding):
            # place each row spaced evenly if only one row
            # else stack into rows/cols with padding]
            x, y = layer_padding, layer_padding
            state_positions = {}
            bar_positions = []  # [{'x_span': (x1, x2), 'y_span': (y1, y2)}, ...]
            for l, layer in enumerate(graph_layers):
                max_cols = (w - 2 * layer_padding) // (grid_size+grid_padding)
                n = len(layer)
                if n <= max_cols:
                    extra_space = w - n * grid_size - 2 * layer_padding
                    extra_padding = extra_space // (n -1)
                    x_positions = np.arange(n) * (grid_size + extra_padding) + layer_padding
                    y_positions = y * np.ones(n)
                    for i, state in enumerate(layer):
                        state_positions[state] = (x_positions[i], y_positions[i])
                    y += grid_size+layer_padding
                else:
                    n_cols = max_cols
                    n_rows = np.ceil(n / n_cols).astype(int)
                    x_positions = np.arange(n_cols) * (grid_size + grid_padding) + layer_padding
                    y_positions = np.arange(n_rows) * (grid_size + grid_padding) + layer_padding
                    print(x_positions[-1], y_positions[-1])
                    for i, state in enumerate(layer):
                        row = i // n_cols
                        col = i % n_cols
                        state_positions[state] = (x_positions[col], y_positions[row])
                    y += n_rows*(grid_size+layer_padding)
                # draw a bar to separate the layers
                if l < len(graph_layers) - 1:
                    bar_positions.append({'x_span': (layer_padding, w-layer_padding),
                                          'y_span': (y, y+layer_bar_w)})
                    y += layer_bar_w + layer_padding

            return state_positions, y, bar_positions

        # find the right grid size
        grid_size = max_grid_size
        while True:

            grid_padding = max_grid_size // 4
            
            state_positions, y, bar_positions = _get_dims(grid_size, grid_padding)
            print("Testing grid size %s:  %i < %i?" % (grid_size, y, h))
            if y < h:
                break
            grid_size -= 1

        return {'positions': state_positions,
                'grid_size': grid_size,
                'grid_padding': grid_padding,
                'bar_positions': bar_positions}

    def build_graph(self, img_size=(3000, 2000), depth=5):
        img = np.zeros((img_size[1], img_size[0], 3), np.uint8)
        img[:] = self._bkg_color

        graph_layers, all_states = self._get_layers(depth)
        positions = self._get_positions(graph_layers, all_states, img_size)
        self._draw(img, graph_layers, positions)
        return img

    def _draw(self, img, layers, pos):
        """
        :param img: numpy array (height, width, 3)
        :param layers: list of lists of states
        :param pos: dict(state: (x, y))
        """
        grid_size = pos['grid_size']
        grid_padding = pos['grid_padding']
        state_positions = pos['positions']
        bar_positions = pos['bar_positions']

        def _draw_state(img, state, x, y):
            state_img = state.get_img(grid_size//3)
            g_h, g_w = state_img.shape[:2]
            x, y = int(x), int(y)   
            print(x, g_h, y, g_w)
            img[y:y+g_h, x:x+g_w] = state_img

        for l, layer in enumerate(layers):
            for state in layer:
                x, y = state_positions[state]
                _draw_state(img, state,x, y)

            for bar in bar_positions:
                x1, x2 = bar['x_span']
                y1, y2 = bar['y_span']
                img[x1:x2, y1:y2] = self._line_color


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    gg = GameGraph()
    img = gg.build_graph()
    cv2.imshow('game graph', img)
    cv2.waitKey(0)
