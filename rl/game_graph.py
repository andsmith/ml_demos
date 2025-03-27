from game_base import Mark, Result
from tic_tac_toe import get_game_tree_cached, Game
import numpy as np
import logging
from tic_tac_toe import Game, GameTree
import logging
import cv2
from colors import COLOR_LINES, COLOR_BG, COLOR_X, COLOR_O, COLOR_DRAW

from node_placement import BoxOrganizerPlotter
LAYOUT = {'win_size': (1900, 950)}


class GameGraphApp(object):
    """
    Arrange game states to show the graph structure of the game.

    States will be in layers, all states in a layer will have the same number of marks made so far.
    States with a single parent will be approximately under their parent.
    States with multiple parents will be placed in the middle of their parents.

    Lines will connect states to their successors.

    """

    def __init__(self,max_levels=10):
        """
        :param size: (width, height) of the app window
        """
        self._tree = get_game_tree_cached(player=Mark.X)
        self._max_levels = max_levels
        if max_levels >10:
            raise ValueError("Max levels must be <= 10 for Tic-Tac-Toe.")
        self._size = LAYOUT['win_size']
        self._blank_frame = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        self._blank_frame[:, :] = COLOR_BG
        self._init_tree()
        self._init_graphics()

    def _init_graphics(self):
        print("Drawing images...")
        self._out_frame=self._box_placer.draw(images=self._state_images, dest=self._blank_frame.copy())
        print("")

    def _init_tree(self):
        """
        Calculate sizes & placement of all states.
            1. Determine number of layers & number of states in each layer
            2. Determine positions of each state
            3. For each state, get lines to draw to it's children states.
        """
        def _get_layer(state):
            return np.sum(state.state != Mark.EMPTY)

        # 1.
        self._term, self._children, self._parents, self._initial = self._tree.get_game_tree(generic=True)
        states_by_layer = [[{'id': s,
                             'state': s,
                             'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
                            for s in self._term if _get_layer(s) == l] for l in range(self._max_levels)]
        layer_counts = np.array([len(states) for states in states_by_layer])
        logging.info(f"States by layer: {layer_counts}")

        # 2. First get the layer spacing.
        self._layer_spacing = self._calc_layer_spacing(layer_counts)
        self._box_placer = BoxOrganizerPlotter(states_by_layer, spacing=self._layer_spacing, size_wh=self._size)

        # 3. get the size of a box in each layer, then make the images
        self._positions, self._box_dims, _ = self._box_placer.get_layout()
        self._state_images = {}

        logging.info("Generating images...")
        for l_ind, layer_states in enumerate(states_by_layer):
            logging.info(f"\tLayer {l_ind} has {len(layer_states)} states")
            box_size = self._box_dims[l_ind]['box_side_len']
            logging.info(f"\tBox_size: {box_size}")
            space_size = Game.get_space_size(box_size)
            logging.info(f"\tUsing space_size: {space_size}")
            box_dims = Game.get_image_dims(space_size, bar_w_frac=.2)
            logging.info(f"\tUsing tiles of size: {box_dims['img_size']}")


            for state_info in layer_states:
                
                state = state_info['state']
                #import pprint
                #pprint.pprint(box_dims[l_ind]['])
                self._state_images[state] = state.get_img(box_dims)

        
        print("\tmade ", len(self._state_images))


    def run(self):
        """
        Run the app
        """
        cv2.imshow("Game Graph", self._out_frame[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _calc_layer_spacing(self, layer_counts):
        """
        for each layer , a dict{'y': (ymin, y_max) the y-extent of the layer in pixels,
                                'n_boxes': number of boxes in the layer,
                                'bar_y': (ymin, y_max) of the bar that will be drawn under the boxes (if not at the bottom)
                                }
        """
        rel = np.sqrt(100 + layer_counts)
        rel = rel/np.sum(rel)
        sizes = (rel * self._size[1]).astype(int)
        y = 0
        bar_w = 5
        layer_spacing = []
        for i, size in enumerate(sizes):
            top = y
            if i < len(sizes) - 1:
                bottom = y + size - bar_w
                bar = (y + size - bar_w, y + size)
            else:
                bottom = self._size[1]
                bar = None
            spacing={'y': (top, bottom),
                                  'n_boxes': layer_counts[i]}
            if bar is not None:
                spacing['bar_y'] = bar
            layer_spacing.append(spacing)
            y += size
        return layer_spacing


def run_app():
    app = GameGraphApp()
    app.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
    logging.info("Exiting.")
