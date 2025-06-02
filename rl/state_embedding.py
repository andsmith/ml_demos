import logging
import numpy as np
import cv2
from layout import LAYOUT
from game_util import get_box_placer, sort_states_into_layers
from mouse_state_manager import MouseBoxManager
from state_key import StateKey
from color_key import ColorKey, ProbabilityColorKey
from drawing import GameStateArtist
from tic_tac_toe import Game
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
from gameplay import ResultSet, Match
from colors import COLOR_SCHEME
from layer_optimizer import SimpleTreeOptimizer


SPACE_SIZES = [7, 2, 2, 2, 2, 3]  # Sizes for the state embedding layers


class StateEmbedding(object):
    """
    Determine a good location for all reachable (RL) game states for player X given the image size.

    Resize this when the tab panel changes size.
    StateArtists (and subclasses) look up size from this class to draw images.

    """

    def __init__(self, env, key_size=(0, 0)):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)
        :param key_size:  (width, height) tuple, space reserved in the upper right corner for keys.

        """
        self.size = None
        self._env = env
        self.box_placer = None
        self._layout = LAYOUT['state_embedding']
        self.key_size = key_size
        self.terminal_states = env.get_terminal_states()
        self.updatable_states = env.get_nonterminal_states()
        self.all_states = self.updatable_states + self.terminal_states
        self.states_by_layer = sort_states_into_layers(self.all_states)
        self.layers_per_state = {state['id']: i for i, layer in enumerate(self.states_by_layer) for state in layer}
        self.box_sizes, self.artists = self._get_box_sizes()

        # Avoid recalculating full state embedding.
        self._size_cache = {}
        logging.info(f"StateEmbedding initialized with {len(self.all_states)} states")

    def set_size(self, new_size):
        # cache the box layouts.
        if new_size != self.size or new_size not in self._size_cache:
            logging.info(f"StateImageManager resized to {new_size}")
            self.size = new_size
            if new_size in self._size_cache:
                self.box_placer = self._size_cache[new_size]
            else:
                self.box_placer = self._calc_dims()
                self._size_cache[new_size] = self.box_placer
            return True
        return False

    def _calc_dims(self):
        """
        Calculate state layout.
        """
        # Place all game states into the image region
        box_placer, _, _ = get_box_placer(self.size,
                                          self.all_states,
                                          box_sizes=self.box_sizes,
                                          layer_vpad_px=1,
                                          layer_bar_w=1,
                                          player=self._env.player, key_size=self.key_size)

        # Swap state positions so wins are on the right, losses on the left, draws in the middle, etc.
        terminal_lut = {state: state.check_endstate() for state in self.all_states}
        tree_opt = SimpleTreeOptimizer(image_size=self.size,
                                       states_by_layer=self.states_by_layer,
                                       state_positions=box_placer.box_positions,
                                       terminal=terminal_lut)
        new_positions = tree_opt.get_new_positions()
        box_placer.box_positions = new_positions

        return box_placer

    def _get_box_sizes(self):
        artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in SPACE_SIZES]
        box_sizes = [artists[layer_no].get_image(Game()).shape[0] for layer_no in range(len(artists))]
        return box_sizes, artists


def test_state_embedding():
    img_size = (1200, 980)
    from reinforcement_base import Environment
    from baseline_players import HeuristicPlayer
    from game_base import Mark
    from mouse_state_manager import MouseBoxManager
    agent = HeuristicPlayer(mark=Mark.X, n_rules=1)
    opponent = HeuristicPlayer(mark=Mark.O, n_rules=2)
    env = Environment(opponent_policy=opponent, player_mark=Mark.X)
    embed = StateEmbedding(env, key_size=(300, 70))
    embed.set_size(img_size)
    mouse = MouseBoxManager(None)
    mouse.set_boxes(embed.box_placer.box_positions)

    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    terminals, nonterminals = env.get_terminal_states(), env.get_nonterminal_states()
    all_states = terminals + nonterminals
    # for state in all_states:
    mouse.render_state(img, nonterminals, 1)
    mouse.render_state(img, terminals, 2)

    cv2.imshow("State Embedding Test", img[:, :, ::-1])  # Flip the image horizontally
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_state_embedding()
    # Example usage:
    # app = MyApp()  # Replace with your application instance
    # env = Environment()  # Replace with your environment instance
    # embedding = StateEmbedding(env)
    # embedding.set_size((800, 600))
    # print(embedding.box_placer.box_positions)  # Check the box positions
