"""
Visualize the value function of a policy as it's updated.
"""
import logging
from colors import COLOR_BG, COLOR_LINES, RED, GREEN, MPL_BLUE_RGB, MPL_GREEN_RGB, MPL_ORANGE_RGB
from game_base import Mark, Result
from layer_optimizer import SimpleTreeOptimizer  # for horizontal sorting
from node_placement import FixedCellBoxOrganizer  # for 2d embedding and vertical sorting
import cv2
import matplotlib.pyplot as plt
import numpy as np
from reinforcement_base import Environment, GameTree, get_game_tree_cached
from drawing import GameStateArtist


class StateFunction(object):
    """
    Wrap the value tables (V(s), etc) with methods for making the images.

    creates a "state" view and "value" view for each image name.

    Organize states in the following way (assuming player_mark = Mark.X):
    - layers have all states with the same number of Xs, in increasing order down.
    - States with more Os than Xs are on the right. 
    - Terminal X-wins are on the left.
    - Terminal O-wins are on the right.
    - Draws are in the middle.
    """

    def __init__(self, img_sizes, state_list, player_mark=Mark.X):
        """
        :param img_sizes:  dict of image sizes.  StateFunction will create one image per key/value pair {'name': (width, height), ...}
        :param state_list:  list of states to be visualized.  Each state is a Game object.
        :param player_mark:  The player for the agent.  The opponent is the other player.
        """
        self._sizes = img_sizes
        self._state_list = state_list
        self._player = player_mark
        self._color_bg = COLOR_BG
        self._color_lines = COLOR_LINES

        self._cmap = plt.get_cmap('hot')

        self._box_sizes = [20, 11, 7, 7, 8, 14]
        self._space_sizes = [GameStateArtist.get_space_size(box_size, bar_w_frac=0.0) for box_size in self._box_sizes]
        print("Space sizes: ", self._space_sizes)
        self._artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in self._space_sizes]
        self._info = {name: self._get_state_info(name) for name in img_sizes}

    def _set_colors(self):
        # map value function values to RGB for each state (for now random)
        raise NotImplementedError("StateFunction._set_colors() not implemented.")
        # TODO:  modify BoxOrganizer to let colors be updated.

    def _get_state_color(self, state):
        # for now a random value in our color map.
        rgb_f = self._cmap(int(np.random.random_integers(0, 255, 1)))[:3]
        return (rgb_f[0] * 255, rgb_f[1] * 255, rgb_f[2] * 255)

    def update(self, name, updates):
        """
        Update the value function for the given state.

        :param name:  The name of the image to update.
        :param updates:  The new values for the state.
        """
        if name not in self._sizes:
            raise ValueError(f"Image {name} not found in sizes.")
        raise NotImplementedError("StateFunction.update() not implemented.")

    def _get_state_info(self, img_name):
        """
        Set state layouts.
        """
        blank = np.zeros((self._sizes[img_name][1], self._sizes[img_name][0], 3), dtype=np.uint8)
        blank[:] = self._color_bg

        layers = []
        state_images = {}
        for n_marks in range(6):
            layer = []
            layer_no = len(layers)
            for state in self._state_list:
                if np.sum(state.state == self._player) == n_marks:
                    layer.append({'id': state,
                                  'color': self._get_state_color(state)})
                    state_images[state] = self._artists[layer_no].get_image(state)
            layers.append(layer)
            print("Layer %i had %i states." % (n_marks, len(layer)))

        box_placer = FixedCellBoxOrganizer(self._sizes[img_name], layers, self._box_sizes,
                                           layer_vpad_px=1, layer_bar_w=1,
                                           color_bg=self._color_bg, color_lines=self._color_lines)
        image = box_placer.draw()
        image_s = box_placer.draw(images=state_images)
        # TODO:  layer placement optimization here

        return {'layers': layers,
                'value_image': image,
                'state_image': image_s,
                'images': state_images,
                'box_placer': box_placer}
    
    def get_image(self, name, which='values'):
        """
        Get the image for the given name.

        :param name:  The name of the image to get.
        :param which:  The type of image to get ('values' or 'states').
        :return:  The image for the given name.
        """
        if name not in self._sizes:
            raise ValueError(f"Image {name} not found in sizes.")
        if which == 'values':
            return self._info[name]['value_image']
        elif which == 'states':
            return self._info[name]['state_image']
        else:
            raise ValueError(f"Invalid image type {which}.")    



class StateFunctionTester(object):
    def __init__(self):
        from baseline_players import HeuristicPlayer
        opponent_policy = HeuristicPlayer(mark=Mark.O, n_rules=6, p_give_up=0.0)
        seed_policy = HeuristicPlayer(mark=Mark.X, n_rules=2, p_give_up=0.0)
        player = Mark.X
        # P.I. initialization:
        self._env = Environment(opponent_policy, player)
        self.children = self._env.get_children()

        self._pi = seed_policy
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        all_states = [gamestate for gamestate in self.updatable_states + self.terminal_states]
        self._sf = StateFunction({'test': (758, 824)}, all_states, player_mark=player)


def test_state_function():
    """
    Test the StateFunction class.
    """
    sft = StateFunctionTester()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_state_function()
    # test_state_function()
