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
from reinforcement_base import  get_game_tree_cached
from drawing import GameStateArtist


BOX_SIZES = [22, 12, 12, 12, 12, 15]  # good for single value function
# BOX_SIZES =  [20, 11, 7, 7, 8, 14] # good for 2-value function windows.

def sort_states_into_layers(state_list, player_mark=Mark.X, key='id'):

    layers = []
    for n_marks in range(6):
        layer = []
        for state in state_list:
            if np.sum(state.state == player_mark) == n_marks:
                layer.append({key: state})
        layers.append(layer)
        print("Layer %i has %i states." % (n_marks, len(layer)))
    return layers

def get_box_placer(img_size, all_states, box_sizes=None, layer_vpad_px=1,
                       layer_bar_w=1, player=Mark.X):
    """
    Get the dict of box positions {x:(xmin, max), y:(ymin, max)} for each state, from a player's POV (RL states)
    Use the FixedCellBoxOrganizer to place the boxes in a grid, return it's .box_positions attribute.
    :param img_size:  The size of the output image.
    :param box_sizes:  List of icon image sizes for each layer (0-5)
    :param all_states:  List of states to be placed.
    :param layer_vpad_px:  Vertical padding between layers.
    :param layer_bar_w:  Width of the bar between layers.
    :param color_bg:  Background color of the image.
    :param color_lines:  Color of the lines between boxes.
    :param player:  The player for the agent.  The opponent is the other player.
    :return:  The box positions for each state, a dict from the Game (state) to the x & y spans.
    """
    box_sizes = BOX_SIZES if box_sizes is None else box_sizes
    state_layers = sort_states_into_layers(all_states, player_mark=player)
    box_placer = FixedCellBoxOrganizer(img_size, state_layers, box_sizes,
                                           layer_vpad_px=layer_vpad_px, layer_bar_w=layer_bar_w)
    return box_placer, box_sizes, state_layers


def optimize_layers(box_placer, all_states, player=Mark.X):
    """
    """

def get_state_icons(state_layers, box_sizes=None, player=Mark.X):
    
    space_sizes = [GameStateArtist.get_space_size(box_size, bar_w_frac=0.0) for box_size in box_sizes]
    artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in space_sizes]
    images = {state['id']: artists[layer_no].get_image(state['id']) for layer_no, layer in enumerate(state_layers) for state in layer}
    return images

