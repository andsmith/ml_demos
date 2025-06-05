import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_SCHEME
from layout import LAYOUT, WIN_SIZE
from game_base import Mark, Result, TERMINAL_REWARDS
from game_util import get_box_placer, get_state_icons, sort_states_into_layers
from layer_optimizer import SimpleTreeOptimizer
import numpy as np
from tic_tac_toe import Game
import time
import logging
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from drawing import GameStateArtist
from scipy.spatial import KDTree
from color_key import ColorKey, ProbabilityColorKey
from state_key import StateKey
from gameplay import ResultSet, Match
from tab_content import TabContentPage
from state_embedding import StateEmbedding


class FullStateContentPage(TabContentPage):
    """
    FullStateContentPage is a base class for content tabs that display the full state embedding.
    """
    def __init__(self, alg, embedding,keys,bg_color=None):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param bg_color:  The background color to use for the image. Defaults to COLOR_BG.
        :param embedding:  The StateEmbedding instance to use for drawing states.
        """
        super().__init__(alg, keys)
        self._app = alg.app
        self._env = alg.app.env
        all_states = self._env.get_terminal_states() + self._env.get_nonterminal_states()
        self._displayed_states = {state: None for state in all_states}
        self._bg_color = bg_color if bg_color is not None else COLOR_SCHEME['bg']
        self._embed = embedding
        self._layout = LAYOUT['state_embedding']
        self._state_icons = self._get_state_icons()
        self._base_image = None  # Base image for the content page, used for mouseover and marking.
        self._marked_image = None
        self._cur_state = None  # The state the algorithm is currently on, if set.
        self._cur_box = None  # The box for the current state, if set.

    def _get_state_icons(self):
        images = {state['id']: self._embed.artists[layer_num].get_image(
            state['id']) for layer_num, state_list in enumerate(self._embed.states_by_layer) for state in state_list}
        return images
    
    def resize(self, new_size):
        # embedding should already be changed, just send its boxes to the mouse manager.
        super().resize(new_size)
        box_placer = self._embed.box_placer
        if box_placer is not None:
            self._mouse_manager.set_boxes(box_placer.box_positions)


    def _draw_base(self):
        size = self._embed.size
        logging.info(f"Regenerating base image.")
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bg_color
        img = self._embed.box_placer.draw(images=self._state_icons, colors=None, show_bars=False, dest=img)
        return img

    def set_current_state(self, state=None):
        old_cur_state = self._cur_state
        self._cur_state = state

        if old_cur_state is None and state is None:
            return

        if (state is None and old_cur_state is not None) or\
            (state is not None and old_cur_state is None) or \
                (old_cur_state != state):
            self.clear_images(marked_only=True)

    def _get_key_value(self, key_name):
        """
        Subclasses with additional/different keys should override this.
        """
        if key_name=='state':
            return self._mouse_manager.mouseover_id
        elif key_name=='embedding':
            pass # static key
        else:
            raise ValueError(f"Unknown key name: {key_name}. Only 'state' is supported in FullStateContentPage.")
        

class ValueFunctionContentPage(FullStateContentPage):
    def __init__(self, alg, embedding, keys,values,  bg_color=None, as_deltas=False):
        """
        :param app:  The application instance.
        :param embedding:  The StateEmbedding instance to use for drawing states.
        :param keys:  Dictionary of keys to display in the key area.
            NOTE:  1 must be 'values', to use its color key.
        :param values:  Dictionary of state values, indexed by state.  


        :param as_deltas:  If True, display the value function as deltas from the previous state.
            * Values (as_deltas=False):  show the value/terminal-reward for every state in 
              the embedding as a colored square, using a ColorKey object to manage the color mapping.
            * Deltas show
        """
        bg_color = bg_color if bg_color is not None else COLOR_SCHEME['func_bg']
        super().__init__(alg=alg, embedding=embedding, keys=keys, bg_color=bg_color)
        self._values = values  # Dictionary of state values, indexed by state.
        self._color_key = keys['values']
        self._as_deltas = as_deltas

    def update_values(self, values):   
        self._values.update(values)
        logging.info(f"Setting values for {len(values)} states.")
        self.clear_images(marked_only=False)

    def _get_key_value(self, key_name):
        if key_name=='values':
            state= self._mouse_manager.mouseover_id
            if state is None or state not in self._values:
                return None
            return self._values[state]
        elif key_name=='state':
            return super()._get_key_value(key_name)
        else:
            raise ValueError(f"Unknown key name: {key_name}. Only 'state' is supported in FullStateContentPage.")

    def _draw_base(self):
        """
        Draw the base image for the value function content page.
        :return:  The base image for the value function content page.
        """
        size = self._embed.size
        logging.info(f"Regenerating base image for Value Function.")
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bg_color
        value_colors = {state: self._color_key.map_color_uint8(self._values[state]) for state in self._values}
        img = self._embed.box_placer.draw(images=None, colors=value_colors, show_bars=False, dest=img)
        return img
