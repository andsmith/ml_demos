"""
Represent one interactive image for the tab panel.
Each DemoAlg subglass creates one or more TabContentPage objects, uses to construct the TabContentManager.
"""
import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_SCHEME, UI_COLORS
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
from layout import COLOR_SCHEME
from mouse_state_manager import MouseBoxManager

# BOX_SIZES = [22, 12, 12, 12, 12, 12]  # good for single value function
SPACE_SIZES = [7, 2, 2, 2, 2, 3]


######
#  NOTE on special tabs:
#    - 'states' tab shows state icons (not values) in their embedding, does not have a color key
#    - 'results' tab shows a results_viz image, not a state embedding.
######

class TabContentPage(ABC):
    
    """

    Base class for interactived images that go into the tab panel.
    
    Tracks mouse interaction for arbitrary bounding boxes:
        -mouseover in green
        -selected in red

    This class also allows mouse interaction, selecting/unselecting/mouseovering states for
    breakpoints in the algorithm loop.
    """

    def __init__(self, app, env, key_info):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param mouse_manager:  A MouseStateManager (if interactive)
        :param key_info:  dict: 
            - 'sizes': {'key_name': {'width': int, 'height': int}, ...}
            - 'x_offsets' : {'key_name': int, ...]  plot key_name at x coordinate (img_width - x_offset[key_name])
            - 'size': (width, height) of all keys)
        """
        self._app = app
        self._size = None
        self._env = env
        self._cur_box = None 
        self._mouse_manager = MouseBoxManager(app)
        self._key_info = key_info

        
        # Base images for states/values, updated as algorithm runs.
        self._base_image = None
        self._marked_image = None
        self._disp_image = None

        self._displayed_states = []  # states currently being rendered, draw boxes if app says they're selected, etc.

    def set_boxes(self, boxes):
        """
        Set the bounding boxes for the states in the image.
        :param boxes:  dict of {state: {'x': [x0, x1], 'y': [y0, y1]}}
        """
        self._mouse_manager.set_boxes(boxes)
        for box in boxes.values():
            if box is not None:
                self._displayed_states.append(box['id'])


    def clear_images(self,  marked_only=False):
        if not marked_only:
            self._base_image = None
        self._marked_image = None
        self._disp_image = None


    def set_size(self, new_size):
        self._size = new_size
        self.clear_images()

    def mouse_move(self, x, y):
        changed = self._mouse_manager.mouse_move((x,y))
        if changed:
            self.clear_images(marked_only=True)

    def mouse_click(self, x, y):
        """
        For state embedding images, just tell the tab panel which
        """
        toggled = self._mouse_manager.mouse_click((x, y))
        if toggled is not None:
            self._app.toggle_selected_state(toggled)
            self.clear_images(marked_only=True)

    def get_tab_frame(self, size,annotated=True):
        if annotated:
            if self._check_invalid(self._disp_image, size):
                self._disp_image= self._draw_annotated()
            return self._disp_image
        else:
            if self._check_invalid(self._marked_image, size):
                self._marked_image = self._draw_marked()
            return self._marked_image 
        
    def _check_invalid(self, img, size):
        return img is None or (
            img.shape[0] != size[1] or img.shape[1] != size[0])

    @abstractmethod
    def _draw_base(self):
        """
        Draw the base image for a tab.
        :param tab:  The tab to draw.
        :return:  The base image for the tab.
        """
        pass

    def _draw_marked(self):
        """
        Mark selected states (from the app) and mouseovered state on the base image using the mouse manager.
        """
        size = self._embed.size

        # Mark on the base image, regenerate if needed.
        if self._check_invalid(self._base_image, size):
            self._base_image = self._draw_base()
            if self._base_image is None:
                return None
        
        img = self._base_image.copy()

        if self._mouse_manager.mouseover_id is not None:



        # Indicate mouseovered state with a box.
        if self.mouseovered is not None:
            self.box_placer.draw_box(image=img,
                                     state_id=self.mouseovered,
                                     color=NEON_GREEN,
                                     thickness=1)
            if tab in self._color_keys:
                indicated_value = self._values[tab].get(self.mouseovered, None)

        # Draw the color key if applicable.
        if tab in self._color_keys:
            self._color_keys[tab].draw(img, line_color=COLOR_LINES, text_color=COLOR_TEXT,
                                       indicate_value=indicated_value)

        # Draw the state key if not the state tab
        if self.mouseovered is not None:
            self._state_key.draw(img, self.mouseovered, pos=self._state_key_pos)

        # Draw the state the algorithm is currently on, if set.
        if self._cur_state is not None:
            self.box_placer.draw_box(image=img,
                                     state_id=self._cur_state,
                                     color=NEON_GREEN,
                                     thickness=2)
        return img

    def _draw_annotated(self):
        """
        Annotations are things to draw while the algorithm is paused.
        """
        annotated_img = self._draw_marked() 
        # Subclasses should add their own annotations here.
        return annotated_img
        