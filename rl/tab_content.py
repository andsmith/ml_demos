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
SPACE_SIZES = LAYOUT['state_embedding']['space_sizes']  # sizes of the spaces between states in pixels


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

    def __init__(self, alg, keys):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param mouse_manager:  A MouseStateManager (if interactive)
        :param keys:  dict: 
            - 'sizes': {'key_name': {'width': int, 'height': int}, ...}
            - 'x_offsets' : {'key_name': int, ...]  plot key_name at x coordinate (img_width - x_offset[key_name])
            - 'size': (width, height) of all keys)
        """
        self._alg = alg
        self._app = alg.app
        self._size = None
        self._env = alg.app.env
        self._cur_box = None
        self._mouse_manager = MouseBoxManager(self._app)

        self._keys = keys

        # Base images for states/values, updated as algorithm runs.
        self._base_image = None
        self._marked_image = None
        self._disp_image = None
        self._displayed_states = {}  # states currently being rendered, draw boxes if app says they're selected, etc.
        self._blank_frame = self._make_blank()

    def resize(self, new_size):
        if self._size is None or (new_size is not None and self._size != new_size):
            self._size = new_size
            self.clear_images()

    def _make_blank(self):
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']
        cv2.putText(img, "No data to display", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        return img

    def clear_images(self,  marked_only=False):
        if not marked_only:
            self._base_image = None
        self._marked_image = None
        self._disp_image = None

    def set_size(self, new_size):
        self._size = new_size
        self.clear_images()

    def mouse_move(self, pos_xy):
        changed = self._mouse_manager.mouse_move(pos_xy)
        if changed:
            self.clear_images(marked_only=True)
            return True
        return False

    def mouse_leave(self):
        if self._mouse_manager.mouse_leave():
            self.clear_images(marked_only=True)
            return True
        return False

    def mouse_click(self, pos_xy):
        """
        For state embedding images, just tell the tab panel which
        """
        toggled, _ = self._mouse_manager.mouse_click(pos_xy)
        if toggled is not None:
            self._app.toggle_selected_state(toggled)
            self.clear_images(marked_only=True)
            return toggled
        return None

    def get_tab_frame(self, size, annotated=True):

        if size is None:
            return self._blank_frame

        if annotated:
            if self._check_invalid(self._disp_image, size):
                self._disp_image = self._draw_annotated()
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

    @abstractmethod
    def _get_key_value(self, key_name):
        """
        See what the mouse is over, the state of the algorithm, etc, to determine the value the
        keys should be indicating.
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

        # Draw the boxes for mouseovered & selected states.
        selected = [s for s in self._app.selected if s in self._displayed_states]
        self._mouse_manager.render_state(img, selected_ids=selected, thickness=1)

        # Draw the state the algorithm is currently on, if set (& displayed)
        if self._alg.state is not None and self._alg.state in self._displayed_states:
            self._mouse_manager.mark_box(img, self._alg.state, color=UI_COLORS['current_state'], thickness=2)

        # Draw all keys.
        for key_name, key in self._keys.items():
            key_value = self._get_key_value(key_name)
            key.draw(img, indicate_value=key_value)

        # draw a box around the key area
        tks = self._alg._total_key_size
        img[tks[1], img.shape[1]-tks[0]:] = COLOR_SCHEME['lines']
        img[:tks[1], img.shape[1]-tks[0]] = COLOR_SCHEME['lines']

        return img

    def _draw_annotated(self):
        """
        Annotations are things to draw while the algorithm is paused.
        """
        annotated_img = self._draw_marked()
        # Subclasses should add their own annotations here.
        return annotated_img
