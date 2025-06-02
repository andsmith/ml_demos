import cv2
import numpy as np
import logging
import tkinter as tk
from colors import COLOR_SCHEME
from abc import ABC, abstractmethod
from layout import WIN_SIZE
from util import tk_color_from_rgb


class Panel(ABC):
    """
    Abstract base class for all panels in the application.
    Every panel has a frame.
    """

    def __init__(self, app, bbox_rel,color_scheme={}, margin_rel=0.0):
        """

        """
        colors = COLOR_SCHEME.copy()
        colors.update(color_scheme)
        self.app = app
        self._bbox_rel = bbox_rel
        self._color_bg = tk_color_from_rgb( colors['bg'])
        self._color_text = tk_color_from_rgb(colors['text'])
        self._color_lines = tk_color_from_rgb(colors['lines'])
        self._frame = tk.Frame(master=self.app.root, bg=self._color_bg)

        y_margin = margin_rel / (WIN_SIZE[1] / WIN_SIZE[0])

        self._frame.place(relx=self._bbox_rel['x_rel'][0]+margin_rel, rely=self._bbox_rel['y_rel'][0]+y_margin,
                          relwidth=self._bbox_rel['x_rel'][1] - self._bbox_rel['x_rel'][0] - margin_rel*2,
                          relheight=self._bbox_rel['y_rel'][1] - self._bbox_rel['y_rel'][0] - y_margin*2)
        # set resize callback:
        self._frame.bind("<Configure>", self._on_resize)
        self._initialized = False
        self._init_widgets()

    def change_algorithm(self, alg):
        """
        Change the algorithm for this panel.
        (subclass should override this method to update their specific algorithm-related data)
        :param alg: The new algorithm to use.
        """
        self._alg = alg

    @abstractmethod
    def _init_widgets(self):
        """
        Initialize the panel.
        """
        pass

    @abstractmethod
    def _on_resize(self, event):
        """
        Handle the resize event.
        :param event: The resize event.
        """
        pass

    def _add_spacer(self, height=5, frame=None):
        """
        Add a spacer label to the given frame.
        :param frame:  The frame to add the spacer to.
        :param height:  Height of the spacer in pixels.
        """
        frame = self._frame if frame is None else frame
        label = tk.Label(frame, text="", bg=self._color_bg, font=('Helvetica', height))
        label.pack(side=tk.LEFT, fill=tk.X, pady=0)

    def get_size(self):
        """
        Get the size of the panel.
        :return: The size of the panel as a tuple (width, height).
        """
        return self._frame.winfo_width(), self._frame.winfo_height()


class Key(ABC):
    """
    Keys go in a row at the top right of tab content panels.
    They are used to indicate the meaning of colors, etc, under the mouse.
    """

    def __init__(self, size, x_offset=0):
        """
        Initialize the key with a color map, range, size, and optional drawing parameters.
        :param size: The size of the key in pixels (width, height).
        :param x_offset: how far LEFT of the image edge to draw the key.
        """
        self.size = size
        self._x_offset = x_offset

    def _get_draw_pos(self, img, center_width=None):
        """
        Get the position to draw the key on the given image.
        :param img: The image to draw on.
        :param center_width: The image to be drawn is narrower than self.size[0], so center it in the key.
        :return: The position to draw the key as (x, y).
        """
        y_top = 0
        x_left = img.shape[1] + self._x_offset
        if center_width is not None:
            pad = (self.size[0] - center_width) // 2
            x_left += pad
        return x_left, y_top

    @abstractmethod
    def draw(self, img, indicate_value=None):
        """
        Draw the key on the given image.
        If a value is indicated, represent it appropriately.
        """
        pass
