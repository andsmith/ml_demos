import cv2
import numpy as np
import logging
import tkinter as tk
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from abc import ABC, abstractmethod

from util import tk_color_from_rgb


class Panel(ABC):
    """
    Abstract base class for all panels in the application.
    Every panel has a frame.
    """

    def __init__(self, app, bbox_rel):
        """

        """
        self.app = app
        self._bbox_rel = bbox_rel
        self._bg_color = tk_color_from_rgb(COLOR_BG)
        self._frame = tk.Frame(master=self.app.root, bg=self._bg_color)
        self._frame.place(relx=self._bbox_rel['x_rel'][0], rely=self._bbox_rel['y_rel'][0],
                          relwidth=self._bbox_rel['x_rel'][1] - self._bbox_rel['x_rel'][0],
                          relheight=self._bbox_rel['y_rel'][1] - self._bbox_rel['y_rel'][0])
        # set resize callback:
        self._frame.bind("<Configure>", self._on_resize)
        self._initialized = False

    @abstractmethod
    def init(self):
        """
        Initialize the panel. This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def _on_resize(self, event):
        """
        Handle the resize event. This method should be overridden by subclasses.
        :param event: The resize event.
        """
        pass
