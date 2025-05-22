import cv2
import numpy as np
import logging
import tkinter as tk
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from abc import ABC, abstractmethod
from layout import WIN_SIZE
from util import tk_color_from_rgb


class Panel(ABC):
    """
    Abstract base class for all panels in the application.
    Every panel has a frame.
    """

    def __init__(self, app, bbox_rel, margin_rel=0.0):
        """

        """
        self.app = app
        self._bbox_rel = bbox_rel
        self._bg_color = tk_color_from_rgb(COLOR_BG)
        self._bg_color_rgb = COLOR_BG
        self._text_color = tk_color_from_rgb(COLOR_TEXT)
        self._line_color = tk_color_from_rgb(COLOR_LINES) 
        self._frame = tk.Frame(master=self.app.root, bg=self._bg_color)
        
        y_margin = margin_rel / (WIN_SIZE[1] / WIN_SIZE[0])
        
        self._frame.place(relx=self._bbox_rel['x_rel'][0]+margin_rel, rely=self._bbox_rel['y_rel'][0]+y_margin,
                          relwidth=self._bbox_rel['x_rel'][1] - self._bbox_rel['x_rel'][0] - margin_rel*2,
                          relheight=self._bbox_rel['y_rel'][1] - self._bbox_rel['y_rel'][0] - y_margin*2)
        # set resize callback:
        self._frame.bind("<Configure>", self._on_resize)
        self._initialized = False
        self._init_widgets()

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
        label = tk.Label(frame, text="", bg=self._bg_color, font=('Helvetica', height))
        label.pack(side=tk.LEFT, fill=tk.X, pady=0)

    def get_size(self):
        """
        Get the size of the panel.
        :return: The size of the panel as a tuple (width, height).
        """
        return self._frame.winfo_width(), self._frame.winfo_height()
