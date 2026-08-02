"""
Small shared GUI helpers: colour conversion, an image-blit helper, and the
abstract ``Panel`` base class every framed panel derives from (mirrors
../rl/gui_base.py, but self-contained for this app).
"""

from abc import ABC, abstractmethod
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from layout import COLOR_SCHEME, WIN_SIZE


def tk_color_from_rgb(rgb):
    """Convert an (r, g, b) tuple in 0..255 to a Tk '#rrggbb' string."""
    r, g, b = (int(c) for c in rgb)
    return f'#{r:02x}{g:02x}{b:02x}'


def blit(label, arr):
    """
    Push a uint8 RGB numpy array into a tk.Label as an image, keeping a
    reference on the label so the PhotoImage isn't garbage-collected.

    :param label: the tk.Label to update
    :param arr: (H, W, 3) uint8 RGB array
    """
    img = ImageTk.PhotoImage(image=Image.fromarray(arr))
    label.config(image=img)
    label.image = img


class Panel(ABC):
    """
    Abstract base for a placed, framed panel.  Builds ``self._frame`` at the
    given relative bounding box, wires a resize handler, then calls the
    subclass ``_init_widgets``.
    """

    def __init__(self, app, bbox_rel, margin_rel=0.0):
        """
        :param app: the owning application (must expose ``.root``)
        :param bbox_rel: dict with 'x_rel' and 'y_rel' (start, end) fractions
        :param margin_rel: inner margin as a fraction of window width
        """
        self.app = app
        self._bbox_rel = bbox_rel
        self._color_bg = tk_color_from_rgb(COLOR_SCHEME['panel_bg'])
        self._color_text = tk_color_from_rgb(COLOR_SCHEME['text'])
        self._frame = tk.Frame(master=app.root, bg=self._color_bg)

        y_margin = margin_rel / (WIN_SIZE[1] / WIN_SIZE[0])
        self._frame.place(
            relx=bbox_rel['x_rel'][0] + margin_rel,
            rely=bbox_rel['y_rel'][0] + y_margin,
            relwidth=bbox_rel['x_rel'][1] - bbox_rel['x_rel'][0] - margin_rel * 2,
            relheight=bbox_rel['y_rel'][1] - bbox_rel['y_rel'][0] - y_margin * 2)
        self._frame.bind("<Configure>", self._on_resize)
        self._init_widgets()

    @abstractmethod
    def _init_widgets(self):
        """Create the panel's widgets."""
        pass

    def _on_resize(self, event):
        """Handle a resize of the panel's frame (override as needed)."""
        pass

    def _add_spacer(self, height=5, frame=None):
        """Add a small vertical spacer to a frame (packed top-down)."""
        frame = self._frame if frame is None else frame
        tk.Label(frame, text="", bg=self._color_bg,
                 font=('Helvetica', height)).pack(side=tk.TOP, fill=tk.X)

    def get_size(self):
        """Return the frame's current (width, height) in pixels."""
        return self._frame.winfo_width(), self._frame.winfo_height()
