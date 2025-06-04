import cv2
import numpy as np
import logging
import tkinter as tk
from colors import COLOR_SCHEME
from abc import ABC, abstractmethod
from layout import WIN_SIZE
from util import tk_color_from_rgb, get_font_scale


import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk


class Panel(ABC):
    """
    Abstract base class for all panels in the application.
    Every panel has a frame.
    """

    def __init__(self, app, bbox_rel, color_scheme={}, margin_rel=0.0):
        """

        """
        colors = COLOR_SCHEME.copy()
        colors.update(color_scheme)
        self.app = app
        self._bbox_rel = bbox_rel
        self._color_bg = tk_color_from_rgb(colors['bg'])
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

    def __init__(self, size, x_offset=None):
        """
        Initialize the key with a color map, range, size, and optional drawing parameters.
        :param size: The size of the key in pixels (width, height).
        :param x_offset: how far LEFT of the image edge to draw the key.
        """
        self.size = size
        self._x_offset = x_offset if x_offset is not None else -size[0]

    def _get_draw_pos(self, img, center_width=None):
        """
        Get the position to draw the key on the given image.
        :param img: The image to draw on.
        :param center_width: The key to be drawn is narrower than self.size[0], so center it in the key bbox.
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

    def draw_bbox(self, img, color, thickness=1):
        """
        Draw a bounding box around the key on the given image.
        :param img: The image to draw on.
        :param color: The color of the bounding box.
        :param thickness: The thickness of the bounding box lines.
        """
        x0, y0 = self._get_draw_pos(img)
        cv2.rectangle(img, (x0, y0), (x0 + self.size[0]-1, y0 + self.size[1]-1), color, thickness)


class TextKey(Key):
    def __init__(self, size, lines, fonts=None, indent_xy=(15, 15), spacing=1.1, rel_sizes=None):
        """
        Initialize a text key with the given size, lines, fonts, indent, spacing, and relative sizes.
        :param size: The size of the key in pixels (width, height).
        :param lines: A list of strings to display in the key.
        :param fonts: A list of font sizes for each line.
        :param indent_xy: Indentation for the text from (left, top)
        :param spacing: lines spaced by: font height * spacing 
        :param rel_sizes: Relative sizes for each line (if None, use default font sizes).
           NOTE:  All sizes==1 will have the same size (i.e. the max that will fit all those lines)
        """
        super().__init__(size)
        self._lines = lines
        self._fonts = fonts if fonts is not None else [cv2.FONT_HERSHEY_SIMPLEX] * len(lines)
        self._indent_xy = np.array(indent_xy)
        self._sp_frac = spacing
        self._rel_sizes = np.array(rel_sizes if rel_sizes is not None else [1] * len(lines))
        self.dims = self._calc_dims()

        lengths = [len(lines), len(fonts), len(self._rel_sizes)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All input lists must have the same length. "
                             f"Got: lines={len(lines)}, fonts={len(fonts)}, rel_sizes={len(self._rel_sizes)}")

    def _calc_dims(self):
        w, h = self.size
        text_w, text_h = w-self._indent_xy[0]*2, h - self._indent_xy[1]*2
        weights = np.array(self._rel_sizes)/np.sum(self._rel_sizes)
        text_heights = (text_h * weights/self._sp_frac).astype(int)
        # Calculate the font scale for each line based on the text height and font size:
        font_scales = np.array([get_font_scale(f, t_h, text_w, incl_baseline=True, text_lines=[l])
                                for t_h, f, l in zip(text_heights, self._fonts, self._lines)])
        smallest = np.where(self._rel_sizes == 1)

        if len(smallest[0]) > 1:
            smallest_scale = font_scales[smallest].min()
            font_scales[smallest] = smallest_scale

        text_sizes = [cv2.getTextSize(line, font, scale, 1)
                      for line, font, scale in zip(self._lines, self._fonts, font_scales)]
        real_text_heights = [size[0][1]+size[1] for size in text_sizes]
        h_diff = text_h - np.sum(real_text_heights)


        dims = {'text_box_heights': text_heights,
                
                'text_heights': real_text_heights,
                'font_scale': font_scales,
                'width': text_w,
                'y_top': self._indent_xy[1],
                'x_left': self._indent_xy[0],
                'text_sizes': text_sizes}
        return dims

    def draw(self, img, indicate_value=None):
        """
        Draw the text key on the given image.
        :param img: The image to draw on.
        :param indicate_value: Not used in this key type.
        """
        x0, y0 = self._get_draw_pos(img)
        x_left, y_top = self.dims['x_left'] + x0, self.dims['y_top'] + y0
        # Fill the background with the key color:
        for i, line in enumerate(self._lines):
            baseline = self.dims['text_sizes'][i][1]
            font = self._fonts[i]
            scale = self.dims['font_scale'][i]
            text_height = self.dims['text_heights'][i]
            pos = (x_left, y_top + text_height-baseline)
            cv2.putText(img, line, pos, font, scale, COLOR_SCHEME['text'], 1, cv2.LINE_AA)
            y_top += int(text_height * self._sp_frac)


class KeySizeTester(object):
    """
    Tk window with single image lable (the key), to show resize capability.
    """

    def __init__(self,size, key_factory):
        key = key_factory(size)
        self._img_size = key.size
        self._make_key = key_factory
        self._init_tk()

    def _init_tk(self):

        self._root = tk.Tk()
        self._root.geometry(f"{self._img_size[0]}x{self._img_size[1]}")
        self._root.title("State Embedding Key Tester")

        self._frame = tk.Frame(self._root, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._frame.pack(fill=tk.BOTH, expand=True)
        self._label = tk.Label(self._frame, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._label.pack(fill=tk.BOTH, expand=True)
        self._label.bind("<Configure>", self._on_resize)

    def start(self):
        """
        Start the Tk main loop.
        """
        self._root.mainloop()

    def _on_resize(self, event):
        print(f"Resize event: {event.width}x{event.height}")
        self._img_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_key_image()

    def refresh_key_image(self):
        print(f"Refreshing key image to size {self._img_size}")

        key= self._make_key(self._img_size)


        img = np.zeros((self._img_size[1], self._img_size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']
        key.draw(img)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self._label.config(image=img)
        self._label.image = img


def test_text_key():

    def _make_key(size):
        lines = ["Text Key Test", "This is the first line.", "And this is the second line."]
        lines_short = ["Ty","T1","ly2","l3","l4","l5"]
        title_font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        sub_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fonts = [title_font, sub_font, sub_font, sub_font, sub_font, sub_font]
        weights = [2, 1, 1, 1, 1, 1]

        nn = 4
        key = TextKey(size,
                    lines=lines_short[:nn],
                    fonts=fonts[:nn],
                    rel_sizes=weights[:nn],
                    spacing=1.5, indent_xy=(20, 20))
        return key
    
    size = (300, 220)

    tester = KeySizeTester(size,_make_key)
    tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_text_key()
