import logging
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from colors import  COLOR_SCHEME
from gui_base import  tk_color_from_rgb
import cv2
from plot_util import get_frame


class ResizingTester(object):
    """
    test somthing that generates frames
    """

    def __init__(self,img_factory, init_size=(640,480)):
        self._img_factory = img_factory
        self._init_tk(init_size)

    def _init_tk(self, init_size):

        self._root = tk.Tk()
        self._root.geometry(f"{init_size[0]}x{init_size[1]}")
        self._root.title("Resizable tester")

        self._frame = tk.Frame(self._root, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._frame.pack(fill=tk.BOTH, expand=True)
        self._label = tk.Label(self._frame, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._label.pack(fill=tk.BOTH, expand=True)
        self._label.bind("<Configure>", self._on_resize)

    def start(self):
        self._root.mainloop()

    def _on_resize(self, event):
        self._img_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_image()

    def _get_img(self):
        img = self._img_factory(self._img_size)
        return img

    def refresh_image(self):
        print(f"Refreshing frame to size {self._img_size}")
        img = self._get_img()
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self._label.config(image=img)
        self._label.image = img


def test_resizer():
    init_size = (800, 600)
    def img_factory(size):
        # Create a simple image with a gradient
        #img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        x_p, y_p= np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        img = np.stack((x_p % 256, y_p % 256, (x_p + y_p) % 256), axis=-1).astype(np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 3
        cv2.circle(img, center, radius, (255, 0, 0), 10,cv2.LINE_AA )  # Draw a red circle
        return img
    resizer = ResizingTester(get_frame)
    resizer.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_resizer()
