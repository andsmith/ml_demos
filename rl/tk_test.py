"""
Test framerate for displaying images in a Tkinter window.
"""
import tkinter as tk
import logging
import numpy as np
import time
import cv2

from PIL import Image, ImageTk
WIN_SIZE = 1200, 800
WIN_NAME = 'Tkinter Animation Test'
ANIM_SIZE = 1180, 700
ANIM_POS = 10, 10

SHIFT_B = 5
SHIFT = 1 << SHIFT_B


class AnimObject(object):
    """
    Thing with position size and color.
    """

    def __init__(self, win_size, pos=None, size=None, color=None):
        """
        :param win_size:  Size of the window.
        :param pos:  Position of the object.
        :param size:  Size of the object.
        :param color:  Color of the object.
        """
        self._win_size = win_size
        self._pos = pos if pos is not None else np.random.randint(0, win_size[0]), np.random.randint(0, win_size[1])
        self._size = size if size is not None else np.random.randint(10, 50), np.random.randint(10, 50)
        self._color = color if color is not None else (100 + np.random.randint(0, 155, 3)).tolist()
        self._vel = np.random.randn(2) * 150

    def _move_wrap(self, pos, inc):
        new_pos = ((pos[0] + inc[0]) % self._win_size[0],
                   ( pos[1] + inc[1] )% self._win_size[1])
        return new_pos 

    def tick(self, dt, rand_fact=0.):
        """
        Update the position of the object.
        :param dt:  Time delta.
        """
        self._pos = self._move_wrap(self._pos, self._vel * dt)
        if rand_fact>0:
            self._pos = self._move_wrap(self._pos, np.random.randn(2) * rand_fact * dt)

        
    def draw(self, image):
        # for now a circle, later a rectangle or something else.
        cv2.circle(image, (int(self._pos[0]*SHIFT), int(self._pos[1]*SHIFT)),
                   int(SHIFT*self._size[0]), self._color, 3, cv2.LINE_AA, shift=SHIFT_B)


class TKAnimation(object):
    """
    Window has one label (the image) and two buttons (increase/decrease complexity)
    """

    def __init__(self, win_size=WIN_SIZE, complexity=1.):
        self._dt = 1.0 / 30.0  # 30 fps
        self._c = complexity
        self._size = win_size
        self._anim_size = ANIM_SIZE
        self._last_time = time.perf_counter()
        self._fps_info = {'t_start': time.perf_counter(),
                           't_last': time.perf_counter(),
                           'fps': 0.0,
                           'frame_count': 0,
                           'update_n': 10}

        self._init_gui()
        self._init_geom()

    def _init_gui(self):
        self._root = tk.Tk()
        self._root.title(WIN_NAME)
        self._root.geometry(f"{self._size[0]}x{self._size[1]}")
        self._root.resizable(False, False)

        # Create a label to display the image
        self._label = tk.Label(self._root)
        self._label.pack()

        # register mouse motion callback so the gui doesn't freeze.
        self._root.bind("<Motion>", lambda e: None)
        self._root.bind("<Button-1>", lambda e: None)
        self._root.bind("<Button-2>", lambda e: None)

        # Create buttons to increase/decrease complexity
        self._increase_button = tk.Button(self._root, text="Increase Complexity",
                                          command=lambda: self.change_complexity(1))
        self._increase_button.pack(side=tk.LEFT, padx=5, pady=5)

        self._decrease_button = tk.Button(self._root, text="Decrease Complexity",
                                          command=lambda: self.change_complexity(-1))
        self._decrease_button.pack(side=tk.LEFT, padx=ANIM_POS[0], pady=ANIM_POS[1])

        # Create a slider for fps and a slider for randomness temperature
        self._fps_slider = tk.Scale(self._root, from_=1, to=50, orient=tk.HORIZONTAL, label="Delay MS")
        self._fps_slider.set(16)  # Set default value to 30 fps
        self._fps_slider.pack(side=tk.LEFT, padx=5, pady=5)

        self._rand_slider = tk.Scale(self._root, from_=0, to=100, orient=tk.HORIZONTAL, label="Randomness")
        self._rand_slider.set(10)  # Set default value to 10%
        self._rand_slider.pack(side=tk.LEFT, padx=5, pady=5)

    def change_complexity(self, direction):
        factor = 1.1 ** direction
        self._c *= factor
        logging.info(f"Complexity: {self._c:.2f}")
        n_objs = int(self._c)
        while n_objs > len(self._objects):
            logging.info(f"Adding object {len(self._objects)}")
            self._objects.append(AnimObject(self._size))
        while n_objs < len(self._objects):
            logging.info(f"Removing object {len(self._objects)-1}")
            self._objects.pop()

    def _init_geom(self):
        self._objects = [AnimObject(self._anim_size) for _ in range(int(self._c))]
        self._blank_frame = np.zeros((self._anim_size[1], self._anim_size[0], 3), dtype=np.uint8)

    def _update(self):
        # Update the objects
        dt = time.perf_counter() - self._last_time
        self._last_time = time.perf_counter()
        for obj in self._objects:
            obj.tick(dt, self._rand_slider.get() *5.)

        # Draw the objects
        image = self._blank_frame.copy()
        for obj in self._objects:
            obj.draw(image)

        # write the current FPS in the top left corner
        cv2.putText(image, f"FPS: {self._fps_info['fps']:.2f}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1., (255, 255, 255), 1)

        # Convert to PIL Image and update label
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self._label.config(image=tk_image)
        self._label.image = tk_image

        # Schedule the next update
        self._root.after(int(self._fps_slider.get()), self._update)
        # Update the window title with the current complexity
        self._root.title(f"{WIN_NAME} - Complexity: {self._c:.2f}")

        # update fps info
        self._fps_info['frame_count'] += 1
        if self._fps_info['frame_count'] % self._fps_info['update_n'] == 0:
            t_now = time.perf_counter()
            self._fps_info['fps'] = self._fps_info['update_n'] / (t_now - self._fps_info['t_last'])
            self._fps_info['t_last'] = t_now
            logging.info(f"FPS: {self._fps_info['fps']:.2f}")
            


    def start(self):
        self._root.after(0, self._update)
        self._root.mainloop()
        logging.info("Animation finished.")


def run():
    ta = TKAnimation()
    ta.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run()
