"""
Algorithm selection, state load/save/reset, fullscreen, start game.
Fake algorithm (bouncing circles/squares)
Fake app (uses real pannels)
"""
'''
from policy_eval import PolicyEvalDemoAlg
from collections import OrderedDict
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import numpy as np
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT, NEON_BLUE, NEON_GREEN, NEON_RED
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE
import logging
import time
from threading import Thread, Event, get_ident
import cv2
from status_ctrl_panel import StatusControlPanel
from alg_panels import StatePanel
from collections import OrderedDict
import layout
TITLE_INDENT = 0
ITEM_INDENT = 20

# from loop_timing.loop_profiler import LoopPerfTimer as LPT


class TestDemoAlg(PolicyEvalDemoAlg):
    """
    Run this by adding it to rl_demo.py, in the top-level list 'ALGORITHMS'.
    """

    def __init__(self, *args, **kwargs):
        self._state = None
        self._colors = COLOR_SCHEME
        self._tab_img_cache = {'paused': {}, 'running': {}}
        self._tab_img_size = None
        self.reset_state()

        super().__init__(*args, **kwargs)

    def reset_state(self):
        print("TEST_ALG RESET_STATE")
        self.pe_iter = 0
        self.next_state_ind = 0
        n_circles = 50
        self._image_data = np.random.rand(4 * n_circles).reshape(n_circles, 4)  # plot these (x,y, rad,thickness)
        self._circle_v = np.random.randn(n_circles * 2).reshape(n_circles, 2) * .0025

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['circle-update'] = "Circle update"
        rco['frame-update'] = "Frame update"
        return rco

    def get_tab_info(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        st = OrderedDict((('hollow', 'Hollow circles  '),
                          ('filled', 'Filled circles  ')))
        return st

    def get_status(self):
        font_default = layout.LAYOUT['fonts']['status']
        font_bold = layout.LAYOUT['fonts']['status_bold']

        status = [('Algorithm:  %s' % self.get_name(), font_bold),
                  ('Algorithm iter:  %i' % self.pe_iter, font_default),
                  ('Next state index:  %i' % self.next_state_ind, font_default),
                  ('Paused:  %s' % str(self.paused), font_default)]
        return status

    def _draw_image_data(self, img, connect=False, fill=False):
        PREC_BITS = 5
        PREC_MULT = 2**PREC_BITS
        w, h = img.shape[1], img.shape[0]
        circle_coords = self._image_data[:, :2] * np.array([w, h]) * PREC_MULT
        circle_coords = circle_coords.astype(int)
        circle_radii = (self._image_data[:, 2] * w/20 * PREC_MULT).astype(int)
        thicknesses = (self._image_data[:, 3] * 20).astype(int) if not fill else -np.ones_like(circle_radii, dtype=int)
        for i, ((x, y), r) in enumerate(zip(circle_coords, circle_radii)):
            color = self._line_color_rgb
            cv2.circle(img, (x, y), r, color, thickness=thicknesses[i], lineType=cv2.LINE_AA, shift=PREC_BITS)
        if connect:
            # draw lines from every circle center to every other circle center:
            for i in range(len(circle_coords)):
                for j in range(i + 1, len(circle_coords)):
                    color = self._line_color_rgb
                    cv2.line(img, tuple(circle_coords[i]), tuple(circle_coords[j]),
                             color, thickness=1, lineType=cv2.LINE_AA, shift=PREC_BITS)
        # draw the circles:
        return img
    # @LPT.time_function

    def get_viz_image(self, size, control_point, is_paused):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bkg_color_rgb
        special = (control_point == 'frame-update')
        # Draw 10 random circles:
        for _ in range(10):
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])
            r = np.random.randint(5, 20)
            color = self._line_color_rgb if not special else NEON_BLUE
            cv2.circle(img, (x, y), r, color, thickness=-1 if (not is_paused) else 3, lineType=cv2.LINE_AA)
            update_txt = "State updated:  %i" % self.next_state_ind
            cv2.putText(img, update_txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_GREEN, 1, cv2.LINE_AA)
        return img

    # @LPT.time_function
    def get_state_image(self, size, tab_name, is_paused):
        """
        """
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bkg_color_rgb
        filled = (tab_name == 'filled')
        connect = (is_paused)
        img = self._draw_image_data(img, connect=connect, fill=filled)
        return img

    def _learn_loop(self):
        """
        If running, move the circles XYZ randomly every frame (FPS times per second).
        """
        print("---------------> STARTING TEST DEMO ALGORITHM")
        # LPT.reset(enable=False, burn_in=4, display_after=5)

        while not self._shutdown:
            self.pe_iter += 1
            time.sleep(.005)
            for ind in range(self._image_data.shape[0]):
                self.next_state_ind = ind
                self._image_data[ind, :2] += self._circle_v[ind, :]
                if self._image_data[ind, 0] < 0 or self._image_data[ind, 0] > 1:
                    self._circle_v[ind, 0] *= -1
                if self._image_data[ind, 1] < 0 or self._image_data[ind, 1] > 1:
                    self._circle_v[ind, 1] *= -1

                if self._maybe_pause('circle-update'): 
                    return
            if self._maybe_pause('frame-update'): 
                return
            
    @staticmethod
    def get_name():
        return 'test_alg'

    @staticmethod
    def get_str():
        return "Test Algorithm (Circles)"

'''