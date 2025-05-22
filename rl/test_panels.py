"""
Algorithm selection, state load/save/reset, fullscreen, start game.
Fake algorithm (bouncing circles/squares)
Fake app (uses real pannels)
"""

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

from loop_timing.loop_profiler import LoopPerfTimer as LPT


class TestDemoAlg(PolicyEvalDemoAlg):
    """
    Run this by adding it to rl_demo.py, in the top-level list 'ALGORITHMS'.
    """

    def __init__(self, *args, **kwargs):
        self._state = None
        self._bkg_color_rgb = COLOR_BG
        self._text_color_rgb = COLOR_TEXT
        self._line_color_rgb = COLOR_LINES
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
        self._circle_v = np.random.randn(n_circles* 2).reshape(n_circles, 2) * .0025

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['circle-update'] = "Circle update"
        rco['frame-update'] = "Frame update"
        return rco

    def get_state_tab_info(self):
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
    @LPT.time_function
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
            update_txt= "State updated:  %i" % self.next_state_ind
            cv2.putText(img, update_txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_GREEN, 1, cv2.LINE_AA)
        return img
    
    @LPT.time_function
    def get_state_image(self, size, tab_name, is_paused):
        """
        """
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bkg_color_rgb
        filled = (tab_name == 'filled')
        connect = (is_paused)
        img = self._draw_image_data(img, connect=connect, fill=filled)
        return img

    def start(self):

        def run_thread():
            """
            If running, move the circles XYZ randomly every frame (FPS times per second).
            """
            print("---------------> STARTING TEST DEMO ALGORITHM")
            LPT.reset(enable=False, burn_in=4, display_after=5)

            while True:
                self.pe_iter +=1
                #if n_frames % 30 == 0:
                #    print("TestDemoAlg: frame %i" % n_frames)
                LPT.mark_loop_start()
                time.sleep(.005)
                for ind in range(self._image_data.shape[0]):       

                    self.next_state_ind = ind

                    self._image_data[ind, :2] += self._circle_v[ind, :] 
                    if self._image_data[ind, 0] < 0 or self._image_data[ind, 0] > 1:
                        self._circle_v[ind, 0] *= -1
                    if self._image_data[ind, 1] < 0 or self._image_data[ind, 1] > 1:
                        self._circle_v[ind, 1] *= -1


                    self._maybe_pause('circle-update')  # check if we should pause here.

                
                self._maybe_pause('frame-update')  # check if we should pause here.

        self._loop_thread = Thread(target=run_thread, daemon=True)
        self._loop_thread.start()
        logging.info("TestDemoAlg: started loop thread.")

    @staticmethod
    def get_name():
        return 'test_alg'

    @staticmethod
    def get_str():
        return "Test Algorithm (Circles)"


'''
class TestApp(object):
    """
    Stand-in for demo app to test the SelectionPanel.
    """

    def __init__(self):
        from layout import WIN_SIZE
        from selection_panel import SelectionPanel
        from game_base import Mark
        from reinforcement_base import Environment
        from baseline_players import HeuristicPlayer

        self._bkg_color_rgb = COLOR_BG
        self._text_color_rgb = COLOR_TEXT
        self._line_color_rgb = COLOR_LINES
        self.win_size = WIN_SIZE
        self._fullscreen = False
        self._running = False

        self.root = tk.Tk()
        self.root.geometry(f"{self.win_size[0]}x{self.win_size[1]}")
        self.root.title("Selection Panel Test")
        self._env = Environment(opponent_policy=HeuristicPlayer(mark=Mark.O, n_rules=2))
        self._alg = TestDemoAlg(self, self._env)
        self.status_control_panel = StatusControlPanel(
            self, self._alg, bbox_rel=LAYOUT['frames']['control'])
        self.selection_panel = SelectionPanel(self, [TestDemoAlg], bbox_rel=LAYOUT['frames']['selection'])

        # test state tabs:
        self._state_img_size = None
        state_tab_info = self._alg.get_state_tab_info()
        self.state_panel = StatePanel(self, bbox_rel=LAYOUT['frames']['state-tabs'])

    def refresh_status(self):
        self.status_control_panel.refresh_status()

        """
        Get the state image from the algorithm
        """
        print("^^^^^^^^^^^^^ App drawing image on blank with size %s" % str(frame_size))
        return self._alg.get_state_img(frame_size, which=which, running=self._running)

    def run(self):
        # Start the run thread:
        self._alg.start()
        self.state_panel.change_algorithm(self._alg)

        self.root.mainloop()

    def save_state(self):
        logging.info("Save state button pressed.")

    def load_state(self):
        logging.info("Load state button pressed.")

    def reset_state(self):
        logging.info("Reset state button pressed.")

    def toggle_fullscreen(self):
        logging.info("Toggle fullscreen button pressed.")

    def toggle_fullscreen(self):

        self._fullscreen = not self._fullscreen
        global geom
        if self._fullscreen:
            geom = self.root.geometry()
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.overrideredirect(True)
            self.root.geometry('%dx%d+0+0' % (w, h))

        else:
            self.root.overrideredirect(False)
            self.root.geometry(geom)

    def set_opponent(self, n_rules):
        logging.info(f"Setting opponent to {n_rules} rules.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = TestApp()
    app.run()

'''
