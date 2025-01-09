"""
App for creating datasets by drawing clusters with the mouse,
and exploring the clustering results.

window layout:
    <----2/3rds----> <--1/3rd-->
    +---------------+----------+ ^
    |   CLUSTER     | spectrum | 1/3rd
    |   DRAWING     +----------+ v
    |     UI        | ev-proj  |
    +--------+------+----------+
    |  toolbar      | clusters |
    +--------+------+----------+

The cluster drawing UI is a window with a canvas where the user can draw and adjust clusters.
Widgets includes buttons for interacting:

    1.  Slider for n_points
    2.  Choose which cluster to draw (or delete clusters selected)
    3.  Clear
    4.  Run clustering.

"""
import numpy as np
from windows import ClustersWindow, ToolsWindow, SpectrumWindow, EigenvectorsWindow, UiWindow
import cv2
import logging
import time
# all dims in unit square, to be scaled to window size
from layout import LAYOUT, TOOLBAR_LAYOUT


class ClusterCreator(object):

    def __init__(self, size=(640, 480)):
        logging.info('Initializing Cluster Creator')
        self._size = size
        self._bkg = np.zeros((size[1], size[0], 3), np.uint8) + LAYOUT['colors']['bkg']
        self._windows = [UiWindow('ui', size,self),
                         ToolsWindow('tools', size,self),
                         SpectrumWindow('spectrum', size,self),
                         ClustersWindow('clusters', size,self),
                         EigenvectorsWindow('eigenvectors', size,self)
                         ]
        self._active_window = None  # Mouse is over this window
        self._clicked_window = None  # Mouse was clicked in this window, but may not be over it
        self._mouse_pos = None
        self._clicked_pos = None
        self._clicked_window = None
        self._fps_info = {'last_time': time.perf_counter(),
                          'n_frames': 0,
                          'update_sec': 2.0}
        
    def recompute(self):
        """
        Recompute the clustering.
        """
        print('Recomputing')
        

    def clear(self):
        """
        Clear the clusters.
        self._windows['ui'].clear()
        """
        print('Clearing')

    def run(self):
        """
        Run the cluster creator app.
        """
        cv2.namedWindow('Cluster Creator')
        cv2.setMouseCallback('Cluster Creator', self._mouse_callback)

        logging.info('Running Cluster Creator')
        while True:
            # refresh
            frame = self._refresh()

            cv2.imshow('Cluster Creator', frame)

            # display & send key event
            k = cv2.waitKey(1) & 0xFF
            if k & 0xFF == 27 or k == ord('q'):
                break
            elif self._active_window is not None:
                self._active_window.keypress(k)

            # update fps
            self._fps_info['n_frames'] += 1
            elapsed = time.perf_counter() - self._fps_info['last_time']
            if elapsed > self._fps_info['update_sec']:
                fps = self._fps_info['n_frames'] / elapsed
                print(f'fps: {fps:.2f}')
                self._fps_info['last_time'] = time.perf_counter()
                self._fps_info['n_frames'] = 0

        cv2.destroyAllWindows()

    def _refresh(self):
        """
        Refresh the windows and return the main frame.
        """
        frame = self._bkg.copy()
        for window in self._windows:
            window.render(frame, active = (window == self._active_window))
        return frame

    def _get_window(self, x, y):
        """
        Return the window that contains the point (x, y).
        """
        for window in self._windows:
            if window.contains(x, y):
                return window
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events:
            if we're in the middle of a click, pass the event to the active window,
            otherwise update the active window and pass the event to it.
        """
        self._mouse_pos = (x, y)
        if self._clicked_window is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                self._clicked_window.mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._clicked_window.mouse_unclick(x, y)
                self._clicked_window = None
            else:
                raise Exception("Can't click in a window while another window is clicked!")
            return
        self._active_window = self._get_window(x, y)
        if self._active_window is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._active_window.mouse_click(x, y):
                    self._clicked_window = self._active_window
                    self._clicked_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                self._active_window.mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                #raise Exception("Can't unclick in a window that wasn't clicked!")
                pass # will happen if double-clicking, so just do nothing.
        else:
            self._active_window = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cc = ClusterCreator()
    cc.run()
