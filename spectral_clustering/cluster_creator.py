"""
App for creating datasets by drawing clusters with the mouse,
and exploring the clustering results.

window layout:
    <----2/3rds----> <--1/3rd-->
    +---------------+----------+ ^
    |   CLUSTER     | widgets  | 1/3rd
    |   DRAWING     +----------+ v
    |     UI        | spectrum |
    +--------+------+----------+
    | E-proj |      | clusters |
    +--------+------+----------+

The cluster drawing UI is a window with a canvas where the user can draw and adjust clusters.
Widgets includes buttons for interacting:

    1.  Slider for n_points
    2.  Choose which cluster to draw (or delete clusters selected)
    3.  Clear
    4.  Run clustering.

"""
import numpy as np
from windows import ClustersWindow, ToolsWindow, SpectrumWindow, EigenvectorsWindow, UiWindow, COLORS
import cv2
import time
# all dims in unit square, to be scaled to window size


class ClusterCreator(object);

    def __init__(self, size=(1200, 950)):
        self._size = size
        self._bkg = np.zeros((size[1], size[0], 3), np.uint8) + COLORS['BLACK']
        self._windows = [UiWindow('ui', size),
                         ToolsWindow('tools', size),
                         SpectrumWindow('spectrum', size),
                         ClustersWindow('clusters', size),
                         EigenvectorsWindow('eigenvectors', size)]
        self._active_window = None
        self._mouse_pos = None
        self._clicked_pos = None
        self._clicked_window = None
        self._fps_info = {'last_time': time.perf_counter(), 
                          'n_frames': 0,
                          'update_sec': 2.0}

    def run(self):
        """
        Run the cluster creator app.
        """
        cv2.namedWindow('Cluster Creator')
        cv2.setMouseCallback('Cluster Creator', self._mouse_callback)


        while True:
            # refresh
            frame = self._refresh()
            cv2.imshow('Cluster Creator', frame)

            # display & send key event
            k = cv2.waitKey(1) & 0xFF
            if k & 0xFF == 27:
                break
            elif self._active_window is not None:
                self._active_window.handle_key(k)

            # update fps
            self._fps_info['n_frames'] += 1
            elapsed = time.perf_counter() - self._fps_info['last_time']
            if elapsed > self._fps_info['update_sec']:
                fps = self._fps_info['n_frames'] / elapsed
                print(f'fps: {fps:.2f}')
                self._fps_info['last_time'] = time.perf_counter()
                self._fps_info['n_frames'] = 0  

                