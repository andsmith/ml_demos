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
from clustering import KMeansAlgorithm
from util import get_n_disp_colors

class ClusterCreator(object):

    def __init__(self, size=(640, 480)):
        logging.info('Initializing Cluster Creator')
        self._size = size
        self._bkg = np.zeros((size[1], size[0], 3), np.uint8) + LAYOUT['colors']['bkg']
        self._windows = {'ui':UiWindow('ui', size,self),
                         'tools':ToolsWindow('tools', size,self),
                         'spectrum':SpectrumWindow('spectrum', size,self),
                         'clusters':ClustersWindow('clusters', size,self),
                         'eigenvectors':EigenvectorsWindow('eigenvectors', size,self)}
        self._active_window_name = None  # Mouse is over this window
        self._clicked_window_name = None  # Mouse was clicked in this window, but may not be over it
        self._mouse_pos = None
        self._cluster_colors = None  # update only when K changes
        self._clicked_pos = None
        self._fps_info = {'last_time': time.perf_counter(),
                          'n_frames': 0,
                          'update_sec': 2.0}
        
    def _do_clustering(self, points, algorithm_name, n_clusters, n_nearest):
        """
        Cluster the points with the given algorithm.
        """
        if algorithm_name == 'K-means':
            return KMeansAlgorithm(n_clusters).cluster(points)
        elif algorithm_name == 'Unnormalized':
            raise NotImplementedError("Unnormalized spectral clustering not implemented")
        elif algorithm_name == 'Normalized':
            raise NotImplementedError("Normalized spectral clustering not implemented")
        else:
            raise ValueError(f"Invalid algorithm name: {algorithm_name}")
        
    def recompute(self):
        """
        Recompute the clustering:
            * gather points from UI window
            * cluster data w/current settings
            * update cluster window w/results
        """
        n_points = self._windows['tools'].get_value('n_pts')
        algorithm_name = self._windows['tools'].get_value('algorithm')
        n_clusters = self._windows['tools'].get_value('k')

        if self._cluster_colors is None or self._cluster_colors.shape[0] != n_clusters:
            self._cluster_colors = get_n_disp_colors(n_clusters)
        n_nearest = self._windows['tools'].get_value('n_nearest')
        print("Recomputing with %i points, %i clusters, %i nearest neighbors, and algorithm %s" % (n_points, n_clusters, n_nearest, algorithm_name))
        points = self._windows['ui'].get_points(n_points)
        cluster_ids = self._do_clustering(points, algorithm_name, n_clusters, n_nearest)
        #import ipdb; ipdb.set_trace()
        self._windows['clusters'].update(points, cluster_ids, self._cluster_colors)


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
            elif self._active_window_name is not None:
                self._windows[self._active_window_name].keypress(k)

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
        for window_name in self._windows:
            self._windows[window_name].render(frame, active = (window_name == self._active_window_name))
        return frame

    def _get_window_name(self, x, y):
        """
        Return the window that contains the point (x, y).
        """
        for window_name in self._windows:
            if self._windows[window_name].contains(x, y):
                return window_name
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events:
            if we're in the middle of a click, pass the event to the active window,
            otherwise update the active window and pass the event to it.
        """
        self._mouse_pos = (x, y)
        if self._clicked_window_name is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                self._windows[self._clicked_window_name].mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._windows[self._clicked_window_name].mouse_unclick(x, y)
                self._clicked_window_name = None
            else:
                raise Exception("Can't click in a window while another window is clicked!")
            return
        self._active_window_name = self._get_window_name(x, y)
        if self._active_window_name is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._windows[self._active_window_name].mouse_click(x, y):
                    self._clicked_window_name = self._active_window_name
                    self._clicked_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                self._windows[self._active_window_name].mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                #raise Exception("Can't unclick in a window that wasn't clicked!")
                pass # will happen if double-clicking, so just do nothing.
        else:
            self._active_window_name = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cc = ClusterCreator()
    cc.run()
