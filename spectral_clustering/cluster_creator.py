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
from windows import UiWindow, ClustersWindow, ToolsWindow, SimMatrixWindow, ToolsWindow,GraphStatsWindow, SpectrumWindow
import cv2
import logging
import time
# all dims in unit square, to be scaled to window size
from layout import WINDOW_LAYOUT, Windows, Tools
from clustering import KMeansAlgorithm, SpectralAlgorithm
from util import get_n_disp_colors, unscale_coords
from spectral import SimilarityGraph


class ClusterCreator(object):

    def __init__(self, size=(640*2, 800)):
        logging.info('Initializing Cluster Creator')
        self._size = size
        self._bkg = np.zeros((size[1], size[0], 3), np.uint8) + WINDOW_LAYOUT['colors']['bkg']
        window_layout = self._get_layouts(WINDOW_LAYOUT['windows'], size)
        self.windows = {Windows.ui: UiWindow()

        self._active_window_name = None  # Mouse is over this window
        self._clicked_window_name = None  # Mouse was clicked in this window, but may not be over it
        self._mouse_pos = None
        self._cluster_colors = None  # update only when K changes
        self._clicked_pos = None
        self.show_cluster_ctrls = True
        self._fps_info = {'last_time': time.perf_counter(),
                          'n_frames': 0,
                          'update_sec': 2.0}

    def _get_layouts(self, layout, size):
        """
        Convert the relative dimensions in LAYOUT['windows'] to absolute dimensions now
        that we have the window size.

        Also, make special adjustments:
            * The similiarity matrix window should be square, so change its width
                to match its height.  Check whatever window is next to it has the same
                height, and adjust the width to match.
            * 
        """
        window_layout = {}
        for window_kind in layout:
            window = layout[window_kind]
            x0, x1 = window['x']
            y0, y1 = window['y']
            window_layout[window_kind] = {'x': (int(x0 * size[0]), int(x1 * size[0])),
                                          'y': (int(y0 * size[1]), int(y1 * size[1]))}
        # adjust sim_matrix window and it's neighbor
        sim_window_k = Windows.sim_matrix
        sim_neighbor_k = [w for w in layout if (layout[w]['y'] == layout[sim_window_k]['y']) and
                             (w != 'sim_matrix') and
                             (layout[sim_window_k]['x'][1] == layout[w]['x'][0])]
        if len(sim_neighbor_k) == 0:
            raise ValueError("No window with y=%s and x[1]=%f in layout." %
                             (layout[sim_window_k]['y'],layout[sim_window_k]['x'][1]))
        sim_neighbor_k = sim_neighbor_k[0]

        sim_height = window_layout[sim_window_k]['y'][1] - window_layout[sim_window_k]['y'][0]
        new_x = window_layout[sim_window_k]['x'][0] + sim_height
        print("Adjusting sim matrix window from %i pixels to %i." % (window_layout[sim_window_k]['x'][1] - window_layout[sim_window_k]['x'][0], sim_height))
        print("Adjusting %s window from %i pixels to %i." % (sim_neighbor_k, window_layout[sim_neighbor_k]['x'][1] - window_layout[sim_neighbor_k]['x'][0], sim_height))
        window_layout[sim_window_k]['x'] = (window_layout[sim_window_k]['x'][0], new_x)
        window_layout[sim_neighbor_k]['x'] = (new_x, window_layout[sim_neighbor_k]['x'][1])

        return window_layout

    def _do_clustering(self, points):
        """
        Cluster the points with the given algorithm.
        """
        algorithm_name = self.windows['tools'].get_value('algorithm')
        n_nearest = self.windows['tools'].get_value('n_nearest')
        epsilon = self.windows['tools'].get_value('epsilon')
        graph_type = self.windows['tools'].get_value('sim_graph')
        n_clusters = self.windows['tools'].get_value('k')

        # scale epsilon to unit square
        width = self.windows['ui'].bbox['x'][1] - self.windows['ui'].bbox['x'][0]
        epsilon = epsilon / width

        if algorithm_name == 'K-means':
            return KMeansAlgorithm(n_clusters).cluster(points)

        elif algorithm_name in ('Unnormalized', 'Normalized'):
            # spectral
            sim_graph = SimilarityGraph(points, kind=graph_type, epsilon_dist=epsilon, n_nearest=n_nearest)
            self.windows['sim_matrix'].set_graph(sim_graph)
            sa = SpectralAlgorithm(n_clusters, sim_graph)
            return sa.get_clusters()
        else:
            raise ValueError(f"Invalid algorithm name: {algorithm_name}")

    def recompute(self):
        """
        Recompute the clustering:
            * gather points from UI window
            * cluster data w/current settings
            * update cluster window w/results
        """
        n_points = self.windows['tools'].get_value('n_pts')
        n_clusters = self.windows['tools'].get_value('k')

        if self._cluster_colors is None or self._cluster_colors.shape[0] != n_clusters:
            self._cluster_colors = get_n_disp_colors(n_clusters)
        # print("Recomputing with %i points, %i clusters, %i nearest neighbors, and algorithm %s" % (n_points, n_clusters, n_nearest, algorithm_name))
        points = self.windows['ui'].get_points(n_points)
        unit_points = unscale_coords(self.windows['ui'].bbox, points)

        cluster_ids = self._do_clustering(unit_points)

        self.windows['clusters'].update(unit_points, cluster_ids, self._cluster_colors)

    def clear(self):
        """
        Clear the clusters.
        self.windows['ui'].clear()
        """
        self.windows['ui'].clear()
        self.windows['clusters'].clear()
        self.windows['sim_matrix'].clear()

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
            elif k == ord(' '):
                self.show_cluster_ctrls = not self.show_cluster_ctrls
                print("Show cluster controls: %s" % self.show_cluster_ctrls)
            elif self._active_window_name is not None:
                self.windows[self._active_window_name].keypress(k)

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
        for window_name in self.windows:
            self.windows[window_name].render(frame, active=(window_name == self._active_window_name))
        return frame

    def _get_window_name(self, x, y):
        """
        Return the window that contains the point (x, y).
        """
        for window_name in self.windows:
            if self.windows[window_name].contains(x, y):
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
                self.windows[self._clicked_window_name].mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.windows[self._clicked_window_name].mouse_unclick(x, y)
                self._clicked_window_name = None
            else:
                raise Exception("Can't click in a window while another window is clicked!")
            return
        self._active_window_name = self._get_window_name(x, y)
        if self._active_window_name is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.windows[self._active_window_name].mouse_click(x, y):
                    self._clicked_window_name = self._active_window_name
                    self._clicked_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                self.windows[self._active_window_name].mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                # raise Exception("Can't unclick in a window that wasn't clicked!")
                pass  # will happen if double-clicking, so just do nothing.
        else:
            self._active_window_name = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cc = ClusterCreator()
    cc.run()
