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
from windows import WINDOW_TYPES, WINDOW_NAMES
import cv2
import logging
import time
# all dims in unit square, to be scaled to window size
from layout import WINDOW_LAYOUT, Windows, Tools
from clustering import KMeansAlgorithm, SpectralAlgorithm
from util import get_n_disp_colors, unscale_coords
from spectral import SimilarityGraphTypes, EpsilonSimGraph, FullSimGraph, NNSimGraph,SIMGRAPH_KIND_NAMES
from threading import Thread, Lock, get_ident

HOTKEYS = {'toggle graph view': 'g',
           'toggle cluster controls': ' ',
           'recalculate clustering': 'r',
           'quit': 'q',
           'print hotkeys': 'h'}


class ClusterCreator(object):
    APP_WINDOWS = [Windows.ui,
                   Windows.toolbar,
                   Windows.clustering,
                   Windows.sim_matrix,
                   Windows.eigenvectors,
                   Windows.rand_proj,
                   Windows.spectrum]

    def __init__(self, size=(1350, 800)):
        logging.info('Initializing Cluster Creator')
        self.size = size
        self.bbox = {'x': (0, size[0]), 'y': (0, size[1])}
        self._bkg = np.zeros((size[1], size[0], 3), np.uint8) + WINDOW_LAYOUT['colors']['bkg']
        window_layout = self._get_layouts(WINDOW_LAYOUT['windows'], size)
        self.windows = {}  # WINDOW_TYPES[k]: WINDOW_TYPES[k](window_layout[k], self) for k in self.APP_WINDOWS}
        for k in self.APP_WINDOWS:
            print("Initializing window %s" % WINDOW_NAMES[k])
            print("\tLayout: %s" % window_layout[k])
            self.windows[k] = WINDOW_TYPES[k](window_layout[k], self)

        self._active_window_name = None  # Mouse is over this window
        self._clicked_window_name = None  # Mouse was clicked in this window, but may not be over it
        self._mouse_pos = None
        self._cluster_colors = None  # update only when K changes
        self._clicked_pos = None
        self.show_cluster_ctrls = True
        self.show_graph = False
        self._fps_info = {'last_time': time.perf_counter(),
                          'n_frames': 0,
                          'update_sec': 2.0}

        self._points = np.zeros((0, 2))
        self._spectrum = None  # SpectralAlgorithm object

        # Asynchronously updated components (if enabled in update_sim_graph):
        self._similarity_graph = {'graph': None, 'lock': Lock(), 'thread': None}
        self._cluster_ids = {'ids': None, 'lock': Lock(), 'thread': None}

        self._print_hotkeys()

    def _print_hotkeys(self):
        print("\n\n\t\tHotkeys:")
        for k, v in HOTKEYS.items():
            print("\t\t\t'%s' - %s" % (v,k))
        print("\n\n")

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
        for window_kind in self.APP_WINDOWS:
            window = layout[window_kind]
            x0, x1 = window['x']
            y0, y1 = window['y']
            window_layout[window_kind] = {'x': (int(x0 * size[0]), int(x1 * size[0])),
                                          'y': (int(y0 * size[1]), int(y1 * size[1]))}
        # adjust sim_matrix window and the toolbar to the left of it
        sim_window_k = Windows.sim_matrix
        sim_neighbor_k = Windows.toolbar

        sim_height = window_layout[sim_window_k]['y'][1] - window_layout[sim_window_k]['y'][0]
        new_x = window_layout[sim_window_k]['x'][1] - sim_height
        print("Adjusting sim matrix window from %i pixels to %i." %
              (window_layout[sim_window_k]['x'][1] - window_layout[sim_window_k]['x'][0], sim_height))
        print("Adjusting %s window from %i pixels to %i." % (sim_neighbor_k,
              window_layout[sim_neighbor_k]['x'][1] - window_layout[sim_neighbor_k]['x'][0], sim_height))
        window_layout[sim_window_k]['x'] = (new_x,window_layout[sim_window_k]['x'][1])
        window_layout[sim_neighbor_k]['x'] = (window_layout[sim_neighbor_k]['x'][0], new_x)

        return window_layout

    def update_points(self):
        """
        Get new points from the UI window.
        """
        # print("Main thread updating points")
        self._points = self.windows[Windows.ui].get_points()
        self.update_sim_graph()

    def update_sim_graph(self, param_val=None, asynch=False):
        """
        Points, clusters, or sim-graph type/param has changed, recompute it in its own thread.
        :param param_val: the value of the parameter that changed (ignored since they're looked up)

        IF asynch:
          if the sim graph is already being updated, stop it and start a new one.
        """
        def update_sim_graph_thread():
            # first, clear the current graph so it doesn't render while we're updating it
            #print("Updating similarity graph with %i points in thread: %s " % (self._points.shape[0], get_ident()))
            with self._similarity_graph['lock']:
                self._similarity_graph['graph'] = None

            unit_points = self._points  # unscale_coords(self.windows[Windows.ui].bbox, self._points)
            if unit_points.shape[0] == 0:
                print("No points to cluster.")
                return
            
            graph_kind = self.windows[Windows.toolbar].get_value('sim_graph')
            if graph_kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.FULL]:
                sigma = self.windows[Windows.toolbar].get_value['sigma']
                sim_graph = FullSimGraph(unit_points, sigma)
            elif graph_kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.NN]:
                n_nearest = self.windows[Windows.toolbar].get_value('n_nearest')
                mutual = self.windows[Windows.toolbar].get_value('mutual')

                sim_graph = NNSimGraph(unit_points, n_nearest, mutual)
            elif graph_kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.EPSILON]:
                epsilon_dist=self.windows[Windows.toolbar].get_value('epsilon')
                sim_graph = EpsilonSimGraph(unit_points, epsilon_dist)
            else:
                raise ValueError("Invalid similarity graph type: %s" % graph_kind)

            with self._similarity_graph['lock']:
                self._similarity_graph['graph'] = sim_graph
                self.windows[Windows.sim_matrix].set_graph(sim_graph)
                self.windows[Windows.ui].set_graph(sim_graph)
            #print("\tDone updating similarity graph in thread:  ", get_ident())
        if asynch:
            with self._similarity_graph['lock']:
                if self._similarity_graph['thread'] is not None:
                    self._similarity_graph['thread'].join()
                self._similarity_graph['thread'] = Thread(target=update_sim_graph_thread)
                self._similarity_graph['thread'].start()
            # print("Started similarity graph thread:  ", get_ident())
        else:
            update_sim_graph_thread()

    def recompute_spectrum(self):
        """
        Called when run button is clicked (because similarity graph has changed).
        This will also update the clustering.
        """
        algorithm_name = self.windows[Windows.toolbar].get_value('algorithm')

        if algorithm_name == 'K-means':
            return
        elif algorithm_name == 'Spectral':
            self._spectrum =  SpectralAlgorithm(self._similarity_graph['graph'])
            values, vectors = self._spectrum.get_eigens()
            self.windows[Windows.spectrum].set_values(values)
            self.windows[Windows.eigenvectors].set_values(vectors)
        else:
            raise ValueError(f"Invalid algorithm name: {algorithm_name}")
        
        self.recompute_clustering()

    def recompute_clustering(self):
        """     
        Called when spectrum is recomputed or arguments to clustering algorithm are changed.
        Update windows:  clustering, rand_proj
        """       
        n_clusters = self.windows[Windows.toolbar].get_value('k')
        n_features = self.windows[Windows.toolbar].get_value('f')
        algorithm_name = self.windows[Windows.toolbar].get_value('algorithm')
        if algorithm_name == 'K-means':
            ka = KMeansAlgorithm(n_clusters)
            cluster_ids = ka.cluster(self._points)
        elif algorithm_name == 'Spectral':
            if self._spectrum is None:
                # self.recompute_spectrum()  # too slow to automatically recompute
                return
            cluster_ids = self._spectrum.cluster(n_clusters, n_features)

        if self._cluster_colors is None or self._cluster_colors.shape[0] != n_clusters:
            self._cluster_colors = get_n_disp_colors(n_clusters)

        #unit_points = unscale_coords(self.windows[Windows.ui].bbox, self._points)
        clustering = {'points': self._points, 'cluster_ids': cluster_ids, 'colors': self._cluster_colors}
        self.windows[Windows.clustering].set_values(clustering)

    def clear(self):
        """
        Clear the clusters.
        self.windows[Windows.ui].clear()
        """
        self._points = np.zeros((0, 2))
        self.windows[Windows.ui].clear()
        self.windows[Windows.clustering].clear()
        self.windows[Windows.sim_matrix].clear()
        self.windows[Windows.eigenvectors].clear()
        self.windows[Windows.rand_proj].clear()
        self.windows[Windows.spectrum].clear()

        with self._similarity_graph['lock']:
            self._similarity_graph['graph'] = None
        with self._cluster_ids['lock']:
            self._cluster_ids['ids'] = None

    def run(self):
        """
        Run the cluster creator app.
        """
        cv2.namedWindow('Cluster Creator')
        cv2.setMouseCallback('Cluster Creator', self._mouse_callback)

        logging.info('Running Cluster Creator')
        while True:
            # print(len(self.windows[Windows.ui]._points), len(self.windows[Windows.ui]._clusters))
            # refresh
            frame = self._refresh()

            cv2.imshow('Cluster Creator', frame)

            # display & send key event
            k = cv2.waitKey(1) & 0xFF
            if k & 0xFF == 27 or k == ord(HOTKEYS['quit']):
                break
            elif k == ord(HOTKEYS['toggle cluster controls']):
                self.show_cluster_ctrls = not self.show_cluster_ctrls
                print("Show cluster controls: %s" % self.show_cluster_ctrls)
            elif k == ord(HOTKEYS['toggle graph view']):
                self.show_graph = not self.show_graph
                print("Show graph: %s" % self.show_graph)
            elif k == ord(HOTKEYS['recalculate clustering']):
                self.recompute_clustering()
            elif k == ord(HOTKEYS['print hotkeys']):
                self._print_hotkeys()
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
