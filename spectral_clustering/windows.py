"""
Handle sub-windows for the cluster creator app.
"""
import numpy as np
from abc import ABC, abstractmethod
import cv2
from util import (scale_bbox, get_good_point_size, add_sub_image,
                  hsplit_bbox, indent_bbox, vsplit_bbox)
from mpl_plots import plot_clustering, plot_eigenvecs, add_alpha
import logging
from tools import RadioButtons, Slider, Button, ToggleButton
from clustering import render_clustering, KMeansAlgorithm
from colors import COLORS
from layout import WINDOW_LAYOUT, TOOLBAR_LAYOUT, OTHER_TOOL_LAYOUT, Windows, Tools
from clusters import EllipseCluster, AnnularCluster, CLUSTER_TYPES
from spectral import SimilarityGraphTypes, SIMGRAPH_PARAM_NAMES, SIMGRAPH_KIND_NAMES
from plot_to_img import PlotRenderer


WINDOW_NAMES = {Windows.ui: "UI",  # default text to render in windows
                Windows.toolbar: "Toolbar",
                Windows.clustering: "Clusters",
                Windows.spectrum: "Spectrum",
                Windows.eigenvectors: "Eigenvectors",
                Windows.sim_matrix: "Similarity matrix",
                Windows.rand_proj: "Random projection"}


class Window(ABC):
    """
    Abstract class for windows in the cluster creator app.
    Each instance reprseents a sub-window, is responsible for rendering itself onto the main window,
    and handling user input.  Instances will only get mouse/keyboard events when the mouse is within
    the window's bounding box.
    """

    def __init__(self, kind, bbox, app):
        """
        :param kind: one of the values in layout.Windows(IntEnum)
        :param app: the main app instance
        :param bbox: bounding box of the window in pixels {'x': (x_min, x_max), 'y': (y_min, y_max)}
        """
        self.app = app
        self.bbox = bbox
        self.margin_px = int(WINDOW_LAYOUT['dims']['margin_px'])
        self.colors = WINDOW_LAYOUT['colors']
        self._txt = {'name': WINDOW_NAMES[kind],
                     'font': WINDOW_LAYOUT['font'],
                     'font_size': WINDOW_LAYOUT['font_size'],
                     'font_thickness': WINDOW_LAYOUT['font_thickness'],
                     'color': self.colors['font'].tolist()}
        self._title_height = cv2.getTextSize(self._txt['name'],
                                             self._txt['font'],
                                             self._txt['font_size'],
                                             self._txt['font_thickness'])[0][1]

        logging.info(f"Created window {self._txt['name']} with bbox {self.bbox}")

    def contains(self, x, y):
        """
        Return True if the point (x, y) is within the window.
        """
        x0, x1 = self.bbox['x']
        y0, y1 = self.bbox['y']
        return x0 <= x <= x1 and y0 <= y <= y1

    def render_title(self, img):
        """
        Render the window title.
        """
        x, y = self.bbox['x'][0], self.bbox['y'][0] + self._title_height
        indent_px = self.margin_px + 10
        pos = (x+indent_px,
               y+indent_px)

        cv2.putText(img, self._txt['name'], pos, self._txt['font'], self._txt['font_size'],
                    self._txt['color'], self._txt['font_thickness'], cv2.LINE_AA)
        
    def render_box(self, img, active=False):
        """
        Render the window box.
        """
        x0, x1 = self.bbox['x']
        y0, y1 = self.bbox['y']
        x0 += self.margin_px
        y0 += self.margin_px
        x1 -= self.margin_px
        y1 -= self.margin_px
        color = self.colors['active_border'] if active else self.colors['border']
        cv2.rectangle(img, (x0, y0), (x1, y1), color.tolist(), 2)

    def render(self, img, active=False):
        """
        Render the window onto the image.
        (override for specific window types)
        """
        self.render_box(img, active=active)
        self.render_title(img)

    def keypress(self, key):
        """
        Handle a keypress event.
        """
        pass

    def mouse_click(self, x, y):
        """
        Handle a mouse click event.
        :returns: True if future mouse events should be sent to this window until unclick is called
        """
        pass

    def mouse_unclick(self, x, y):
        """
        Handle a mouse unclick event.
        """
        pass

    def mouse_move(self, x, y):
        """
        Handle a mouse move event.
        """
        pass


class UiWindow(Window):
    """
    Window for the cluster creator UI.
    Manage clusters, points.  Render similarity graph.
    """

    def __init__(self, bbox, app):
        super().__init__(Windows.ui, bbox, app)
        self._mouseover_ind = None
        self._adjusting_ind = None
        self._mouse_pos = None
        self._sim_graph = None
        self._clicked_pos = None
        self._clusters = []

    def get_points(self):
        points = [cluster.get_points() for cluster in self._clusters]
        if len(points) == 0:
            return np.array([])
        return np.vstack(points)

    def n_pts_slider_callback(self, n_pts):
        for cluster in self._clusters:
            cluster.set_n_pts(int(n_pts))

        # update the similarity graph, but not clustering yet
        self.app.update_points()

    def render(self, img, active=False):
        self.render_box(img, active=active)  # remove?

        # draw edges of similiarity graph
        if self.app.show_graph and self._sim_graph is not None:
            self._sim_graph.draw_graph(img)

        # draw points
        n_points = self.app.windows[Windows.toolbar].get_value('n_pts') * len(self._clusters)
        pts_size = get_good_point_size(n_points, self.bbox)

        for cluster in self._clusters:
            cluster.render(img, show_ctrls=self.app.show_cluster_ctrls, pt_size=pts_size)

        # render cluster metadata
        status = "N clusters: %i" % len(self._clusters)
        status_height = cv2.getTextSize(
            status, WINDOW_LAYOUT['font'], WINDOW_LAYOUT['font_size'], WINDOW_LAYOUT['font_thickness'])[0][1]
        text_pos = (self.bbox['x'][0] + self.margin_px*2, self.bbox['y'][0] + self.margin_px*2 + status_height)
        cv2.putText(img, status, text_pos, WINDOW_LAYOUT['font'], WINDOW_LAYOUT['font_size'], self.colors['font'].tolist(
        ), WINDOW_LAYOUT['font_thickness'], cv2.LINE_AA)

    def get_cluster_color_ids(self):
        """
        :returns: dict(ids=[cluster id list], colors=[color list])
        where each cluster id corresponds to a color in the color list
        """
        colors = []
        ids = []
        for color_id, cluster in enumerate(self._clusters):
            colors.append(cluster.get_color()[::-1])
            ids.append(np.ones(cluster.get_n_pts()).astype(int) * color_id)
        return {'ids': np.hstack(ids), 'colors': np.vstack(colors)}

    def set_graph(self, new_sim_graph):
        # just created from current set of points, so
        # be sure to delete when points/clusters change.
        self._sim_graph = new_sim_graph

    def keypress(self, key):
        pass

    def _update_mouse_pos(self, x, y):
        """
        Update internal state.
        Determine which control point of which cluster is being hovered over.
        :returns: cluster_ind 
        """
        self._mouse_pos = (x, y)
        self._mouseover_ind = None
        for i, cluster in enumerate(self._clusters):
            if cluster.update_mouse_pos(x, y):
                self._mouseover_ind = i
                return i
        return None

    def mouse_click(self, x, y):
        """
        Handle a mouse click event.
        :returns: True if future mouse events should be sent to this window until unclick is called
        """
        clicked_ind = self._update_mouse_pos(x, y)
        self._clicked_pos = (x, y)

        if clicked_ind is not None:
            self._adjusting_ind = clicked_ind
            return self._clusters[clicked_ind].start_adjusting()
        else:
            self._clusters.append(self._create_cluster(x, y))
            self._clusters[-1].update_mouse_pos(x, y)
            self._clusters[-1].start_adjusting()
            self._adjusting_ind = len(self._clusters) - 1
            self.app.update_points()  # inform the app that the points have changed
            return True

    def clear(self):
        """
        Clear all clusters.
        """
        self._clusters = []
        self._sim_graph = None

    def mouse_unclick(self, x, y):
        """
        Send the unclick event to the cluster being adjusted.
        Un-hold it.
        """
        if self._adjusting_ind is not None:
            if self._clusters[self._adjusting_ind].stop_adjusting(self.bbox):
                # if the cluster wants to be removed
                del self._clusters[self._adjusting_ind]
                self.app.update_points()

            self._adjusting_ind = None
        self._update_mouse_pos(x, y)

    def mouse_move(self, x, y):
        """
        If a cluster is being held, send it the move signal, 
        else just update the mouse position.
        """
        if self._adjusting_ind is not None:
            self._clusters[self._adjusting_ind].drag_ctrl(x, y)
            self.app.update_points()
        else:
            self._update_mouse_pos(x, y)

    def _create_cluster(self, x, y):
        """
        Lookup current params, create a cluster with them.
        """
        c_name = self.app.windows[Windows.toolbar].get_value('kind')
        n_points = self.app.windows[Windows.toolbar].get_value('n_pts')

        cluster = CLUSTER_TYPES[c_name](x, y, n_points, self.bbox)
        return cluster


class WindowMouseManager(ABC):
    """
    Classes with a member '.tools' that is a list of Tool objects can
    inherit from this to handle mouse signals.
    """

    def __init__(self):
        self._held_tool = None

    def mouse_click(self, x, y):
        """
        Send to all tools, each will handle if it was clicked.
        Return if a tool captures the mouse.
        """
        for tool in self.tools.values():
            if tool.mouse_click(x, y):
                self._held_tool = tool
                return True
        return False

    def mouse_unclick(self, x, y):
        if self._held_tool is not None:
            self._held_tool.mouse_unclick(x, y)
            self._held_tool = None

    def mouse_move(self, x, y):
        if self._held_tool is not None:
            return self._held_tool.mouse_move(x, y)
        else:
            for tool in self.tools.values():
                tool.mouse_move(x, y)
        return False


class ToolsWindow(WindowMouseManager, Window):
    """
    Create & manage toolbox for the app.
    """

    def __init__(self, bbox, app):
        super(WindowMouseManager, self).__init__(Windows.toolbar, bbox, app)
        super().__init__()
        self._setup_tools()

    def _setup_tools(self):
        indent_px = 5
        indented_bbox = {'x': (self.bbox['x'][0] + indent_px, self.bbox['x'][1] - indent_px),
                         'y': (self.bbox['y'][0] + indent_px, self.bbox['y'][1] - indent_px)}
        # What parameter (name) does each slider control:

        # What are the kinds of sim graphs that can be selected:
        self._sim_kind_names = {Tools.nn_slider: SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.NN],
                                Tools.epsilon_slider: SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.EPSILON],
                                Tools.sigma_slider: SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.FULL]}

        self.tools = {Tools.kind_radio: RadioButtons(scale_bbox(TOOLBAR_LAYOUT[Tools.kind_radio], indented_bbox),
                                                     'Cluster Type', lambda x: None,  # no callback, updates when user starts a new cluster
                                                     options=['Gaussian', 'Ellipse', 'Annulus'],
                                                     default_selection=1, spacing_px=7),
                      Tools.alg_radio: RadioButtons(scale_bbox(TOOLBAR_LAYOUT[Tools.alg_radio], indented_bbox),
                                                    'Algorithm', lambda x: None,    # no callback, updates when Run button is clicked
                                                    options=['Spectral', 'K-means'],
                                                    default_selection=0, spacing_px=9),
                      Tools.sim_graph_radio: RadioButtons(scale_bbox(TOOLBAR_LAYOUT[Tools.sim_graph_radio], indented_bbox),
                                                          'Sim Graph', callback=self._change_sim_param_visibility,
                                                          options=[self._sim_kind_names[Tools.nn_slider],
                                                                   self._sim_kind_names[Tools.epsilon_slider],
                                                                   self._sim_kind_names[Tools.sigma_slider]],
                                                          default_selection=1, spacing_px=9),
                      Tools.n_pts_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.n_pts_slider], indented_bbox),
                                                 'Num Pts', self.app.windows[Windows.ui].n_pts_slider_callback,
                                                 range=[5, 2000], default=100, format_str="=%i"),
                      Tools.k_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.k_slider], indented_bbox),
                                             # no callback, updates when Run button is clicked
                                             'K (clusters)', self._update_k_slider,
                                             range=[2, 20], default=3, format_str="=%i"),
                      Tools.f_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.f_slider], indented_bbox),
                                             'F (features)', self._update_f_slider,
                                             range=[2, 20], default=3, format_str="=%i"),
                      Tools.run_button: Button(scale_bbox(TOOLBAR_LAYOUT[Tools.run_button], indented_bbox), 'Run',
                                                callback=self.app.recompute_spectrum, border_indent=1),
                      Tools.clear_button: Button(scale_bbox(TOOLBAR_LAYOUT[Tools.clear_button], indented_bbox), 'Clear', 
                                                 callback=self.app.clear, border_indent=1),

                      # Only one of these three is on at a time:
                      Tools.nn_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.nn_slider], indented_bbox),
                                              SIMGRAPH_PARAM_NAMES[SimilarityGraphTypes.NN],
                                              self.app.update_sim_graph,
                                              range=[1, 20], default=5, format_str="=%i", visible=False),
                      Tools.nn_toggle: ToggleButton(scale_bbox(TOOLBAR_LAYOUT[Tools.nn_toggle], indented_bbox),
                                                    'Mutual',
                                                    self.app.update_sim_graph,
                                                    default=True, visible=False, spacing_px=5, border_indent=2),

                      Tools.epsilon_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.epsilon_slider], indented_bbox),
                                                   SIMGRAPH_PARAM_NAMES[SimilarityGraphTypes.EPSILON],
                                                   self.app.update_sim_graph,
                                                   range=[1., 50], default=25, format_str="=%.3f", visible=True),
                      Tools.sigma_slider: Slider(scale_bbox(TOOLBAR_LAYOUT[Tools.sigma_slider], indented_bbox),
                                                 SIMGRAPH_PARAM_NAMES[SimilarityGraphTypes.FULL],
                                                 self.app.update_sim_graph,
                                                 range=[1., 500], default=100, format_str="=%.3f", visible=False)}

        logging.info(f"Created tools window with {len(self.tools)} tools")

    def _update_k_slider(self, k):
        self.app.recompute_clustering(False)

    def _update_f_slider(self, f):
        self.app.recompute_clustering(True)
        self.app.windows[Windows.spectrum].refresh()
        self.app.windows[Windows.eigenvectors].refresh()



    def _change_sim_param_visibility(self, kind_name):
        """
        Set the visibility of the sim_param sliders based on the current sim_graph_radio selection.
        kind_name:  name of the kind of similarity graph to show sliders for.
        """
        turn_on_toggle = False
        for param_kind in self._sim_kind_names:
            if self._sim_kind_names[param_kind] == kind_name:
                self.tools[param_kind].set_visible(True)
                if param_kind == Tools.nn_slider:
                    turn_on_toggle = True
            else:
                self.tools[param_kind].set_visible(False)

        if turn_on_toggle:
            self.tools[Tools.nn_toggle].set_visible(True)
        else:
            self.tools[Tools.nn_toggle].set_visible(False)

        self.app.update_sim_graph()

    def render(self, img, active=False):
        # super().render(img, active=active)
        w, h = self.bbox['x'][1] - self.bbox['x'][0], self.bbox['y'][1] - self.bbox['y'][0]
        # cv2.rectangle(img, (self.margin_px, self.margin_px),
        # (w - self.margin_px, h - self.margin_px),
        #              LAYOUT['colors']['border'].tolist(), 2)
        for tool in self.tools.values():
            tool.render(img)

    def keypress(self, key):
        pass

    def get_value(self, param):
        if param == 'kind':
            return self.tools[Tools.kind_radio].get_value()
        elif param == 'algorithm':
            return self.tools[Tools.alg_radio].get_value()
        elif param == 'n_nearest':
            return int(self.tools[Tools.nn_slider].get_value())
        elif param == 'n_pts':
            return int(self.tools[Tools.n_pts_slider].get_value())
        elif param == 'epsilon':
            return self.tools[Tools.epsilon_slider].get_value()
        elif param == 'sigma':
            return self.tools[Tools.sigma_slider].get_value()
        elif param == 'sim_graph':
            return self.tools[Tools.sim_graph_radio].get_value()
        elif param == 'k':
            return int(self.tools[Tools.k_slider].get_value())
        elif param == 'f':
            return int(self.tools[Tools.f_slider].get_value())
        elif param == 'mutual':
            return self.tools[Tools.nn_toggle].get_value()
        else:
            raise ValueError(f"Invalid parameter: {param}")


class SimMatrixWindow(Window):
    def __init__(self, bbox, app):
        super().__init__(Windows.sim_matrix, bbox, app)
        self._img_bbox = {'x': (self.bbox['x'][0] + self.margin_px, self.bbox['x'][1] - self.margin_px),
                          'y': (self.bbox['y'][0] + self.margin_px, self.bbox['y'][1] - self.margin_px)}

        w = (self._img_bbox['x'][1] - self._img_bbox['x'][0])
        h = (self._img_bbox['y'][1] - self._img_bbox['y'][0])
        if h != w:
            raise ValueError("SimMatrixWindow must be square")
        self._s = w
        self._m = None  # the similarity matrix
        self._image_rgb_resized = None
        self._colormap = cv2.COLORMAP_HOT

    def set_graph(self, sim_mat):
        """
        Set to display on next render.
        :param sim_mat: spectral.SimilarityGraph instance
        """
        self._m = sim_mat.get_matrix()
        # img_full = image_from_floats(self._m)
        img_full = sim_mat.make_img(self._colormap)
        img_resized = cv2.resize(img_full, (self._s, self._s), interpolation=cv2.INTER_NEAREST)
        # cv2.merge((img_resized, img_resized, img_resized))  # colormap handles this now
        self._image_rgb_resized = img_resized

    def clear(self):
        self._m = None
        self._image_rgb_resized = None

    def render(self, img, active=False):
        """
        Render the window onto the image.
        For now grayscale, min_value=(0,0,0), max_value=(255,255,255)
        """
        if self._image_rgb_resized is None:
            super().render(img, active=active)  # default window render
        else:
            img[self._img_bbox['y'][0]:self._img_bbox['y'][0]+self._s,
                self._img_bbox['x'][0]:self._img_bbox['x'][0]+self._s] = self._image_rgb_resized


class PlotWindow(Window, ABC):
    def __init__(self,  kind, bbox, app, tool_frac=0.0, split='h'):
        """
        Plot on the left, tool area to the right/below, split proportional to tool_frac.
        """
        super().__init__(kind, bbox, app)
        if split == 'h':
            self._plot_bbox, self._tool_bbox = hsplit_bbox(bbox, [1-tool_frac, tool_frac])
        elif split == 'v':
            self._plot_bbox, self._tool_bbox = vsplit_bbox(bbox, [1-tool_frac, tool_frac])
        else:
            raise ValueError(f"Invalid split: {split}")

        self._tool_bbox = indent_bbox(self._tool_bbox, self.margin_px)
        w = self._plot_bbox['x'][1] - self._plot_bbox['x'][0]
        h = self._plot_bbox['y'][1] - self._plot_bbox['y'][0]
        self._plotter = PlotRenderer((w, h))

        self._disp_img = None
        self._values = None
        self.tools = {}  # dict with values are all the tool objects to render (after disp_img)
        self._init_tools()

    def render(self, img, active=False):
        if self._disp_img is None:
            self.render_box(img, active=active)  # remove?
            self.render_title(img)
        else:
            add_sub_image(img, self._disp_img, self._plot_bbox)

            # render tools:
            for tool in self.tools:
                self.tools[tool].render(img)

    def set_values(self, values):
        self._values = values
        self.refresh()

    def clear(self):
        self._values = None
        self._disp_img = None

    @abstractmethod
    def refresh(self):
        # set self._disp_img
        # only called when self._values is set (by default) to create a new display image.
        pass

    @abstractmethod
    def _init_tools(self):
        # set self.tools = dict(tool_id: Tool)
        pass


class ClustersWindow(PlotWindow):
    def __init__(self,  bbox, app):
        super().__init__(Windows.clustering, bbox, app)
        self._bbox_size = (bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0])

    def _init_tools(self):
        pass

    def refresh(self):
        if self._values is None:
            return

        fig, axes = self._plotter.get_axis(1, 1)
        colors = self._values['colors']
        points = self._values['points']
        cluster_ids = self._values['cluster_ids']
        k = self.app.windows[Windows.toolbar].get_value('k')

        colors4 = add_alpha(colors, 0.5) / 255.
        plot_clustering(axes, points, colors4, cluster_ids, image_size=self._bbox_size, alpha=0.5)
        # add title
        axes.set_title("Clustering (k=%i)" % k)
        fig.tight_layout()
        self._disp_img = self._plotter.render_fig(fig)


class SpectrumWindow(WindowMouseManager, PlotWindow):
    def __init__(self, bbox, app):
        self._n_to_plot = 10
        super().__init__()
        super(WindowMouseManager, self).__init__(Windows.spectrum, bbox,
                                                 app, tool_frac=OTHER_TOOL_LAYOUT['spectrum_slider_w_frac'])

    def _init_tools(self):
        # init slider
        slider = Slider(self._tool_bbox, 'n', self._update_notifications, orient='vertical',
                        range=[1, 20], default=self._n_to_plot, format_str="=%i",
                        spacing_px=0,
                        visible=False)
        self.tools = {'n': slider}

    def _update_notifications(self, n):
        """
        We own this control, so we need to update anyone else who needs it.
        """
        self.update_n_plot(n)
        self.app.windows[Windows.eigenvectors].update_n_plot(n)

    def refresh(self):
        if self._values is None:
            return
        fig, ax = self._plotter.get_axis()

        n_features = self.app.windows[Windows.toolbar].get_value('f') 
        if self._n_to_plot < n_features:
            self._n_to_plot = n_features


        ax.plot(self._values[:self._n_to_plot], 'o-')
        # draw a vertical red line at f
        x_k = n_features - 0.5
        ax.axvline(x=x_k, color='r', linestyle='-')
        ax.set_title("Eigenvalues")
        self._disp_img = self._plotter.render_fig(fig)

    def update_n_plot(self, n):
        self._n_to_plot = int(n)
        self.refresh()

    def get_n_to_plot(self):
        return self._n_to_plot

    def set_values(self, values):
        super().set_values(values)
        self.tools['n'].set_visible(True)

    def clear(self):
        super().clear()
        self.tools['n'].set_visible(False)


class EigenvectorsWindow(PlotWindow):
    def __init__(self, bbox, app):
        super().__init__(Windows.eigenvectors, bbox, app, tool_frac=0)  # no tools

    def _init_tools(self):
        pass

    def refresh(self):
        if self._values is None:
            return
        n_to_plot = self.app.windows[Windows.spectrum].get_n_to_plot()
        k = self.app.windows[Windows.toolbar].get_value('f')
        fig, axes = self._plotter.get_axis(n_to_plot, 1, sharex=True, sharey=True)
        colors = self.app.windows[Windows.ui].get_cluster_color_ids()
        plot_eigenvecs(fig, axes, self._values, n_to_plot, k, colors=colors)
        self._disp_img = self._plotter.render_fig(fig)

    def update_n_plot(self, n):
        self._n_to_plot = int(n)
        self.refresh()


class RandProjWindow(WindowMouseManager, PlotWindow):
    """
    Project the data in feature space down to two random axes,
    plot it as a clustering with original colors.
    Use a slider to control noise added to the points,
    and a button to generate new random axes.
    """

    def __init__(self, bbox, app):
        # self._values will be the data to project, an N x F array
        self._noise = 0.1
        self._noise_offsets = None  # 2d, scale by _noise and add to data after projecting
        self._axes = None  # 2xf, orthogonal vectors in feature space
        # calls WindowMouseManager.__init__
        super().__init__()
        # calls PlotWindow.__init__
        super(WindowMouseManager, self).__init__(Windows.rand_proj, bbox,
                                                 app, tool_frac=OTHER_TOOL_LAYOUT['rand_proj_button_h_frac'],
                                                 split='v')
        self._bbox_size = (bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0])

    def _init_tools(self):
        """
        Random Projection window has two controls:
            - a button to generate 2 random orthoganal vectors in feature space
            - a slider to control the noise added to each point
        """
        self._slider_bbox, self._button_bbox = hsplit_bbox(self._tool_bbox, [3, 1])
        self.tools = {'noise_slider': Slider(self._slider_bbox, 'noise', self._update_noise, orient='horizontal',
                                             range=[0, 1], default=self._noise, format_str="=%.2f", visible=True),
                      'project_button': Button(self._button_bbox, 'randomize', self._remake_axes, visible=True)}

    def _update_noise(self, noise):
        self._noise = noise
        self.refresh()

    def set_values(self, values):
        """
        Override to create noise vector every time we
        get a new set of feature vectors.
        """
        self._values = values
        self._noise_offsets = np.random.randn(self._values.shape[0]*2).reshape(-1, 2)
        self._remake_axes()
        self.tools['noise_slider'].set_visible(True)
        self.tools['project_button'].set_visible(True)
        self.refresh()

    def _remake_axes(self):
        if self._values is None:
            return
        # make random axes, orthogonal in feature space
        self._axes = np.random.randn(2, self._values.shape[1])
        # normalize lengths
        self._axes[0] /= np.linalg.norm(self._axes[0])
        self._axes[1] -= self._axes[1] @ self._axes[0] * self._axes[0]
        self._axes[1] /= np.linalg.norm(self._axes[1])
        # check orthogonality
        err = np.abs(np.sum(self._axes[0] * self._axes[1]))
        if err > 1e-6:
            raise ValueError("Axes are not orthogonal!?")
        self.refresh()

    def refresh(self):
        if self._values is None:
            return
        points = self._values @ self._axes.T
        noisy_points = points + self._noise_offsets * self._noise * .1
        fig, ax = self._plotter.get_axis()
        colors = self.app.windows[Windows.ui].get_cluster_color_ids()
        # import ipdb; ipdb.set_trace()

        plot_clustering(ax, noisy_points, colors['colors']/255., colors['ids'], image_size=self._bbox_size, alpha=0.5)
        # since clusters will be on the border if there are many connected components, move everything in by a percentage
        marg_frac = 0.025
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        w, h = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
        ax.set_xlim(x_lim[0] - marg_frac*w, x_lim[1] + marg_frac*w)
        ax.set_ylim(y_lim[0] - marg_frac*h, y_lim[1] + marg_frac*h)
        ax.set_title("Random Projection")
        self._disp_img = self._plotter.render_fig(fig)

    def clear(self):
        super().clear()
        self._axes = None
        self._noise_offsets = None
        self.tools['noise_slider'].set_visible(True)
        self.tools['project_button'].set_visible(True)


WINDOW_TYPES = {Windows.ui: UiWindow,
                Windows.toolbar: ToolsWindow,
                Windows.clustering: ClustersWindow,
                Windows.spectrum: SpectrumWindow,
                Windows.eigenvectors: EigenvectorsWindow,
                Windows.sim_matrix: SimMatrixWindow,
                Windows.rand_proj: RandProjWindow}
