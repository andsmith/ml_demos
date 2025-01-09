"""
Handle sub-windows for the cluster creator app.
"""
import numpy as np
from abc import ABC, abstractmethod
import cv2
from util import scale_bbox
import logging
from tools import RadioButtons, Slider, Button
from clustering import render_clustering, KMeansAlgorithm
from colors import COLORS
from layout import LAYOUT, TOOLBAR_LAYOUT


class Window(ABC):
    """
    Abstract class for windows in the cluster creator app.
    Each instance reprseents a sub-window, is responsible for rendering itself onto the main window,
    and handling user input.  Instances will only get mouse/keyboard events when the mouse is within
    the window's bounding box.
    """

    def __init__(self, name, win_size, app):
        self.name = name
        self.app = app
        self.win_size = win_size
        self.margin_px = int(LAYOUT['dims']['margin_px'])
        self.bbox = self._get_bbox()
        self.colors = LAYOUT['colors']

        self._title_height = cv2.getTextSize(self.name, LAYOUT['font'], LAYOUT['font_size'],
                                             LAYOUT['font_thickness'])[0][1]

        if name not in LAYOUT['windows']:
            raise ValueError(f"Invalid window name: {name}")
        logging.info(f"Created window {name} with bbox {self.bbox}")

    def _get_bbox(self):
        """
        Transform the relative layout dims to actual pixel coords.
        """
        x0, x1 = LAYOUT['windows'][self.name]['x']
        y0, y1 = LAYOUT['windows'][self.name]['y']

        x0 = int(x0 * self.win_size[0])
        x1 = int(x1 * self.win_size[0])
        y0 = int(y0 * self.win_size[1])
        y1 = int(y1 * self.win_size[1])

        return {'x': (x0, x1), 'y': (y0, y1)}

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

        cv2.putText(img, self.name, pos, LAYOUT['font'], LAYOUT['font_size'],
                    self.colors['font'].tolist(), LAYOUT['font_thickness'],)

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

    @abstractmethod
    def keypress(self, key):
        """
        Handle a keypress event.
        """
        pass

    @abstractmethod
    def mouse_click(self, x, y):
        """
        Handle a mouse click event.
        :returns: True if future mouse events should be sent to this window until unclick is called
        """
        pass

    @abstractmethod
    def mouse_unclick(self, x, y):
        """
        Handle a mouse unclick event.
        """
        pass

    @abstractmethod
    def mouse_move(self, x, y):
        """
        Handle a mouse move event.
        """
        pass


class UiWindow(Window):
    """
    Window for the cluster creator UI.
    """

    def __init__(self, name, win_size, app):
        super().__init__(name, win_size, app)
        self._clusters = []


    def get_points(self, n_per_cluster):
        """
        Each cluster has a weight in [0, 100].

        points = []
        for cluster in self._clusters:
            points.append(cluster.generate_points(n_per_cluster))
        """
        points = []
        # for now, random
        for _ in range(2):  # three test clusters
            c_center = np.random.rand(2)
            c_points = np.random.randn(n_per_cluster, 2) * .1 + c_center
            points.append(c_points)
        return np.vstack(points)

    def render(self, img, active=False):
        super().render(img, active=active)

    def keypress(self, key):
        pass

    def mouse_click(self, x, y):
        print(f'{self.name} clicked at ({x}, {y})')
        return True

    def mouse_unclick(self, x, y):
        print(f'{self.name} unclicked at ({x}, {y})')

    def mouse_move(self, x, y):
        print(f'{self.name} mouse moved to ({x}, {y})')


class ToolsWindow(UiWindow):
    """
    Create & manage toolbox for the app.
    """

    def __init__(self, name, win_size, app):
        super().__init__(name, win_size, app)
        self._setup_tools()
        self._held_tool = None

    def _setup_tools(self):
        indent_px = 5
        indented_bbox = {'x': (self.bbox['x'][0] + indent_px, self.bbox['x'][1] - indent_px),
                         'y': (self.bbox['y'][0] + indent_px, self.bbox['y'][1] - indent_px)}
        self.tools = {'kind_radio': RadioButtons(scale_bbox(TOOLBAR_LAYOUT['kind_radio'], indented_bbox),
                                                 'Cluster Type:',
                                                 ['gauss', 'ellipse', 'annulus'],
                                                 default_selection=0),
                      'alg_radio': RadioButtons(scale_bbox(TOOLBAR_LAYOUT['alg_radio'], indented_bbox),
                                                 'Algorithm:',
                                                 ['Unnormalized', 'Normalized','K-means'],
                                                 default_selection=2),
                      'n_nearest_slider': Slider(scale_bbox(TOOLBAR_LAYOUT['n_nearest_slider'], indented_bbox),
                                                 'N-Nearest:',
                                                 [3, 20], 5, format_str="=%i"),
                      'n_pts_slider': Slider(scale_bbox(TOOLBAR_LAYOUT['n_pts_slider'], indented_bbox),
                                             'Num Pts',
                                             [5, 2000], 100, format_str="=%i"),
                      'k_slider': Slider(scale_bbox(TOOLBAR_LAYOUT['k_slider'], indented_bbox),
                                             'K (clusters)',
                                             [2, 25], 5, format_str="=%i"),
                      'run_button': Button(scale_bbox(TOOLBAR_LAYOUT['run_button'], indented_bbox), 'Run',callback= self.app.recompute),
                      'clear_button': Button(scale_bbox(TOOLBAR_LAYOUT['clear_button'], indented_bbox), 'Clear', self.app.clear)}
        logging.info(f"Created tools window with {len(self.tools)} tools")

    def render(self, img, active=False):
        #super().render(img, active=active)
        w, h = self.bbox['x'][1] - self.bbox['x'][0], self.bbox['y'][1] - self.bbox['y'][0]
        #cv2.rectangle(img, (self.margin_px, self.margin_px),
        ##              (w - self.margin_px, h - self.margin_px),
        #              LAYOUT['colors']['border'].tolist(), 2)
        for tool in self.tools.values():
            tool.render(img)

    def keypress(self, key):
        pass

    def mouse_click(self, x, y):
        """
        Send to all tools, each will handle if it was clicked.
        Return whatever tool uses the click.
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

    def get_value(self, param):
        if param == 'kind':
            return self.tools['kind_radio'].get_value()
        elif param == 'algorithm':
            return self.tools['alg_radio'].get_value()
        elif param == 'n_nearest':
            return int(self.tools['n_nearest_slider'].get_value())
        elif param == 'n_pts':
            return int(self.tools['n_pts_slider'].get_value())
        elif param == 'k':
            return int(self.tools['k_slider'].get_value())
        else:
            raise ValueError(f"Invalid parameter: {param}")


class SpectrumWindow(UiWindow):
    pass


class ClustersWindow(UiWindow):
    def __init__(self, name, win_size, app):
        super().__init__(name, win_size, app)
        w, h = self.bbox['x'][1] - self.bbox['x'][0], self.bbox['y'][1] - self.bbox['y'][0]
        self._bkg_color = + LAYOUT['colors']['bkg']
        self._blank = np.zeros((h, w, 3), np.uint8) +self._bkg_color  # only redraw after updates
        cv2.rectangle(self._blank, (self.margin_px, self.margin_px),
                      (w - self.margin_px, h - self.margin_px),
                      LAYOUT['colors']['border'].tolist(), 2)
        self._frame = self._blank.copy()

    def update(self, points, cluster_ids, colors):
        """
        Update the cluster window with new points and cluster assignments.
        """
        self._frame = self._blank.copy()
        render_clustering(self._frame, points, cluster_ids, colors, margin_px=self.margin_px)
        print("Regenerated cluster window")

    def render(self, img, active=False):
        img[self.bbox['y'][0]:self.bbox['y'][1],
            self.bbox['x'][0]:self.bbox['x'][1]] = self._frame


class EigenvectorsWindow(UiWindow):
    pass
