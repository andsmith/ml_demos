"""
Handle sub-windows for the cluster creator app.
"""
import numpy as np
from abc import ABC, abstractmethod
import cv2

COLORS = {'white': (255, 255, 255),
          'black': (0, 0, 0),
          'red': (0, 0, 255),
          'green': (0, 255, 0),
          'blue': (255, 0, 0),
          'yellow': (0, 255, 255),
          'magenta': (255, 0, 255),
          'cyan': (255, 255, 0)}

LAYOUT = {"windows": {'ui': {'x': (0, 0.667),  # scale from unit square to window size
                             'y': (0, 0.667)},
                      'tools': {'x': (0.667, 1),
                                'y': (0, .333)},
                      'spectrum': {'x': (0, 1),
                                   'y': (.667, 1)},
                      'clusters': {'x': (0, 1),
                                   'y': (.333, .667)},
                      'eigenvectors': {'x': (0, .333),
                                       'y': (.667, 1)}},
          'dims': {'margin': 0.0025,
                   'pt_size': 2,
                   'mouseover_rad_px': 20},
          'font': cv2.FONT_HERSHEY_SIMPLEX,
          'font_size': 1,
          'font_color': (0, 0, 0),
          'font_thickness': 2}


def _draw_border(img, bbox, color=COLORS['white'], margin_px=5):
    """
    Draw a border around a bounding box.
    """
    x0, y0, x1, y1 = bbox
    img[y0:y0+margin_px, x0:x1] = color
    img[y1-margin_px:y1, x0:x1] = color
    img[y0:y1, x0:x0+margin_px] = color
    img[y0:y1, x1-margin_px:x1] = color


class Window(ABC):
    """
    Abstract class for windows in the cluster creator app.
    Each instance reprseents a sub-window, is responsible for rendering itself onto the main window,
    and handling user input.  Instances will only get mouse/keyboard events when the mouse is within
    the window's bounding box.
    """

    def __init__(self, name, win_size):
        self.name = name
        self.win_size = win_size
        self.bbox = self._get_bbox()

        if name not in LAYOUT['windows']:
            raise ValueError(f"Invalid window name: {name}")

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

        return {(x0, y0, x1, y1)}

    def render_title(self, img):
        """
        Render the window title.
        """
        x0, y0, x1, y1 = self.bbox
        indent_px = 10
        pos = (x0+indent_px, y0+indent_px)
        cv2.putText(img, self.name, pos, LAYOUT['font'], LAYOUT['font_size'],
                    LAYOUT['font_color'], LAYOUT['font_thickness'])

    def render(self, img):
        """
        Render the window onto the image.
        (override for specific window types)
        """
        x0, y0, x1, y1 = self.bbox

        _draw_border(img, self.bbox)
        # write window name in top left corner
        self.render_title(img)

    @abstractmethod
    def keypress(self, key):
        """
        Handle a keypress event.
        """
        pass

    @abstractmethod
    def mouse_event(self, event, x, y, flags, param):
        """
        Handle a mouse event.
        """
        pass


class UiWindow(Window):
    """
    Window for the cluster creator UI.
    """

    def __init__(self, name, win_size, bbox):
        super().__init__(name, win_size, bbox)

    def render(self, img):
        super().render(img)

    def keypress(self, key):
        pass

    def mouse_event(self, event, x, y, flags, param):
        pass


class ToolsWindow(UiWindow):
    pass


class SpectrumWindow(UiWindow):
    pass


class ClustersWindow(UiWindow):
    pass


class EigenvectorsWindow(UiWindow):
