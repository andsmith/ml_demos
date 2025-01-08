"""
Handle sub-windows for the cluster creator app.
"""
import numpy as np
from abc import ABC, abstractmethod
import cv2
import logging

COLORS = {'white': np.array((255, 255, 255)).astype(np.uint8),
          'gray': np.array((128, 128, 128)).astype(np.uint8),
          'black':  np.array((0, 0, 0)).astype(np.uint8),
          'red': np.array((0, 0, 255)).astype(np.uint8),
          'green':  np.array((0, 255, 0)).astype(np.uint8),
          'blue':  np.array((255, 0, 0)).astype(np.uint8),
          'yellow':  np.array((0, 255, 255)).astype(np.uint8),
          'magenta':  np.array((255, 0, 255)).astype(np.uint8),
          'cyan':  np.array((255, 255, 0)).astype(np.uint8)}

LAYOUT = {"windows": {'ui': {'x': (0, .666),  # scale from unit square to window size
                             'y': (0, .666)},
                      'tools': {'x': (0, .666),
                                'y': (.667, 1)},
                      'eigenvectors': {'x': (.667, 1),
                                   'y': (.333, .667)},
                      'clusters': {'x': (.667, 1),
                                   'y': ( .667,1)},
                      'spectrum': {'x': ( .667,1),
                                       'y': (0, .333)}},
          'colors': {'bkg': COLORS['black'],
                     'border': COLORS['gray'],
                     'active_border': COLORS['white'],
                     'font': COLORS['white']},

          'dims': {'margin_px': 5,
                   'pt_size': 2,
                   'mouseover_rad_px': 20},
          'font': cv2.FONT_HERSHEY_SIMPLEX,
          'font_size': .9,
          'font_color': (0, 0, 0),
          'font_thickness': 1}


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

    def render(self,img, active=False):
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

    def __init__(self, name, win_size):
        super().__init__(name, win_size)

    def render(self, img, active=False):
        super().render(img,active = active)

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
    pass


class SpectrumWindow(UiWindow):
    pass


class ClustersWindow(UiWindow):
    pass


class EigenvectorsWindow(UiWindow):
    pass
