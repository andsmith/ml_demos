"""
Class to plot eigenvalues/vectors in a bounding box with various options.
Should have a method for adjusting plot parameters (n vecs, etc.)  in realtime.
(e.g. with a callback that can be connected to a slider, log/linear button, etc.)
"""
import numpy as np
import cv2
from colors import COLORS
from layout import PLOT_LAYOUT, WINDOW_LAYOUT
from util import calc_font_size


class SpectrumPlot(object):
    def __init__(self, app, bbox, label, max_plot=None, visible=True, title_v_frac=0.2):
        """
        Plot nonnegative values in a bbox.
        :param app: ClusterCreator application object
        :param bbox: bounding box in which to plot, dict(x=(left, right), y=(top, bottom))
        :param label: label for the plot
        :param max_plot: maximum number of values to plot
        :param visible: whether to display the plot 
        """
        self._bbox = bbox
        self._label = label
        self._max_plot = max_plot
        self._visible = visible
        self._disp_img = None
        self._bkg_color = + WINDOW_LAYOUT['colors']['bkg']
        self._app = app
        w = bbox['x'][1] - bbox['x'][0]
        h = bbox['y'][1] - bbox['y'][0]
        self._blank = np.zeros((h, w, 3), np.uint8) + self._bkg_color  # only redraw after updates
        print("Initialized spectrum plot with size:  %s x %s" % (w, h))
        self._values = None

        # plot layout parameters
        self._font_color = PLOT_LAYOUT['title_color'].tolist()
        self._axis_color = PLOT_LAYOUT['axis_color'].tolist()
        self._tick_color = PLOT_LAYOUT['tick_color'].tolist()
        self._spacing_px = PLOT_LAYOUT['axis_spacing']
        self._title_v_frac = title_v_frac
        self._calc_dims()

    def _calc_dims(self):
        """
        Calculate dimensions of the plot that don't change with different values.
        The abstract version will calculate the x & y axis endpoints, the tickmarks endpoints,
        the tickmark texts, the title text & position.

        All graphics should be  within self._spacing_px pixels inside the edge of self.bbox.
        """
        h, w, _ = self._blank.shape
        axis_indent = self._spacing_px*3  # how much space is for the negative quadrants
        title_v_indent = self._spacing_px*2  # how much space is for the title

        # Since we're drawing on a blank image, span the whole thing minus the indents
        self._x_axis = [(self._spacing_px, h-axis_indent), (w-self._spacing_px, h-axis_indent)]
        self._y_axis = [(axis_indent, self._spacing_px), (axis_indent, h-self._spacing_px)]
        print("X axis: %s" % (self._x_axis,))
        print("Y axis: %s" % (self._y_axis,))
        title_v_height = int(h * self._title_v_frac)

        self._fontsize, _ = calc_font_size(self._label,
                                           {'x': (self._spacing_px, w-self._spacing_px),
                                            'y': (self._spacing_px, title_v_height - self._spacing_px)},
                                           cv2.FONT_HERSHEY_SIMPLEX, 0, 0)
        (title_w, title_h), _ = cv2.getTextSize(self._label, cv2.FONT_HERSHEY_SIMPLEX, self._fontsize, 1)
        self._pos_quad_bbox = {'x': (axis_indent, w-self._spacing_px),
                                'y': (self._spacing_px, h-axis_indent)}

        self._title_pos = ((w - title_w) // 2, title_v_indent + title_h)
        print("Font and position: %s, %s" % (self._fontsize, self._title_pos))

    def _calc_scale(self):
        """
        """

    def set_values(self, values):
        """
        Update the plot with new values.
        :param values: 1D numpy array of values to plot
        """
        self._values = values
        self._refresh()

    def _refresh(self):
        img = self._blank.copy()
        print(img.shape)
        self._draw_axes(img)
        self._draw_title(img)
        self._draw_values(img)
        self._disp_img = img

        print(img.shape)

    def _draw_axes(self, img):
        cv2.line(img, self._x_axis[0], self._x_axis[1], self._axis_color, 1, cv2.LINE_AA)
        cv2.line(img, self._y_axis[0], self._y_axis[1], self._axis_color, 1, cv2.LINE_AA)
        # draw rect in pos quad
        #cv2.rectangle(img, (self._pos_quad_bbox['x'][0], self._pos_quad_bbox['y'][0]),
        #              (self._pos_quad_bbox['x'][1], self._pos_quad_bbox['y'][1]), self._axis_color,-1)  

    def _draw_title(self, img):

        cv2.putText(img, self._label, self._title_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, self._fontsize, self._font_color, 1)

    def _draw_values(self, img):
        pass

    def clear(self):
        self._values = None

    def render(self, frame):
        if self._visible:
            if self._disp_img is None:
                print("How did this happen?")
                self._refresh()
            frame[self._bbox['y'][0]:self._bbox['y'][1], self._bbox['x'][0]:self._bbox['x'][1]] = self._disp_img
        return frame

    def toggle_visibility(self):
        self._visible = not self._visible
        if self._visible:
            self._refresh()
        print("Plot visibility for %s: %s" % (self._label, self._visible))
