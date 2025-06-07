r"""
Slope diagrams show the relative change of a set of real values from epoch t to t+1.

    Let S be the set of N states, 
        V(s) be the value of the N variables at epoch t,
        W(s) be the value at epoch t+1,
        CV(r) be the counts of the number of states in S st. V(s) = r, 
        CW(r) be the counts for W,
        P be the set of unique pairs {x,y : V(s_j)=x, W(s_j)=y for at least one j}, and
        PC be the counts of each pair, PC_j in [1, N] for each pair P_j.

      * Show two vertical lines representing the span of V and W, one on the left and one on the right.
      * For each unique value x in CV, 
          * plot a tick on the left line, 
          * color-mapped by the value, 
          * size proportional to CV(x).

      * Show the same on the right side for CW.
      * Each unique pair (v_j, w_j) in P is plotted as a line with thickness and/or transparency proportional to C_j.
      
      
    +-------------------------------------+
    |                                     |  
    |     Title:  Values from T=3 to 4    |
    |                                     |  
    |         +       1.0        +        |
    |         |    /-------------|-       |
    |        -|\--/--------------|-       |
    |      ---|.\/....0.0........|--      |
    |         | /\               |        |
    |    -----|/  \              |        |
    |         | \--\-------------|-----   |
    |         +      -1.0        +        |
    |                                     | 
    +-------------------------------------+
    
    Three unique values on the left, with counts, 1, 3, and 5 (by the vertical histogram's lines), split
    into four unique values on the right with counts 1, 2, 3 and 5.
    Lines show large shift donwards, consolidating around -1.0, and a smaller shift upwards, 
    splitting into mulitiple values closer to 1.0  (Using sloped lines instead of ASCII art)

    NOTE:  Both axes must be on the same scale (same endpoints) or the slopes will be arbitrary.

  
"""
import tkinter as tk
from util import tk_color_from_rgb
from PIL import Image, ImageTk
import logging
import numpy as np
import cv2
from colors import COLOR_SCHEME
from gui_base import get_font_scale
from loop_timing.loop_profiler import LoopPerfTimer as LPT
# TODO:  add side-histograms
REL_DIMS = {'hist_width_frac': 0.175,  # fraction of bounding box width to use for histogram width
            'padding_frac': (0.05, 0.10),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'spread': 0.8,  # lines are this far apart in their available horizontal space.
            'line_alpha_range': (0.1, 1.0),
            'line_thickness_range': (1, 15),  # Sloped lines
            'min_hist_bar_thickness': 2,
            'max_hist_bar_thickness_frac': .03,  # thickness of the histogram bars, fraction of image height
            'axis': {'thickness': 3,
                     'tick_len': 3,
                     'margin': 0.1},  # reserve this much on the top & bottom of each axis (before points)
            'lable_spacing_frac': 0.05,  # fraction of text height between axis ends and end labels.
            # fraction of bounding box height to use for title height
            'text_height_frac': {'title': 0.1},

            }


class SlopeDiagram(object):
    _SHIFT_BITS = 0
    _SHIFT_MUL = 1 << _SHIFT_BITS

    def __init__(self, size, old_vals, new_vals, title=None):
        self._v0 = old_vals
        self.size = size
        self._v1 = new_vals
        self._title = title
        self._colors = COLOR_SCHEME

        pair_arr = np.array(list(zip(self._v0, self._v1)))
        # Draw one line per unique pair
        self._distinct_pairs, self._pair_counts = np.unique(pair_arr, axis=0, return_counts=True)
        # Draw one line on the histogram per unique value on each side.
        self._left_vals, self._left_counts = np.unique(self._v0, return_counts=True)
        self._right_vals, self._right_counts = np.unique(self._v1, return_counts=True)

    @LPT.time_function
    def draw(self, img, bbox=None):
        bbox = bbox if bbox is not None else {'x': (0, img.shape[1]), 'y': (0, img.shape[0])}
        dims = self._calc_dims(bbox)

        def draw_bbox(bbox, color):
            x0, x1 = bbox['x']
            y0, y1 = bbox['y']
            img[y0, x0:x1] = color
            img[y1, x0:x1] = color
            img[y0:y1, x0] = color
            img[y0:y1, x1] = color

        # draw the bounding box
        # draw_bbox(dims['inner_bbox'], self._colors['lines'])

        # axis lines
        axis_y = dims['axis']['y']
        t = dims['axis']['axis_thickness']
        tick_len = dims['axis']['tick_len']
        max_line_thickness = REL_DIMS['line_thickness_range'][1]

        slope_bbox = dims['inner_bbox']

        def draw_axis(x, color, thick):

            if thick > 0:
                x0, x1 = x, x+thick
                y_thick = thick
                # Set area next to axis to the BG color
                y_upper = axis_y[0] + y_thick
                y_lower = axis_y[1] - y_thick
                img[y_upper:y_lower, max(0, x0-max_line_thickness):x0] = self._colors['bg']
            else:
                # Right axis
                x0, x1 = x+thick, x
                y_thick = -thick
                y_upper = axis_y[0] + y_thick
                y_lower = axis_y[1] - y_thick
                img[y_upper:y_lower, x1:min(x1+max_line_thickness,slope_bbox['x'][1])] =self._colors['bg']

            # Vertical axis line:
            img[axis_y[0]:axis_y[1], x0:x1] = color
            # draw the upper and lower ends:
            x_t_left = x0 - tick_len
            x_t_right = x1 + tick_len
            img[axis_y[0]:axis_y[0]+y_thick, x_t_left:x_t_right] = color  # top tick
            img[axis_y[1]-y_thick:axis_y[1], x_t_left:x_t_right] = color  # bottom tick

        # axis labels
        axis_label_font_scale = dims['font']
        for line, pos in dims['axis']['labels']:
            cv2.putText(img, line, (pos[0], pos[1]), dims['font'],
                        axis_label_font_scale, self._colors['text'], 1, cv2.LINE_AA)

        # histograms
        def draw_histogram(hist_lines, bbox):
            """
            Draw the histogram lines in the given bounding box.
            :param hist_lines:  list of line dicts, each with keys: p0, p1, thickness, alpha
            :param bbox:  bounding box for the histogram
            """
            for line in hist_lines:
                p0 = int(line['p0'][0]*self._SHIFT_MUL), int(line['p0'][1]*self._SHIFT_MUL)
                p1 = int(line['p1'][0]*self._SHIFT_MUL), int(line['p1'][1]*self._SHIFT_MUL)
                thickness = line['thickness']
                alpha = line['alpha']
                color = (np.array((255, 255, 255))*(1.0-alpha)).astype(np.uint8).tolist()
                cv2.line(img, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        # slopes
        for line in dims['lines']:

            p0 = int(line['p0'][0]*self._SHIFT_MUL), int(line['p0'][1]*self._SHIFT_MUL)
            p1 = int(line['p1'][0]*self._SHIFT_MUL), int(line['p1'][1]*self._SHIFT_MUL)

            thickness = line['thickness']
            alpha = line['alpha']
            color = (np.array((255, 255, 255))*(1.0-alpha)).astype(np.uint8).tolist()
            cv2.line(img, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        # Draw axis at the end to cover up CV2 messy line ends.
        axis_left = dims['axis']['x_left']
        axis_right = dims['axis']['x_right']
        draw_bbox(dims['l_hist']['bbox'], self._colors['lines'])
        draw_bbox(dims['r_hist']['bbox'], self._colors['lines'])

        draw_axis(axis_left, self._colors['lines'], t)
        draw_axis(axis_right, self._colors['lines'], -t)
        if dims['l_hist']['bbox'] is not None:
            # Draw the left histogram
            draw_histogram(dims['l_hist']['lines'], dims['l_hist']['bbox'])
        if dims['r_hist']['bbox'] is not None:
            # Draw the right histogram
            draw_histogram(dims['r_hist']['lines'], dims['r_hist']['bbox'])




        # title
        if self._title is not None:
            title_x, title_y = dims['title']['x'], dims['title']['y']
            cv2.putText(img, dims['title']['text'], (title_x, title_y), dims['font'],
                        dims['title']['font_scale'], self._colors['text'], 1, cv2.LINE_AA)



        return img

    @LPT.time_function
    def _calc_dims(self, bbox):
        """
        Find absolute dimensions & layout locations for everything, return as dict.
        """
        w, h = bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]
        padding_x = int(w * REL_DIMS['padding_frac'][0])
        padding_y = int(h * REL_DIMS['padding_frac'][1])
        x_left, x_right = bbox['x'][0] + padding_x, bbox['x'][1] - padding_x
        y_top, y_bottom = bbox['y'][0] + padding_y, bbox['y'][1] - padding_y

        x_w = x_right - x_left
        x_center = x_left + x_w // 2
        text_padding = int(h * REL_DIMS['lable_spacing_frac'])

        dims = {'bbox': bbox,
                'inner_bbox': {'x': (x_left, x_right), 'y': (y_top, y_bottom)},
                'x_center': x_center,
                'title': None,
                'font': REL_DIMS['font'],
                'axis': {}, }
        # 'l_hist': {'bbox': h_bbox_left, 'lines': l_hist_lines},
        # 'r_hist': {'bbox': h_bbox_right, 'lines': r_hist_lines}

        if self._title is not None:
            title_h = int(h * REL_DIMS['text_height_frac']['title'])
            font_scale = _gfs_stub(REL_DIMS['font'],
                                   max_height=title_h,
                                   max_width=x_w,
                                   text_lines=[self._title])
            (t_width, t_height), t_baseline = cv2.getTextSize(self._title, REL_DIMS['font'], font_scale, 1)
            title_y = y_top + t_height
            title_x = x_center - t_width // 2
            dims['title'] = {'x': title_x,
                             'y': title_y,
                             'font_scale': font_scale,
                             'baseline': t_baseline,
                             'text': self._title}
            y_top += t_height + t_baseline + text_padding
        else:
            font_scale = 1.0

        # Need to know how big these are for where the axes go.
        hist_width = int(w * REL_DIMS['hist_width_frac'])
        if hist_width == 0:
            h_bbox_left, h_bbox_right = None, None
        else:
            h_bbox_left = {'x': (x_left, x_left + hist_width),
                           'y': (y_top, y_bottom)}
            h_bbox_right = {'x': (x_right - hist_width, x_right),
                            'y': (y_top, y_bottom)}
            x_left += hist_width
            x_right -= hist_width

        # Now we can place the axes.
        axis_y = (y_top, y_bottom)
        axis_tick_len = REL_DIMS['axis']['tick_len']
        axis_dist = int((x_right-x_left) * REL_DIMS['spread'])
        ax_thick = REL_DIMS['axis']['thickness']

        dims['axis'] = {'x_left': x_left,
                        'x_right': x_right,
                        'y': axis_y,
                        'axis_thickness':  ax_thick,
                        'tick_len': axis_tick_len,
                        'font': REL_DIMS['font'],
                        'labels': [],
                        'bbox': {'x': (x_left, x_right), 'y': axis_y},
                        'font_scale': 3}

        # Calculate where the values go on both axes.
        val_scale = self._get_scaling()
        y_height = axis_y[1] - axis_y[0]

        lines = []
        line_scale = self._scale_counts(self._pair_counts)
        for ind, (v0, v1) in enumerate(self._distinct_pairs):
            thickness, alpha = line_scale['t'][ind], line_scale['alpha'][ind]
            y_span = (axis_y[0], axis_y[0]+y_height)
            y_left, y_right = (self._scale_y_value(v0, y_span=y_span, value_span=val_scale),
                               self._scale_y_value(v1, y_span=y_span, value_span=val_scale))

            lines.append({'p0': (x_left+int(ax_thick)-axis_tick_len, y_left),
                          'p1': (x_right-int(ax_thick)+axis_tick_len, y_right),
                          'thickness': thickness,
                          'alpha': alpha})
        dims['lines'] = lines

        # Now we can draw the histograms at the same y positions as the lines.

        # Place histograms on the sides.

        def make_hist(values, counts, bbox, backwards=False):
            """
            Create a histogram for the given values, in the given x range, with the given y span.
            Can extend to the left  (xlow > x_high)  or right (xlow < x_high) 
            :param values:  list of values (y-positions)
            :param counts:  list of counts for each value
            :param px_low:  x position of the small-value edge of the histogram
            :param px_high: x position of the large-value edge of the histogram (can be left of px_low)
            :param y_span:  (y_min, y_max)  y positions for the histogram
            :returns:  list of line dicts, each with keys: p0, p1, thickness, color, alpha
            """
            
            x_min, x_max = bbox['x'] 
            x_span = x_max - x_min
            bar_norm = counts / np.max(counts)
            if not backwards:
                x_left = x_min
                x_right = x_left + x_span * bar_norm
            else:
                x_right = x_max
                x_left = x_right - x_span * bar_norm

            y_positions = self._scale_y_value(values, val_scale, y_span=(bbox['y'][0], bbox['y'][1]))
            lines = []
            thicknesss, alphas = self._scale_counts(counts).values()
            for (y_pos, bar_norm_len, thick, alpha) in zip(y_positions, bar_norm, thicknesss, alphas):
                if backwards:
                    bar_right = x_right
                    bar_left = x_right - x_span * bar_norm_len
                else:
                    bar_left = x_left
                    bar_right = x_left + x_span * bar_norm_len

                lines.append({'p0': (bar_left, y_pos),
                              'p1': (bar_right, y_pos),
                              'thickness': thick,
                              'alpha': alpha})

            return lines
        
        l_hist_lines = make_hist(self._left_vals, self._left_counts, h_bbox_left, backwards=True)
        r_hist_lines = make_hist(self._right_vals, self._right_counts, h_bbox_right)

        dims['l_hist'] = {'bbox': h_bbox_left, 'lines': l_hist_lines}
        dims['r_hist'] = {'bbox': h_bbox_right, 'lines': r_hist_lines}

        return dims

    def _scale_counts(self, all_counts):
        ltr = REL_DIMS['line_thickness_range']
        lar = REL_DIMS['line_alpha_range']
        rel_counts = all_counts / np.max(all_counts)
        # Thicknesses:
        t_scaled_counts = (rel_counts * (ltr[1] - ltr[0]) + ltr[0]).astype(int)
        t_scaled_counts = np.clip(t_scaled_counts, ltr[0], ltr[1])
        # Alphas:
        a_scaled_counts = rel_counts * (lar[1] - lar[0]) + lar[0]
        a_scaled_counts = np.clip(a_scaled_counts, lar[0], lar[1])
        alphas = a_scaled_counts*0 + 0.5

        return {'t': t_scaled_counts.astype(int),
                'alpha': alphas}

    def _scale_y_value(self, value, value_span, y_span):
        val_norm = (value - value_span[0]) / (value_span[1] - value_span[0])

        # flip the value to go from top to bottom.

        # return int(y_span[1] - val_norm * (y_span[1] - y_span[0]))
        return (y_span[0] + val_norm * (y_span[1] - y_span[0]))

    def _get_scaling(self):
        """
        Determine the vertical positions for values on both sides.

        """

        def _get_scale(vals):
            """
            Get the max & min, accounting for the margin.
            """
            if len(vals) == 0:
                min_val, max_val = -1.0, 1.0
            min_val, max_val = np.min(vals), np.max(vals)
            if min_val == max_val:
                # all values are the same, so just use the axis height.
                margin = 1.0
            else:
                # scale to fit in the axis height, with a margin on top & bottom.
                diff = max_val - min_val
                margin = diff * REL_DIMS['axis']['margin']
                min_val -= margin
                max_val += margin
            return min_val, max_val

        left_scale = _get_scale(self._v0)
        right_scale = _get_scale(self._v1)

        # Common scale for both sides, to show the real slope.
        scale = np.min((left_scale[0], right_scale[0])), np.max((left_scale[1], right_scale[1]))

        return scale


@LPT.time_function
def _gfs_stub(*args, **kwargs):
    """
    Stub for get_font_scale, to avoid import errors in the test script.
    """
    return get_font_scale(*args, **kwargs)


class DiagramSizeTester(object):
    """
    Tk window with single image lable (the diag), to show resize capability.
    """

    def __init__(self, size, diag_factory):
        diag = diag_factory(size)
        self._img_size = diag.size
        self._make_diag = diag_factory
        self._init_tk()

    def _init_tk(self):

        self._root = tk.Tk()
        self._root.geometry(f"{self._img_size[0]}x{self._img_size[1]}")
        self._root.title("State Embedding diagram Tester")

        self._frame = tk.Frame(self._root, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._frame.pack(fill=tk.BOTH, expand=True)
        self._label = tk.Label(self._frame, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
        self._label.pack(fill=tk.BOTH, expand=True)
        self._label.bind("<Configure>", self._on_resize)

    def start(self):
        """
        Start the Tk main loop.
        """
        if False:
            # for timing the draw function
            LPT.reset(enable=True, burn_in=5, display_after=30)
            for iter in range(100):
                LPT.mark_loop_start()
                _ = self._get_img()

        self._root.mainloop()

    def _on_resize(self, event):
        print(f"Resize event: {event.width}x{event.height}")
        self._img_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_diag_image()

    def _get_img(self):
        diag = self._make_diag(self._img_size)
        img = np.zeros((self._img_size[1], self._img_size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']
        diag.draw(img)
        return img

    def refresh_diag_image(self):
        print(f"Refreshing diagram image to size {self._img_size}")
        img = self._get_img()
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self._label.config(image=img)
        self._label.image = img


def test_diag():
    vals1 = np.concatenate((np.random.randn(100), np.random.randn(100)*.5 + 2.0))
    vals2 = np.concatenate((np.random.randn(50)*.7-.5, np.random.randn(150)*1.0 + 1.0))
    vals1 = np.zeros(10)
    vals1[-1] = 1
    vals2 = np.array([-1.0]*1 + [0.0]*1 + [1.0]*2 + [2.0]*5 + [-3.0]*1)
    print("vals1", vals1)
    print("vals2", vals2)

    def _make_diag(size):
        diag = SlopeDiagram(size, vals1, vals2, title="Values from T=0 -> T=1")
        return diag

    size = (300, 220)

    tester = DiagramSizeTester(size, _make_diag)
    tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_diag()
