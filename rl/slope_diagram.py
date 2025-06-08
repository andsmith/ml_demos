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
from plot_util import draw_alpha_line, scale_y_value, scale_counts, calc_alpha_adjust
from micro_histogram import MicroHist

from loop_timing.loop_profiler import LoopPerfTimer as LPT
# TODO:  move to LAYOUT
REL_DIMS = {'hist_width_frac': 0.2,  # fraction of bounding box width to use for histogram width
            'padding_frac': (0.05, 0.08),
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'spread': 0.8,  # lines are this far apart in their available horizontal space.
            'line_alpha_range': (0.025, 0.3),
            'line_thickness_range_rel': (2, .15),  # min ing pixels, max is relative to height

            'ticks': {'alpha': 0.5, 
                      'w': 2},

            'axis': {'thickness': 2,
                     'tick_len': 2,
                     'margin': 0.1},  # reserve this much on the top & bottom of each axis (before points)
            'title_spacing': 1.15,  # fraction of text height between axis ends and end labels.
            # fraction of bounding box height to use for title height
            'text_height_frac': {'title': 0.1},
            'max_font_scale': 1.2  # maximum font scale for text

            }


class SlopeDiagram(object):
    _SHIFT_BITS = 0
    _SHIFT_MUL = 1 << _SHIFT_BITS

    def __init__(self, size, old_vals, new_vals, title=None, fast=False):
        """
        Initialize a SlopeDiagram object with the given values and title.
        :param size:  (width, height) of the diagram in pixels.
        :param old_vals:  list of values at epoch t.
        :param new_vals:  list of values at epoch t+1.
        :param title:  Title of the diagram, displayed at the top.
        :param fast:  draw grayscale lines instead of shaded bars.  TODO:  Implement (as subclass?).
        """
        self._v0 = old_vals
        self.size = size
        self._v1 = new_vals
        self._title = title
        self._colors = COLOR_SCHEME
        self._fast = fast

        pair_arr = np.array(list(zip(self._v0, self._v1)))
        # Draw one line per unique pair
        self._distinct_pairs, self._pair_counts = np.unique(pair_arr, axis=0, return_counts=True)
        # Draw one line on the histogram per unique value on each side.
        self._left_vals, self._left_counts = np.unique(self._v0, return_counts=True)
        self._right_vals, self._right_counts = np.unique(self._v1, return_counts=True)


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
        x_h = y_bottom - y_top
        x_center = x_left + x_w // 2
        bbox_inner = {'x': (x_left, x_right), 'y': (y_top, y_bottom)}
        fig_bbox = bbox_inner

        ltr_rel = REL_DIMS['line_thickness_range_rel']
        line_thickness_range = (ltr_rel[0],
                                max(ltr_rel[0], int(x_h * ltr_rel[1])))
        med_thick = (line_thickness_range[0] + line_thickness_range[1]) // 2

        dims = {'bbox': bbox,
                'inner_bbox': bbox_inner,
                'fig_bbox': fig_bbox,
                'x_center': x_center,
                'title': None,
                'font': REL_DIMS['font'],
                'axis': {},
                'line_thickness_range': line_thickness_range}
        y_top, y_bottom = fig_bbox['y']
        title_h = int(x_h * REL_DIMS['text_height_frac']['title'])
        text_padding = int((title_h) * (REL_DIMS['title_spacing']-1))

        if self._title is not None:
            font_scale = get_font_scale(REL_DIMS['font'],
                                   max_height=title_h,
                                   max_width=x_w,
                                   text_lines=[self._title], max_font_scale = REL_DIMS['max_font_scale'])
            (t_width, t_height), t_baseline = cv2.getTextSize(self._title, REL_DIMS['font'], font_scale, 1)
            text_padding = int((title_h ) * REL_DIMS['title_spacing'])  # adjust

            title_y = y_top + t_height
            title_x = x_center - t_width // 2
            dims['title'] = {'x': title_x,
                             'y': title_y,
                             'font_scale': font_scale,
                             'baseline': t_baseline,
                             'text': self._title}
            y_top += text_padding
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
        ax_thick = int(REL_DIMS['axis']['thickness'])

        val_scale = self._get_scaling()

        dims['axis'] = {'x_left': x_left,
                        'x_right': x_right,
                        'y0': scale_y_value(0.0, y_span=axis_y, value_span = val_scale),
                        'y': axis_y,
                        'axis_thickness':  ax_thick,
                        'tick_len': axis_tick_len,
                        'font': REL_DIMS['font'],
                        'bbox': {'x': (x_left, x_right), 'y': axis_y},
                        'font_scale': font_scale}

        # Calculate where the values go on both axes.
        lines = []

        line_scale = scale_counts(self._pair_counts, line_thickness_range,
                                  REL_DIMS['line_alpha_range'])

        alpha_adjust = calc_alpha_adjust(n_lines=len(self._pair_counts),
                                         y_span=axis_y[1] - axis_y[0],
                                         line_t=med_thick,
                                         alpha_range=REL_DIMS['line_alpha_range'])

        for ind, (v0, v1) in enumerate(self._distinct_pairs):
            thickness, alpha = line_scale['t'][ind], line_scale['alpha']
            y_left = scale_y_value(v0, y_span=axis_y, value_span=val_scale)
            y_right = scale_y_value(v1, y_span=axis_y, value_span=val_scale)

            lines.append({'x': (x_left+ax_thick, x_right-ax_thick),
                          'y': (y_left, y_right),
                          'thickness': thickness,
                          'alpha': alpha_adjust})
        dims['lines'] = lines
        # Now we can draw the histograms at the same y positions as the lines.

        dp = {'line_alpha_range': REL_DIMS['line_alpha_range'],
              'line_thickness_range': line_thickness_range,
              'single_line_t': med_thick}

        dims['l_hist'] = MicroHist(self._left_vals, self._left_counts,
                                   v_scale=val_scale,
                                   bbox=h_bbox_left,
                                   left_facing=True, draw_params=dp)
        dims['r_hist'] = MicroHist(self._right_vals, self._right_counts,
                                   v_scale=val_scale,
                                   bbox=h_bbox_right,
                                   left_facing=False, draw_params=dp)
        return dims

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
    def draw(self, img, bbox=None):
        bbox = bbox if bbox is not None else {'x': (0, img.shape[1]), 'y': (0, img.shape[0])}
        dims = self._calc_dims(bbox)
        
        def draw_bbox(bbox, color):
            x0, x1 = bbox['x']
            y0, y1 = bbox['y']
            img[y0, x0:x1] = color
            img[min(img.shape[0]-1, y1), x0:x1] = color
            img[y0:y1, x0] = color
            img[y0:y1, min(img.shape[1]-1, x1)] = color

        # draw the bounding box
        #draw_bbox(dims['inner_bbox'], self._colors['lines'])
        #draw_bbox(dims['bbox'], self._colors['lines'])

        # draw_bbox(self._fig_bbox, self._draw_params['color'])
        # draw_bbox(self.bbox, self._draw_params['color'])
        # axis lines
        axis_y = dims['axis']['y']
        t = dims['axis']['axis_thickness']
        tick_len = dims['axis']['tick_len']
        ltr = dims['line_thickness_range']
        max_line_thickness = ltr[1]

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
                img[y_upper:y_lower, x1:min(x1+max_line_thickness, slope_bbox['x'][1])] = self._colors['bg']

            # Vertical axis line:
            img[axis_y[0]:axis_y[1], x0:x1] = color
            # draw the upper and lower ends:
            x_t_left = x0 - tick_len
            x_t_right = x1 + tick_len
            img[axis_y[0]:axis_y[0]+y_thick, x_t_left:x_t_right] = color  # top tick
            img[axis_y[1]-y_thick:axis_y[1], x_t_left:x_t_right] = color  # bottom tick



        # slopes
        for line in dims['lines']:

            x_left, x_right = line['x']
            y_left, y_right = line['y']
            thickness = line['thickness']
            alpha = line['alpha']
            draw_alpha_line(img, (x_left, x_right), (y_left, y_right),
                            color=None, thickness=thickness, alpha=alpha)
        # Draw axis over slopes
        axis_left = dims['axis']['x_left']
        axis_right = dims['axis']['x_right']
        draw_axis(axis_left, self._colors['lines'], t)
        draw_axis(axis_right, self._colors['lines'], -t)


        # ticks (just y=0 for now)
        y0 = dims['axis']['y0']

        if y0 >= dims['axis']['y'][0] and y0 <= dims['axis']['y'][1]:
            # draw the tick at y=0
            x_left = dims['axis']['x_left']
            x_right = dims['axis']['x_right']
            tick_w = REL_DIMS['ticks']['w']
            tick_alpha = REL_DIMS['ticks']['alpha']
            draw_alpha_line(img, (x_left, x_right), (y0, y0),
                            color=(255,128,128), thickness=tick_w, alpha=1.0)

        if dims['l_hist'] is not None:
            # Draw the left histogram
            dims['l_hist'].draw(img)
        if dims['r_hist'] is not None:
            # Draw the right histogram
            dims['r_hist'].draw(img)
        


        # title
        if self._title is not None:
            title_x, title_y = dims['title']['x'], dims['title']['y']
            cv2.putText(img, dims['title']['text'], (title_x, title_y), dims['font'],
                        dims['title']['font_scale'], self._colors['text'], 1, cv2.LINE_AA)
            
        return img


class DiagramSizeTester(object):
    """
    Tk window with single image lable (the diag), to show resize capability.
    """

    def __init__(self, size, img_factory):
        img = img_factory(size)
        self._img_size = img.shape[1], img.shape[0]
        self._make_frame = img_factory
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
        print(f"Resizing frame to {event.width}x{event.height}")
        self._img_size = self._frame.winfo_width(), self._frame.winfo_height()
        self.refresh_diag_image()

    def refresh_diag_image(self):
        img = self._make_frame(self._img_size)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self._label.config(image=img)
        self._label.image = img


def get_small_test_vals():
    vals1 = np.concatenate((np.random.randn(100), np.random.randn(100)*.5 + 2.0))
    vals2 = np.concatenate((np.random.randn(50)*.7-.5, np.random.randn(150)*1.0 + 1.0))
    vals1 = np.zeros(10)
    vals1[-1] = 1
    vals2 = np.array([-1.0]*1 + [0.0]*1 + [1.0]*2 + [2.0]*5 + [-3.0]*1)

    return vals1, vals2


def get_test_vals(n_points=100, n_clust_init=6, splits=3):
    """
    Start with N clusters, splitting/merging into M clusters.
    Use a small number of distinct values in each cluster.
    """

    init_vals = np.random.randn(n_clust_init) * 10.0  # initial cluster centers
    c_dist = np.random.rand(n_clust_init)  # initial cluster distribution
    c_dist = c_dist / np.sum(c_dist)  # normalize
    cluster_dist = np.random.multinomial(n_points, c_dist)  # initial cluster sizes
    # Now split the clusters into n_clusters
    start_vals = []
    end_vals = []

    for start_clust_ind, c_val in enumerate(init_vals):
        c_size = cluster_dist[start_clust_ind]
        new_start_vals = np.ones(c_size) * c_val
        # Split the cluster into splits clusters
        new_split_vals = np.random.randn(splits)*5.0 + c_val
        # divide up the cluster into the splits
        s_dist = np.random.rand(splits)
        s_dist = s_dist / np.sum(s_dist)
        split_dist = np.random.multinomial(c_size, s_dist)
        new_end_vals = 0 * new_start_vals
        n_ind = 0
        for split_ind, split_val in enumerate(new_split_vals):
            split_size = split_dist[split_ind]
            new_end_vals[n_ind: n_ind + split_size] = split_val
            n_ind += split_size
        start_vals.extend(new_start_vals)
        end_vals.extend(new_end_vals)

    start_vals = np.array(start_vals)
    end_vals = np.array(end_vals)

    return start_vals, end_vals


def test_diag():
    tests = [get_test_vals(10, 1, 3),
             get_test_vals(324, 3, 6),
             get_test_vals(2016, 10, 2),
             get_test_vals(2928, 30, 10),
             get_test_vals(1080, 6, 8)]

    def _make_diag_img(size, start_vals, end_vals):
        n_vals = start_vals.size
        diag = SlopeDiagram(size, start_vals, end_vals, title=None)#="Shift in %i values" % n_vals)
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']
        diag.draw(img)
        return img

    def _frame_factory(size):
        img_size = (size[0], size[1]//len(tests))
        #spacer_img = np.zeros((20, img_size[0], 3), dtype=np.uint8)

        imgs = [_make_diag_img(img_size, start_vals, end_vals) for (start_vals, end_vals) in tests]
        #frame_layout = [[f] for img in imgs for f in (img, spacer_img)][:-1]
        frame_layout = [[img] for img in imgs]
        frame = np.concatenate([np.concatenate(row, axis=1) for row in frame_layout], axis=0)
        return frame

    tester = DiagramSizeTester((100,    950), _frame_factory)
    tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_diag()
