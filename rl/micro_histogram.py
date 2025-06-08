import numpy as np
import cv2
from resize_test import ResizingTester
from colors import COLOR_SCHEME
import logging
from plot_util import scale_y_value, draw_alpha_line


def get_n_bins(values, min_bins=20):
    """
    Use the Freedman-Diaconis rule to determine the number of bins for a histogram.
    """
    n_distinct = len(np.unique(values))
    if n_distinct <= min_bins:
        return None
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(values) ** (1 / 3))
    if bin_width == 0:
        bin_width = 1
    # print(values.size,n_distinct, bin_width)
    # print(values.min(), values.max(), q25, q75, iqr)

    n_bins = int(np.ceil((values.max() - values.min()) / bin_width))
    if n_bins < min_bins:
        return None
    return n_bins


class MicroHist(object):

    _DEFAULT_PARAMS = {'single_line_t': 10,  # fraction of height
                       'color': COLOR_SCHEME['lines'],
                       'line_alpha_range': (0.1, 0.95)}

    def __init__(self, values, counts, v_scale,  bbox, left_facing=False, draw_params=None):
        """
        Initialize a MicroHist object with the given values and counts.

        Decide whether to use bins or show every count as a line based on which is predicted to be more
        informative visually.

        :param values:  list of values to be plotted in the histogram
        :param counts:  list of counts corresponding to the values
        :param v_scale:  (min_value, max_value) to span the histogram vertically.
                        All values will be in this range.
        :param bbox:  {'x': (x_min, x_max), 'y': (y_min, y_max)}
        :param left_facing:  If True, the histogram will extend to the left instead of the right.
        """
        self._left_facing = left_facing
        self._values = values
        self._counts = counts
        self._c_norm = self._counts / np.max(self._counts)
        self._v_scale = v_scale
        self._bbox = bbox
        self._draw_params = MicroHist._DEFAULT_PARAMS.copy()
        if draw_params is not None:
            self._draw_params.update(draw_params)

        # 
        self._kind = 'lines'  # 'bins' if self._n_bins is not None else 'lines'

        if self._kind == 'lines':
            self._y_positions = scale_y_value(values, v_scale, y_span=(bbox['y'][0], bbox['y'][1]))
            self._lines = self._calc_lines(backwards=left_facing)

        elif self._kind == 'bins':
            raise NotImplementedError("Drawing bins is not implemented yet.")
            self._n_bins = get_n_bins(len(counts))
            self._counts, self._bins = np.histogram(values,
                                                    bins=self._n_bins,
                                                    range=(v_scale[0], v_scale[1]),
                                                    weights=counts)
            self._c_norm = self._counts / np.max(self._counts)
            self._bin_y = scale_y_value(self._bins, v_scale, y_span=(bbox['y'][0], bbox['y'][1]))
            self._bars = self._calc_bars(backwards=left_facing)
        else:
            raise ValueError(f"Unknown histogram type: {self._kind}")

    def _calc_bars(self, backwards=False):
        raise NotImplementedError("Drawing bins is not implemented yet.")

    def _calc_alpha_lines(self, n_lines, y_span, line_t):
        """
        Given how many lines there are, the span, and their thickness,
        how transparent should they be?
        """
        est_coverage = n_lines * line_t / y_span
        est_alpha_raw = 1.0 / (est_coverage*2)
        est_alpha = np.clip(est_alpha_raw, self._draw_params['line_alpha_range']
                            [0], self._draw_params['line_alpha_range'][1])
        print(
            f"Estimated coverage: {est_coverage:.2f} for {n_lines} lines, y_span={y_span}, line_t={line_t} --->  alpha_raw={est_alpha_raw:.2f}--->  alpha={est_alpha:.2f}")
        return est_alpha

    def _calc_lines(self, backwards=False):
        """
        Get the layout for  a sideways histogram for the given values, 
        in the given bounding box.  Default has lines extending to the right.
        (Backwards is the other way.)

        :param bbox:  {'x': (x_min, x_max), 'y': (y_min, y_max)}
        :param vscale:  (min_value, max_value) to span the histogram vertically.
        (all values will be in this range)
        :param backwards:  If True, the histogram will extend to the left instead of the right.
        :returns:  list of line dicts, each with keys: p0, p1, thickness, color, alpha
        """

        x_min, x_max = self._bbox['x']
        x_span = x_max - x_min
        y_span = self._bbox['y'][1] - self._bbox['y'][0]

        lines = []
        line_t = self._draw_params['single_line_t']
        alpha = self._calc_alpha_lines(len(self._y_positions), y_span, line_t)

        for (y_pos, bar_norm_len) in zip(self._y_positions, self._c_norm):
            if backwards:
                bar_right = x_max
                bar_left = x_max - x_span * bar_norm_len
            else:
                bar_left = x_min
                bar_right = x_min + x_span * bar_norm_len

            if y_pos - line_t//2 < self._bbox['y'][0]:
                y_pos = self._bbox['y'][0] + line_t // 2
            if y_pos + line_t // 2 > self._bbox['y'][1]:
                y_pos = self._bbox['y'][1] - line_t // 2
            lines.append({'x': (bar_left, bar_right),
                          'y_pos': y_pos,
                          'thickness': line_t,
                          'color': self._draw_params['color'],
                          'alpha': alpha})

        return lines

    def draw(self, img):

        def draw_bbox(bbox, color):
            x0, x1 = bbox['x']
            y0, y1 = bbox['y']
            img[y0, x0:x1] = color
            img[min(y1, img.shape[0]-1), x0:x1] = color
            img[y0:y1, x0] = color
            img[y0:y1, min(x1, img.shape[1]-1)] = color

        draw_bbox(self._bbox, self._draw_params['color'])

        if self._kind == 'lines':
            for line in self._lines:
                x = line['x']
                y_pos = line['y_pos']
                draw_alpha_line(img, x, (y_pos, y_pos), color=line['color'],
                                alpha=line['alpha'], thickness=line['thickness'])

        elif self._kind == 'bins':
            raise NotImplementedError("Drawing bins is not implemented yet.")
        
def test_micro_hist():

    n_vals = 100
    count_range = 1000
    bvalues = np.random.randn(n_vals)
    bcounts = np.random.randint(1, count_range, n_vals)

    n_vals = 10
    count_range = 100
    svalues = np.random.randn(n_vals)
    scounts = np.random.randint(1, count_range, n_vals)

    n_vals = 3
    count_range = 3
    mvalues = np.random.randn(n_vals)
    mcounts = np.random.randint(1, count_range, n_vals)

    def _make_test_frame(values, counts, size):

        v_scale = (np.min(values), np.max(values))
        m = 10
        bbox_left = {'x': (m, size[0]//2-m//2), 'y': (m, size[1]-m)}
        bbox_right = {'x': (size[0]//2+m//2, size[0]-m), 'y': (m, size[1]-m)}
        mh_left = MicroHist(values, counts, v_scale, bbox_left, left_facing=True)
        mh_right = MicroHist(values, counts, v_scale, bbox_right, left_facing=False)
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']
        mh_left.draw(img)
        mh_right.draw(img)
        return img

    def img_factory(size):

        tests = [(bvalues, bcounts),
                 (mvalues, mcounts),
                 (svalues, scounts)]

        pwidth = size[0] // len(tests)
        frame_size = (pwidth, size[1])
        frames = [_make_test_frame(vals,counts, frame_size) for (vals, counts) in tests]

        img = np.concatenate(frames, axis=1)
        return img

    tester = ResizingTester(img_factory, (900, 600))
    tester.start()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_micro_hist()
