"""
Organize large numbers of nodes in a window.

Tic-tac-toe game trees bulge in the middle, so they are displayed diagonally from the top left to the bottom right.

States with N non-empty cells are placed in band N.


"""
import numpy as np
import matplotlib.pyplot as plt

MIN_BOX_SIZE = 15
MAX_BOX_SIZE = 50


class BoxOrganizer(object):
    _DIMS = {'layer_space_px': 20,  # padding between layers
             'min_space_size': 6,  # a "space" is 1/3 of a game board side length.
             'max_space_size': 12,  # min layer_size = (max_space_size + 2 * layer_space_px)
             'grid_padding_frac': 0.05,   # multiply by grid_size to get padding on either side of a grid.
             'bar_thickness': 8}

    def __init__(self, layers, size_wh=(1900, 880)):
        """
        Calculate layout.
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i, and has the value:
             {'n_boxes': int,
             }
        :param size_wh:  size of the window in pixels (width, height).
        """
        self._layers = layers
        self._size_wh = size_wh
        self._layer_spacing = self._calc_layer_spacing()
        self._box_spacing = self._calc_box_positions()
        self._bkg_color = (127, 127, 127)
        self._line_color = (0, 0, 0)

        self._positions = []
        self._lines = []

    def get_layout(self):
        """
        return the computed positions of all the boxes & divison lines.
        """
        return self._positions,
        self._lines

    def _calc_layer_spacing(self):
        # for now, just evenly space the layers
        spacing = []
        y = 0
        import ipdb
        ipdb.set_trace()
        layer_h = int(self._size_wh[1] / len(self._layers))
        bar_h = self._DIMS['bar_thickness']
        for i, layer in enumerate(self._layers):
            layer_def = {'y': (y, y + layer_h),
                         'n_boxes': len(layer)}
            y += layer_h
            if i < len(self._layers) - 1:
                layer_def['bar_y'] = y, y + bar_h
                y += bar_h
            spacing.append(layer_def)
        return spacing

    def _calc_box_positions(self):
        box_pos = []
        for i, layer in enumerate(self._layers):
            w, h = self._size_wh[0], self._layer_spacing[i]['y'][1] - self._layer_spacing[i]['y'][0]
            # Use area of layer / number of boxes as upper bound for box size.
            n_boxes = len(layer)
            box_s, box_p = self._get_box_size(n_boxes, w=w, h=h)
            n_rows = h // (box_s + 2 * box_p)
            n_cols = w // (box_s + 2 * box_p)
            x, y = 0, self._layer_spacing[i]['y'][0]
            ind = 0
            for row in range(n_rows):
                row_len = min(n_cols, len(layer) - row * n_cols)
                x_lefts = np.linspace(0, w, row_len, endpoint=False) - box_s / 2
                for col in range(row_len):
                    box_pos[layer[ind]['id']] = {'x': (x_lefts[col], x_lefts[col]+box_s),
                                                 'y': (y, y+box_s)}
                    ind += 1

    def _get_box_size(self, n, w, h):
        """
        Largest box size s so that n boxes of size s x s can fit in a space with dimensions w x h.

        Arrange in a grid with as little wasted space as possible.
        Padding is a fraction of the box size, and applies to all four sides of the box.

        :param n: number of boxes to fit in the space
        :param w: width of the space
        :param h: height of the space
        :returns: s, the size of the largest box that can fit in the space, and the padding on all sides of the box.
        """
        # Find the largest square that can fit in the space
        s = min(w, h)
        while s > MIN_BOX_SIZE:
            pad = int(s * self._DIMS['grid_padding_frac'])
            n_rows = h // (s + 2 * pad)
            n_cols = w // (s + 2 * pad)
            if n_rows * n_cols >= n:
                return s, pad
            s -= 1
        return s, pad

    def draw(self):
        img = np.zeros((self._size_wh[1], self._size_wh[0], 3), np.uint8)
        img[:, :] = self._bkg_color
        # draw layer bars
        for layer in self._layer_spacing:
            y0, y1 = layer['y']
            if 'bar_y' in layer:
                img[layer['bar_y'][0]:layer['bar_y'][1], :] = self._line_color
        # draw boxes
        return img


def test_BoxOrganizer():
    next_id = [0]

    def make_boxes(n):
        boxes = [{'id': next_id[0] + i} for i in range(n)]
        next_id[0] += n
        return boxes

    layers = [make_boxes(1), make_boxes(9), make_boxes(72), make_boxes(729)]
    bo = BoxOrganizer(layers)
    img = bo.draw()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    test_BoxOrganizer()
