"""
Organize large numbers of nodes in a window.

Tic-tac-toe game trees bulge in the middle, so they are displayed diagonally from the top left to the bottom right.

States with N non-empty cells are placed in band N.


"""
import numpy as np
import matplotlib.pyplot as plt

MIN_BOX_SIZE = 7


class BoxOrganizer(object):
    _DIMS = {'layer_padding_px': 20,  # padding on either side of the layer division bar
             'grid_padding_frac': 0.05,   # multiply by grid_size to get padding on either side of a grid.
             'bar_thickness': 8  # thickness of the layer division bar
             }

    def __init__(self, layers, spacing=None, size_wh=(1900, 2080)):
        """
        Calculate layout.
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
              Each box is a dict with an 'id' key.
        :param size_wh:  size of the window in pixels (width, height).
        """
        self._min_layer_size = 50
        self._layers = layers
        self._size_wh = size_wh
        self._layer_spacing = self._calc_layer_spacing() if spacing is None else spacing
        self._box_positions, self._box_dims = self._calc_box_positions()
        self._bkg_color = (127, 127, 127)
        self._line_color = (0, 0, 0)

    def get_layout(self):
        """
        return the computed positions of all the boxes & divison lines.
        """
        return self._box_positions, self._box_dims, self._layer_spacing

    def _calc_layer_spacing(self):
        # for now, just evenly space the layers
        spacing = []
        y = 0
        bar_h = self._DIMS['bar_thickness']
        layer_h = int((self._size_wh[1]-bar_h * (len(self._layers)-1)) / len(self._layers))
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
        """
        +-------------------------+
        |                         |    
        | +-+  +-+  +-+  +-+  +-+ |
        | | |  | |  | |  | |  | | | V & H spacing divides space between boxes evenly.
        | +-+  +-+  +-+  +-+  +-+ |
        |                         |
        | +-+  +-+  +-+  +-+  +-+ |
        | | |  | |  | |  | |  | | |
        | +-+  +-+  +-+  +-+  +-+ |
        |                         |
        +-------------------------+

        :returns: 
            box_pos: dict -> box_id -> {'x': (left, right), 'y': (top, bottom)}
            box_dims: list, for each layer, a dict -> {'box_side_len': int, 'n_rows': int, 'n_cols': int}
        """
        box_dims = []
        box_pos = {}
        for i, layer in enumerate(self._layers):
            top_y = self._layer_spacing[i]['y'][0]+1
            #print("Layer", i, "top_y", top_y)
            bottom_y = self._layer_spacing[i]['y'][1]
            w, h = self._size_wh[0],  bottom_y - top_y

            # Use area of layer / number of boxes as upper bound for box size.
            n_boxes = len(layer)

            # box_p is padding on all sides of each box
            # box_s is the size of the box
            box_s, box_p, n_rows, n_cols = self._get_box_size(n_boxes, w=w, h=h)
            #print(n_rows, n_cols)
            n_rows, n_cols = self._row_col_adjust(n_boxes, box_s, w, h)
            # calculate extra space, and divide it between rows and columns
            #print(n_rows, n_cols, "after")
            v_space = h - n_rows * box_s
            v_pad = v_space / (n_rows + 1)

            ind = 0
            for row in range(n_rows):
                y = int(top_y + v_pad + row * (box_s + v_pad))
                #print(y, v_pad)

                row_len = min(n_cols, len(layer) - row * n_cols)  # last row may have fewer boxes
                h_space = w - row_len * box_s
                h_pad = h_space / (row_len + 1)  # between boxes on this row

                x_lefts = [int(h_pad + col * (box_s + h_pad)) for col in range(row_len)]
                for col in range(row_len):
                    x = x_lefts[col]
                    box_pos[layer[ind]['id']] = {'x': (x, x + box_s),
                                                 'y': (y, y + box_s)}
                    ind += 1
                if ind==len(layer):
                    break

            if ind != len(layer):
                raise Exception("Did not place all boxes in layer: %d vs %d (MinBoxSize too high.)" % (ind, len(layer)))

            box_dims.append({'box_side_len': box_s, 'n_rows': n_rows, 'n_cols': n_cols})
        return box_pos, box_dims

    def _get_box_size(self, n, w, h):
        """
        Largest box size s so that n boxes of size s x s can fit in a space with dimensions w x h.

        Arrange in a grid with as little wasted space as possible.

        :param n: number of boxes to fit in the space
        :param w: width of the space
        :param h: height of the space
        :returns: s, the size of the largest box that can fit in the space, and the padding on all sides of the box.
        """
        # Find the largest square that can fit in the space
        s = min(w, h)
        while s >= MIN_BOX_SIZE:
            pad = int(s * self._DIMS['grid_padding_frac'])
            n_rows = h // (s + 2 * pad)
            n_cols = w // (s + 2 * pad)

            # print("\tTrying box size %s: %s x %s = %s (<>%s)" % (s, n_rows, n_cols, n_rows*n_cols, n))
            if n_rows * n_cols >= n:
                return s, pad, n_rows, n_cols
            s -= 1

        if s == MIN_BOX_SIZE:
            raise ValueError("Box size too small to fit all boxes: W=%i, H=%i, N=%i" % (w, h, n))

        return s, pad, n_rows, n_cols

    @staticmethod
    def get_h_v_spacing(n_rows, n_cols, box_size, w, h):
        v_space = h - n_rows * box_size
        v_pad = v_space / (n_rows + 1)

        h_space = w - n_cols * box_size
        h_pad = h_space / (n_cols + 1)
        return h_pad, v_pad

    def _row_col_adjust(self, n_boxes, box_size, w, h):
        """
        Adjust the number of rows and columns to minimize wasted space.  If (n_rows, n_cols) is be the output 
        of _get_box_size, they are the numbe of rows and columns that will fit the largest box size with enough
        space to fit n_boxes.  This may contain extra rows and/or a mostly empty final row.  

        1. Calculate how many (whole) rows and columns of boxes can fit (should be >= n_boxes).
        2. Assume columns are spaced as close as possible, count the rows.
        3. Adjust the number of columns and rows until each box as roughly the same amount of empty space around it
           (assuming boxes are spread as evenly as possible).
        4. Reduce the number of columns until the last row is as full as possible.
        5. Return the adjusted number of rows and columns.

        :param n_boxes: number of boxes to fit 
        :param box_size: size of the boxes
        :param w: width of the space
        :param h: height of the space
        :returns: n_rows, n_cols, adjusted to eaven out empty space.
        """
        n_rows, n_cols = np.floor(h / box_size).astype(int),  np.floor(w / box_size).astype(int)
        n_rows_used = np.ceil(n_boxes / n_cols).astype(int)
        h_pad, v_pad = self.get_h_v_spacing(n_rows_used, n_cols, box_size, w, h)
        last = {'h_pad': h_pad, 'v_pad': v_pad, 'n_rows': n_rows_used, 'n_cols': n_cols}

        while h_pad < v_pad and (n_rows_used * box_size <=h):
            # remove a column, add it to the end of the last row, creat a new row if needed.
            last = {'h_pad': h_pad, 'v_pad': v_pad, 'n_rows': n_rows_used, 'n_cols': n_cols}
            n_cols -= 1
            n_rows_used = np.ceil(n_boxes / n_cols).astype(int)
            h_pad, v_pad = self.get_h_v_spacing(n_rows_used, n_cols, box_size, w, h)

        # Decide if this result is better than the previous one
        # by which has smaller difference in h/v padding.
        if abs(h_pad - v_pad) >= abs(last['h_pad'] - last['v_pad']):
            n_rows_used = last['n_rows']
            n_cols = last['n_cols']

        # reduce the number of columns until the last row is as full as possible (but don't increase the number of rows)
        if n_rows_used == 1:
            return 1, n_boxes

        new_n_cols = n_cols
        new_n_rows = n_rows_used
        while new_n_rows == n_rows_used:
            new_n_cols -= 1
            new_n_rows = np.ceil(n_boxes / new_n_cols).astype(int)

        n_rows, n_cols = new_n_rows-1, new_n_cols+1
        return n_rows, n_cols
    

class BoxOrganizerPlotter(BoxOrganizer):
    """
    Like a BoxOrganizer, but draw can fill in the boxes with colors or images
    """
    def draw(self, images = None,dest=None, show_bars=False):
        """
        :param images: dict(box_id = image).  If None, will use the argument in ['colors'] key of each box.
        """
        if dest is None:
            img = np.zeros((self._size_wh[1], self._size_wh[0], 3), np.uint8) 
            img[:, :] = self._bkg_color
        else:
            img = dest
        # draw layer bars
        if show_bars:
            for layer in self._layer_spacing:
                if 'bar_y' in layer:
                    img[layer['bar_y'][0]:layer['bar_y'][1], :] = self._line_color
        # draw boxes
        for l_ind, layer_boxes in enumerate(self._layers):
            #if l_ind==6:
            #    import ipdb; ipdb.set_trace()
            #print("Drawing layer", l_ind)
            for i, box in enumerate(layer_boxes):
                bos_pos = self._box_positions[box['id']]
                x,y= bos_pos['x'], bos_pos['y']
                if images is None:
                    img[y[0]:y[1], x[0]:x[1]] = box['color']
                else:
                    tile = images[box['id']]
                    img[y[0]:y[0]+ tile.shape[1], x[0]:x[0]+ tile.shape[0]] = tile
                

        return img


def test_BoxOrganizer():
    next_id = [0]

    def make_boxes(n):

        boxes = [{'id': next_id[0] + i,
                  'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
                 for i in range(n)]

        next_id[0] += n

        return boxes

    layers = [make_boxes(1), make_boxes(2), make_boxes(20), make_boxes(2200),
              make_boxes(2), make_boxes(729), make_boxes(72)]
    bo = BoxOrganizerPlotter(layers)
    img = bo.draw()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    test_BoxOrganizer()
