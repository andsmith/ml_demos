"""
Organize large numbers of nodes in a window.

Tic-tac-toe game trees bulge in the middle, so they are displayed diagonally from the top left to the bottom right.

States with N non-empty cells are placed in band N.


"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import cv2

import logging


MIN_BOX_SIZE = 7


class BoxOrganizer(ABC):
    """
    Base class for organizing a large number of small images in a single window.

    Images are arranged in layers.
    """

    def __init__(self, size_wh, layers, layer_vpad_px=7, layer_bar_w=4, brickwork=True, color_bg=(127, 127, 127), color_lines=(0, 0, 0)):
        """
        :param size_wh: size of the window in pixels (width, height).
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
               Each box is a dict with an 'id' key.
        :param layer_spacing_px: space above and below each layer of boxes.
        :param layer_bar_w: width of the layer division bar in pixels.
        :param brickwork: if True, boxes are arranged in a brickwork pattern (i.e. staggered rows).
        """
        self._brick = brickwork
        self.size_wh = size_wh
        self.layers = layers
        self._layer_vpad_px = layer_vpad_px
        self._layer_bar_w = layer_bar_w
        self._layer_counts = [len(layer) for layer in layers]
        # for drawing debug images
        self._bkg_color = color_bg
        self._line_color = color_lines
        self.layer_spacing = self._calc_layer_spacing()
        self.box_positions, self.grid_shapes = self._calc_box_positions()

    @abstractmethod
    def _calc_layer_spacing(self):
        """
        Calculate the spacing between layers.
        :return: list of dicts, each dict is a layer definition with keys:
            'y': (top, bottom) pixel coordinates of the layer
            'bar_y': (top, bottom) pixel coordinates of the layer division bar (i.e between each 'y' region)
            'n_boxes': number of boxes in this layer
        """
        pass

    @abstractmethod
    def _calc_box_positions(self):
        """
        Calculate the positions of the boxes in each layer.
        :return: box_pos: dict -> box_id -> {'x': (left, right), 'y': (top, bottom)}
                grid_shapes: list, for each layer:  {'box_side_len': int, 'n_rows': int, 'n_cols': int}
        """
        pass

    def draw_box(self, image, state_id, color, thickness=0):
        """
        Draw a box on the image.
        :param image: image to draw on.
        :param state_id: id of the box to draw.
        :param color: color of the box.
        """
        if color is None:
            return
        bos_pos = self.box_positions[state_id]
        x, y = bos_pos['x'], bos_pos['y']
        if thickness == 0:
            image[y[0]:y[1], x[0]:x[1]] = color
        else:
            color = int(color[0]), int(color[1]), int(color[2])
            cv2.rectangle(image, (x[0], y[0]), (x[1], y[1]), color, thickness, cv2.LINE_AA)

    def _draw_layer_bars(self, image):
        """
        Draw the layer division bars on the image.
        :param image: image to draw on.
        """
        for layer in self.layer_spacing:
            if 'bar_y' in layer:
                bar_y = layer['bar_y']
                image[bar_y[0]:bar_y[1], :] = self._line_color

    def draw(self, images=None, colors=None, dest=None, show_bars=False, **kwargs):
        """
        :param images: dict(box_id = image).  If None, will use the argument in ['colors'] key of each box.
        :Param colors: dict(box_id = color).  If None, will use the argument in ['colors'] key of each box
        :param dest: image to draw on.  If None, will create a new image.
        :param show_bars: if True, will draw the layer division bars.
        :param kwargs: additional arguments to pass to the draw_box (if drawing boxes)
        """
        if images is None and colors is None:
            raise Exception("No images or colors provided.")
        if dest is None:
            img = np.zeros((self.size_wh[1], self.size_wh[0], 3), np.uint8)
            img[:, :] = self._bkg_color
        else:
            img = dest

        # draw layer bars
        if show_bars:
            self._draw_layer_bars(img)

        # draw boxes
        for l_ind, layer_boxes in enumerate(self.layers):
            bad_boxes = set()
            for i, box in enumerate(layer_boxes):
                if box['id'] not in self.box_positions:
                    bad_boxes.add(box['id'])
                    continue

                if images is None:
                    self.draw_box(img, box['id'], colors[box['id']], **kwargs)
                else:
                    bos_pos = self.box_positions[box['id']]
                    x, y = bos_pos['x'], bos_pos['y']
                    tile = images[box['id']]
                    img[y[0]:y[0] + tile.shape[1], x[0]:x[0] + tile.shape[0]] = tile
            if len(bad_boxes) > 0:
                raise Exception("Layer %d: Missing boxes %s" % (l_ind, str(bad_boxes)))
        return img

    @staticmethod
    def get_h_v_spacing(n_rows, n_cols, box_size, w, h):
        """
        Fitting boxes of size box_size into n_rows and n_cols within a W x H rectangle,
        what horizontal and vertical padding will evenly space the boxes.
        """
        v_space = h - n_rows * box_size
        v_pad = v_space / (n_rows + 1)

        h_space = w - n_cols * box_size
        h_pad = h_space / (n_cols + 1)
        return h_pad, v_pad

    def _n_cols_n_rows(self, c_max, n):
        """
        Filling a grid with c_max columns, top to bottom, left to right, calculate the number of rows and columns 
        that n items will actually occupy.

        If the brickwork option is used, every other row will be shortened by 1, so it may require more rows.

        :param c_max: maximum number of columns
        :param r_max: maximum number of rows
        :param n: number of boxes to fit in the rectangle
        :return: n_rows, n_cols, the number of rows and columns that will fit in the rectangle.
        """
        n_cols = min(c_max, n)
        n_rows = np.ceil(n / n_cols).astype(int)  # without bricks.
        if self._brick and n_rows > 1:
            # fill up, brick_wise,
            r = 0
            n_filled = 0
            while n > 0:
                row_len = n_cols if r % 2 == 0 else n_cols-1
                n_filled += row_len
                n -= row_len
                r += 1
            n_rows = r

        return n_cols, n_rows

    def _row_col_adjust(self, r_max, c_max, n, y, x, s):
        """
        Reduce the number of columns until spacing around each box is as even as possible.
        """

        n_cols, n_rows = self._n_cols_n_rows(c_max, n)
        h_pad, v_pad = self.get_h_v_spacing(n_rows, n_cols, s,
                                            self.size_wh[0], y)
        ratio = np.abs(h_pad / v_pad - 1)  # want to     minimize this

        best = {'n_rows': n_rows, 'n_cols': n_cols, 'ratio': ratio}

        n_rows_used = n_rows

        while n_cols > 1:

            # Shift a column.
            n_cols -= 1

            new_n_rows_used = self._n_cols_n_rows(n_cols, n)[1]
            # print("\t\tTesting %i cols: requires %i rows to fit %i things." % (n_cols, new_n_rows_used, n))
            # out of space?
            if new_n_rows_used > r_max:
                break

            # Does this row/col size have a better spacing ratio?
            h_pad, v_pad = self.get_h_v_spacing(new_n_rows_used, n_cols,
                                                s,
                                                self.size_wh[0], y)

            ratio = np.abs(h_pad / v_pad - 1)  # want to minimize this

            if ratio > best['ratio'] and (new_n_rows_used > n_rows_used):
                # we're getting worse, so stop here.
                break

            best = {'n_rows': new_n_rows_used, 'n_cols': n_cols, 'ratio': ratio}
            n_rows_used = new_n_rows_used

        return best['n_rows'], best['n_cols']

    def get_layer_at(self, pos):
        """
        Return the index of the layer containing the x, y position, or None if it is out of bounds / between layers.
        """
        for l_ind, layer in enumerate(self.layer_spacing):
            y_min, y_max = layer['y']
            if y_min <= pos[1] <= y_max:
                return l_ind
        return None

    def get_state_at(self, pos):
        """
        Return the state id at the given x, y position, or None if it is out of bounds / between layers.
        """

        l_ind = self.get_layer_at(pos)
        if l_ind is None:
            return None, -1
        layer = self.layer_spacing[l_ind]
        y_min, y_max = layer['y']
        if y_min <= pos[1] <= y_max:
            for box_id, box_pos in self.box_positions.items():
                x_min, x_max = box_pos['x']
                if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                    return box_id, l_ind
        return None, -1


class FixedCellBoxOrganizer(BoxOrganizer):

    """
    Given each layer's box size, and the number of boxes in each layer, determine best layer spacing and box positions. 


    (Assume box sizes already include padding.)
    """

    def __init__(self, size_wh, layers, box_sizes, **argv):
        """
        :param size_wh: size of the window in pixels (width, height).
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
               Each box is a dict with an 'id' key.
        :param box_sizes: list of ints, the side-length of the square boxes in each layer.
        """
        self._box_side_lengths = box_sizes
        super().__init__(size_wh, layers, **argv)

    def _calc_layer_spacing(self):
        """
        Calculate layer spacing:
            1. Calculate each layer height assuming boxes are completely packed.  
            2. Distribute extra vertical space to each row, irrespective of layer, i.e. layers with r rows will get r+1 extra pixels of vertical space.

        Return:  layer_spacing: list of dicts, each dict is a layer definition with keys:
                    'y': (top, bottom) pixel coordinates of the layer
                    'bar_y': (top, bottom) pixel coordinates of the layer division bar (i.e between each 'y' region)
                    'n_boxes': number of boxes in this layer
        """
        layer_heights, layer_row_counts, usable_height = self._calc_layer_heights()

        print("Layer heights:", layer_heights)
        print("Layer row counts:", layer_row_counts)
        print("Window_height:  %i ,remaining_height: %i" %
              (self.size_wh[1], usable_height))
        layer_spacing = self._adjust_layer_heights(layer_heights, layer_row_counts, usable_height)
        return layer_spacing

    def _calc_layer_heights(self):

        n_layers = len(self._layer_counts)
        # 1. packed layers
        layer_row_counts = []

        for l, n_boxes in enumerate(self._layer_counts):
            n_cols = self.size_wh[0] // (self._box_side_lengths[l])
            n_rows = np.ceil(n_boxes / n_cols).astype(int)
            layer_row_counts.append(n_rows)
        layer_row_counts = np.array(layer_row_counts)

        # h = window height - layer bars - layer padding
        total_padding = n_layers * 2 * self._layer_vpad_px  # on each side
        total_bar_height = (n_layers - 1) * self._layer_bar_w
        used_height = total_padding + total_bar_height

        usable_height = self.size_wh[1] - used_height  # height not used for padding
        layer_heights = np.array([layer_row_counts[l] * self._box_side_lengths[l] for l in range(n_layers)])

        return layer_heights, layer_row_counts, usable_height

    def _adjust_layer_heights(self, layer_heights, layer_row_counts, usable_height):
        # 2. distribute extra space to layers equally until doing so would require too much, then
        # stop distributing to the layer w/the most rows and continue until there is no more space.

        n_layers = len(self._layer_counts)

        extra_height = usable_height - np.sum(layer_heights)
        adding_to = np.ones(n_layers, dtype=bool)  # which layers are still getting extra space
        adding_to[0] = False  # don't add to the first layer, it's already oversized for the keys
        while extra_height > 0:
            n_req = np.sum(layer_row_counts[adding_to])
            while n_req > extra_height and np.sum(adding_to) >= 1:
                # remove the layer with the most rows that we're still adding to from the list of layers to add to.
                # (i.e. the one that will require the most extra space to fill)
                consuption = layer_row_counts * adding_to
                biggest_portion = np.max(consuption)
                hungriest_set = np.where(consuption == biggest_portion)[0]
                hungriest = hungriest_set[-1]
                adding_to[hungriest] = False
                n_req = np.sum(layer_row_counts[adding_to])
            if np.all(~adding_to):
                break
            # distribute the extra space to the layers that are still getting it.
            layer_heights[adding_to] += layer_row_counts[adding_to]
            extra_height -= n_req

        # Now we can set layer spacing.
        layer_spacing = []

        y = self._layer_vpad_px
        for l, layer_h in enumerate(layer_heights):
            layer_top = y
            layer_bottom = layer_top + layer_h
            layer_def = {'y': (layer_top, layer_bottom),
                         'n_boxes': self._layer_counts[l]}
            y = layer_bottom + self._layer_vpad_px
            if l < n_layers - 1:
                layer_def['bar_y'] = (y, y + self._layer_bar_w)
                y += self._layer_bar_w + self._layer_vpad_px
            layer_spacing.append(layer_def)

        return layer_spacing

    def _calc_box_positions(self):
        """
        We know how big each layer is, how many boxes are in each, and what size each layer's boxes are.

        Find the final placement for each box in the image.

        1. Adjust rows/columns within each layer to get the most equal vertical & horizontal spacing between boxes.
           (i.e. reduce the number of columns until the last row is as full as possible, since they start out packed by column).    
        2. Distribute boxes evenly in each layer.

        returns:
            box_positions: dict mapping each box_id to a bounding box in pixels {'x': (left, right), 'y': (top, bottom)}
            grid_sizes: list, for each layer:  {'box_side_len': int, 'n_rows': int, 'n_cols': int}
        """

        # 3. Now we can calculate each layer's row/col count, and adjust the number of rows and columns to minimize wasted space.
        #      - Start assuming rows are packed, then
        #      - shift columns to the bottom row until its full, then
        #      - Add another row if it will improve the ratio of horizontal to vertical padding (between boxes)
        #      - Repeat until out of vertical space
        #  (assume no minimum box padding)

        grid_shapes = []
        layer_heights = [spacing['y'][1] - spacing['y'][0] for spacing in self.layer_spacing]

        for l, n_boxes in enumerate(self._layer_counts):
            n_cols_max = self.size_wh[0] // self._box_side_lengths[l]
            n_rows_max = layer_heights[l] // self._box_side_lengths[l]
            y_space = self.layer_spacing[l]['y'][1] - self.layer_spacing[l]['y'][0]
            x_space = self.size_wh[0]
            box_size = self._box_side_lengths[l]

            n_rows, n_cols = self._row_col_adjust(n_rows_max, n_cols_max, n_boxes, y_space, x_space, box_size)
            # print("Fit %i items into a %i x %i grid, using a %i x %i sub-grid." %
            #      (n_boxes, n_cols_max, n_rows_max, n_cols, n_rows))
            grid_shapes.append({'box_side_len': self._box_side_lengths[l],
                                'n_rows': n_rows,
                                'n_cols': n_cols})

        # Now we can calculate the box positions in each layer.
        box_positions = {}  # state -> {'x': (left, right), 'y': (top, bottom)}
        for l, boxes in enumerate(self.layers):
            # print("Layer %i: %i boxes" % (l, len(boxes)))
            layer_top, layer_bottom = self.layer_spacing[l]['y']
            layer_h = layer_bottom - layer_top
            layer_w = self.size_wh[0]
            box_s = grid_shapes[l]['box_side_len']
            n_rows = grid_shapes[l]['n_rows']
            n_cols = grid_shapes[l]['n_cols']

            # Keep padding floats to distribute evenly.
            x_pad, y_pad = self.get_h_v_spacing(
                n_rows, n_cols, box_s, self.size_wh[0], layer_h)  # use v-padding for placement
            n = 0
            for row in range(n_rows):
                y = int(layer_top + row * (box_s + y_pad) + y_pad)

                row_len = min(n_cols, len(boxes) - n)
                x_pad = (layer_w - row_len * box_s) / (row_len + 1)  # between boxes on this row

                # if this is a odd-brick row (and not the last row)
                if self._brick and (row % 2 == 1) and (row != n_rows - 1) and (n_rows > 2):
                    row_len = min(n_cols-1, len(boxes) - n)  # shorten row, don't change padding, add offset
                    # print("BRICK ON LAYER %i, row %i (n_rows=%i)" % (l, row, n_rows))
                    x_offset = (box_s + x_pad)/2
                else:
                    x_offset = 0.0
                # print("\tRow %i of %i has %i items." % (row, n_rows, row_len))

                for col in range(row_len):
                    x = int(col * (box_s + x_pad) + x_pad + x_offset)
                    box_id = boxes[n]['id']
                    box_positions[box_id] = {'x': (x, x + box_s),
                                             'y': (y, y + box_s)}
                    n += 1
                if n == len(boxes):
                    break
            # print("\n")
            if n != len(boxes):
                raise Exception("Did not place all boxes in layer: %d vs %d (MinBoxSize too high.)" % (n, len(boxes)))
        return box_positions, grid_shapes


class FixedCellWithKey(FixedCellBoxOrganizer):
    """
    States in the top row are moved toward the left, upper right area is reserved for a key.
    """

    def __init__(self, size_wh, layers, box_sizes, min_key_h=0, min_key_w=0, **argv):
        """
        :param size_wh: size of the window in pixels (width, height).
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
               Each box is a dict with an 'id' key.
        :param box_sizes: list of ints, the side-length of the square boxes in each layer.
        :param min_key_h: minimum height of the color key area.
        """
        self._min_key_h = min_key_h
        self._min_key_w = min_key_w
        super().__init__(size_wh, layers, box_sizes, **argv)  # initializes box positions
        self.key_bbox = self._shift_box_positions()

    def _calc_layer_spacing(self):
        layer_heights, layer_row_counts, usable_height = self._calc_layer_heights()

        if layer_heights[0] < self._min_key_h:
            extra = min(usable_height, self._min_key_h - layer_heights[0])
            logging.info("Adding %d pixels to the first layer to make room for the color key." % extra)
            layer_heights[0] += extra
        layer_spacing = self._adjust_layer_heights(layer_heights, layer_row_counts, usable_height)
        return layer_spacing

    def _shift_box_positions(self):
        """
        The bounding box for the color key has width min_key_w and occupies the right portion of the top row.
        Horizontally space the boxes for those states evenly in the remaining space on the left 
        """
        if self._min_key_w <= 0:
            return

        rel_x_shfit = (self.size_wh[0] - self._min_key_w) / self.size_wh[0]
        logging.info("Shifting box positions by %f to make room for the color key." % rel_x_shfit)
        for box_info in self.layers[0]:
            position = self.box_positions[box_info['id']]
            x_left, x_right = position['x']
            center = (x_left + x_right) / 2
            new_center = center * rel_x_shfit
            x_delta = new_center - center
            position['x'] = (int(x_left + x_delta), int(x_right + x_delta))

        bbox_bottom = self.layer_spacing[0]['bar_y'][0]
        return {'x': (self.size_wh[0] - self._min_key_w, self.size_wh[0]),
                'y': (0, bbox_bottom)}

    def _draw_layer_bars(self, image):
        # draw a vertical line between the top row and the color key area.
        if False:
            line_x = self.key_bbox['x'][0]
            line_y_top = self.key_bbox['y'][0]
            line_y_bottom = self.layer_spacing[0]['bar_y'][0]
            line_x_left = line_x - self._layer_bar_w//2
            line_x_right = line_x_left + self._layer_bar_w
            image[line_y_top:line_y_bottom, line_x_left:line_x_right] = self._line_color
        return super()._draw_layer_bars(image)


class LayerwiseBoxOrganizer(BoxOrganizer):
    """ 
    Given layer spacing, fit n = Layer[l] boxes in each layer.
    """

    def __init__(self, size_wh, layers, v_spacing, min_separation_frac=0.1):
        """
        Calculate layout.
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
              Each box is a dict with an 'id' key.
        :param v_spacing: list of dicts, each dict is a layer definition:
            { 'y': (top, bottom),
              'n_boxes': number of boxes in this layer,
              'bar_y': (top, bottom) # optional 
              }
        :param size_wh:  size of the window in pixels (width, height).
        :param min_separation_frac: minimum separation between boxes as a fraction of the box size.
        """
        self._min_sep_frac = min_separation_frac
        self._min_layer_size = 50
        self._ls_temp = v_spacing
        super().__init__(size_wh, layers)

    def _calc_layer_spacing(self):
        return self._ls_temp

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
            box_sizes: list, for each layer, {'box_side_len': int, 'n_rows': int, 'n_cols': int}
        """
        box_sizes = []
        box_pos = {}
        for i, layer in enumerate(self.layers):
            top_y = self.layer_spacing[i]['y'][0]+1
            # print("Layer", i, "top_y", top_y)
            bottom_y = self.layer_spacing[i]['y'][1]
            w, h = self.size_wh[0],  bottom_y - top_y

            # Use area of layer / number of boxes as upper bound for box size.
            n_boxes = len(layer)

            # box_p is padding on all sides of each box
            # box_s is the size of the box
            box_s, box_p, n_rows, n_cols = self._get_box_size(n_boxes, w=w, h=h)
            # print(n_rows, n_cols)
            n_rows, n_cols = self._row_col_adjust(n_rows, n_cols, n_boxes, h, w, box_s+box_p)
            # calculate extra space, and divide it between rows and columns
            # print(n_rows, n_cols, "after")
            v_space = h - n_rows * box_s
            v_pad = v_space / (n_rows + 1)

            ind = 0
            for row in range(n_rows):
                y = int(top_y + v_pad + row * (box_s + v_pad))
                # print(y, v_pad)

                row_len = min(n_cols, len(layer) - row * n_cols)  # last row may have fewer boxes
                h_space = w - row_len * box_s
                h_pad = h_space / (row_len + 1)  # between boxes on this row

                x_lefts = [int(h_pad + col * (box_s + h_pad)) for col in range(row_len)]
                for col in range(row_len):
                    x = x_lefts[col]
                    box_pos[layer[ind]['id']] = {'x': (x, x + box_s),
                                                 'y': (y, y + box_s)}
                    ind += 1
                if ind == len(layer):
                    break

            if ind != len(layer):
                raise Exception("Did not place all boxes in layer: %d vs %d (MinBoxSize too high.)" % (ind, len(layer)))

            box_sizes.append({'box_side_len': box_s, 'n_rows': n_rows, 'n_cols': n_cols})
        return box_pos, box_sizes

    def _get_box_size(self, n, w, h):
        """
        Largest box size s so that n boxes of size s x s can fit in a space with dimensions w x h.

        Arrange in a grid with as little wasted space as possible.

        :param n: number of boxes to fit in the space
        :param w: width of the space
        :param h: height of the space
        :returns: s, the size of the largest box that can fit in the space, and the padding on all sides of the box.

        Ensure minimum separation (in pixels):

            Height = n_rows * box_size + (n_rows -1) * pad
            Width = n_cols * box_size + (n_cols -1) * pad


        Rearranging gives number of rows and columns for a given box size, padding, height and width:

                n_rows = (height - pad) / (box_size + pad)
                n_cols = (width - pad) / (box_size + pad) 

        """
        # Find the largest square that can fit in the space
        s = min(w, h)
        while s >= MIN_BOX_SIZE:
            pad = 1 if s < 10 else int(s*self._min_sep_frac)
            n_rows = (h - pad) // (s + 2 * pad)
            n_cols = (w - pad) // (s + 2 * pad)

            if n_rows * n_cols >= n:
                break
            s -= 1

        if s < MIN_BOX_SIZE:
            raise ValueError("Box size too small to fit all boxes: W=%i, H=%i, N=%i" % (w, h, n))
        print("\tbox size %s (pad %s): (rows: %s) x (cols: %s) = %s is at least %s" %
              (s, pad, n_rows, n_cols, n_rows*n_cols, n))

        return s, pad, n_rows, n_cols

    def _row_col_adjustX(self, n_boxes, box_size, w, h):
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

        while h_pad < v_pad and (n_rows_used * box_size <= h):
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


class AutoBoxOrganizer(LayerwiseBoxOrganizer):
    """
    Find the best fit for n_l boxes in each horizontal layer l (largest box size, best balance of rows/cols).
    Boxes are square, and the same size in each layer.
    """

    def __init__(self, layers, size_wh=(1900, 2080)):
        """
        Calculate layout.
        :param layers: list of lists of boxes, each list is a layer:
           layer[i] is a list of boxes that will be placed in band i.
              Each box is a dict with an 'id' key.
        :param v_spacing: list of dicts, each dict is a layer definition with keys:
        :param size_wh:  size of the window in pixels (width, height).
        """
        self._min_layer_size = 50
        super().__init__(size_wh, layers)

        self.layer_spacing = self._calc_layer_spacing()

    def _calc_layer_spacing(self):
        # for now, just evenly space the layers
        spacing = []
        y = 0
        bar_h = self._layer_bar_w
        layer_h = int((self.size_wh[1]-bar_h * (len(self.layers)-1)) / len(self.layers))
        for i, layer in enumerate(self.layers):
            layer_def = {'y': (y, y + layer_h),
                         'n_boxes': len(layer)}
            y += layer_h
            if i < len(self.layers) - 1:
                layer_def['bar_y'] = y, y + bar_h
                y += bar_h
            spacing.append(layer_def)

        return spacing


def test_BoxOrganizer():
    next_id = [0]

    def make_boxes(n):

        boxes = [{'id': next_id[0] + i,
                  'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
                 for i in range(n)]

        next_id[0] += n

        return boxes
    game_layers = [1,  18,  72, 504,  756, 2520, 1668, 2280, 558, 156]
    space_sizes = [9,   6,   5,   3,    2,    2,    2,    2,   3,   4]
    from drawing import GameStateArtist
    box_sizes = [GameStateArtist(s).dims['img_size'] for s in space_sizes]
    print("Space sizes: ", space_sizes)
    print("Box sizes: ", box_sizes)
    size_wh = 1920, 1055
    layers = [make_boxes(ttt) for ttt in game_layers]

    bo = FixedCellBoxOrganizer(size_wh, layers, box_sizes=box_sizes)
    img = bo.draw(show_bars=True)
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img)

    plt.show()


if __name__ == "__main__":

    test_BoxOrganizer()
