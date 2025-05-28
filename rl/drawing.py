import numpy as np
from game_base import Mark, Result, WIN_MARKS, get_cell_center
from util import get_annulus_polyline
import cv2
from colors import COLOR_BG, COLOR_LINES, COLOR_DRAW, COLOR_X, COLOR_O, COLOR_DRAW_SHADE, COLOR_SHADE

MARKER_COLORS = {Mark.X: COLOR_X,
                 Mark.O: COLOR_O,
                 Mark.EMPTY: COLOR_BG}

_SHIFT_BITS = 5
_SHIFT = 1 << _SHIFT_BITS


def get_font_scale(font, max_height, incl_baseline=False):
    """
    Find the maximum font scale that fits a number in the given height.
    :param font_name: Name of the font to use.
    :param max_height: Maximum height of the text.
    :return: The maximum font scale that fits the text in the given height.
    """
    scale = 5.0
    while True:
        (_, text_height), baseline = cv2.getTextSize('0', font, scale, 1)
        # print("Text height for scale %.2f is %i  (should be under %i)" % (scale, text_height , max_height))
        h = text_height + baseline if incl_baseline else text_height
        if h < max_height:
            break
        scale -= 0.01
    return scale

class GameStateArtist(object):
    """
    Render a game state to an image, in these categories, depending on the space size:

        * "normal" sized images (size >= 10):  
            - box/grid lines have thickness that is a fraction of the space size. (int)
            - Markers X and O also have proportional thickness (int).  (for now, the same as the box and grid lines)
            - Lines are drawn with anti-aliased edges.
            - Terminal states have box/grid lines drawn in the winner/draw color and a line through each winning triplet ("win lines").

        * "small" sized images: (only size 5)
            - lines are 1-pixel wide, with no anti-aliasing
            - marker X is a 3x3 + sign, 
            - marker O is a 3x3 square with a hole in the middle.
            - For terminal states, box/grid lines change color and single-pixel wide win-lines are drawn.

        * "tiny" sized images: (size < 5)
            - lines are 1-pixel wide, with no anti-aliasing
            - markers are both completely filled squares with the player's color.
            - For terminal states, the losing player's marked cells have an inner border of the winner's color, (or all cells if a draw).
            - no win lines are drawn.

    Auxiliary functions:
        * Compute marker / grid / box line widths.
        * Compute the image sizes, the width/height/offsets of each of the 9 cells.
        * Compute the "attachment points" for the vertical grid lines (endpoints of both).  This is where states are connected by edges.
            - These are the (float) centerpoint of the line endpoint, whatever its width (will end with .5 if grid lines are even width).


    """

    @staticmethod
    def get_size(space_size):
        if space_size >= 11:
            return 'normal'
        elif space_size >= 5:
            return 'small'
        elif False:  # space_size > 3:
            return 'tiny'
        else:
            return 'micro'
            # no grid lines

    def __init__(self, space_size, bar_w_frac=.2):
        """
        :param space_size: int, width in pixels of one of the 9 squares in the 3x3 game grid.
        :param bar_w_frac: float, width of the lines of the grid as a fraction of space_size.
        """
        self._space_size = space_size
        self._bar_w_frac = bar_w_frac
        self.dims = self._get_image_dims()

        # For clean drawing:
        self._SHIFT_BITS = 5
        self._SHIFT = 1 << self._SHIFT_BITS

    def _get_image_dims(self):
        """
        Get the dimensions dict for drawing a game state.

                +--u--u--+
                |  |  |  |
                +--+--+--+
                |  |  |  |
                +--+--+--+
                |  |  |  |
                +--l--l--+

        * The grid image is composed of 3x3 "cells" or [marker] "spaces."
        * The line width is expressed as a fraction of the cell side length.
        * The upper and lower attachment points are the 'u' and 'l' points, respectively.
        * The bounding box may or may not be drawn, but is included in the grid's side length.

        Therefore, the smallest possible grid size is 7x7 pixels, space_size=1.
        (The minimum bar width is 1 pixel, regardless of bar_w_frac.)

        :param space_size: int, width in pixels of a squares in the 3x3 game grid.
        :param bar_w_frac: float, width of the lines of the grid as a fraction of space_size.

        :returns: dict with
          'img_size': grid image's side length
          'line_t': line width for the 4 grid lines and the bounding box
          'upper': [(x1, y1), (x2, y2)] attachment points for the upper grid line (floats, non-integers if line_width is even)
          'lower': [(x1, y1), (x2, y2)] attachment points for the lower grid line
            'space_size': space_size
            'bar_w': grid line width
        """
        if GameStateArtist.get_size(self._space_size) == 'normal':
            grid_line_width = max(1, int(self._space_size * self._bar_w_frac))
        else:
            grid_line_width = 1

        img_side_len = self._space_size * 3 + grid_line_width * 4
        upper = [(self._space_size + 3 * grid_line_width / 2, grid_line_width),
                 (self._space_size * 2 + 5 * grid_line_width / 2, grid_line_width)]
        lower = [(upper[0][0], img_side_len-grid_line_width),
                 (upper[1][0], img_side_len-grid_line_width)]

        cell_x = (grid_line_width, grid_line_width + self._space_size)  # x range of first cell
        cell_y = (grid_line_width, grid_line_width + self._space_size)  # y range of first cell
        cell_x_offset = self._space_size + grid_line_width  # add one or two to cell_X to get the other cells
        cell_y_offset = self._space_size + grid_line_width  # add one or two to cell_Y to get the other cells

        # cell_span[row][col] = {'x': (x1, x2), 'y': (y1, y2)}
        cell_spans = [[{'x': (cell_x[0] + col * cell_x_offset, cell_x[0] + col * cell_x_offset + self._space_size),
                        'y': (cell_y[0] + row * cell_y_offset, cell_y[0] + row * cell_y_offset + self._space_size)}
                       for col in range(3)]
                      for row in range(3)]

        return {'img_size': img_side_len,
                'line_t': grid_line_width,
                'bar_w': grid_line_width,
                'upper': upper,
                'lower': lower,
                'cells': cell_spans}

    @staticmethod
    def get_space_size(img_size, bar_w_frac=.15):
        """
        Attempt to predict a good cell size for a given image size
        (i.e. inverse of get_image_dims).
        """
        space_size = img_size
        dims = GameStateArtist(space_size, bar_w_frac=bar_w_frac).dims
        while dims['img_size'] > img_size:
            space_size -= 1
            dims = GameStateArtist(space_size, bar_w_frac=bar_w_frac).dims
        return space_size

    def get_image(self, game):
        """
        Return an image of the game board & its dimension dictionary.

        :param dims: dict, output of get_image_dims
        :param color_bg: color of the background
        :param color_lines: color of the lines
        :param draw_box: draw a bounding box around the grid (with (non)terminal color)
            if None, only draw the box around terminal states

        """
        line_t = self.dims['line_t']
        img_s = self.dims['img_size']

        # Create the image
        img = np.zeros((img_s, img_s, 3), dtype=np.uint8)
        img[:, :] = COLOR_BG

        term = game.check_endstate()
        grid_line_color = COLOR_LINES

        size = GameStateArtist.get_size(self._space_size)

        # Draw grid lines
        if size != 'micro':
            for i in [1, 2]:
                # always dark or draw color
                line_color = grid_line_color if not (term is not None and term == Result.DRAW) else COLOR_DRAW
                z_0 = i * (self._space_size + line_t)
                z_1 = z_0 + line_t

                w0 = line_t
                w1 = img_s - line_t

                img[z_0:z_1, w0:w1] = line_color
                img[w0:w1, z_0:z_1] = line_color

        # Draw markers
        for i in range(3):
            for j in range(3):
                if game.state[i, j] == Mark.EMPTY:
                    continue
                self._add_marker(img, row=i, col=j, marker=game.state[i, j])

        # shade the whole image (including the bbox area) is averaged with the winner/draw color.
        shade_color = COLOR_SHADE if term is None else {Result.DRAW: COLOR_DRAW_SHADE,
                                                        Result.X_WIN: COLOR_X,
                                                        Result.O_WIN: COLOR_O}[term]
        weight = 0.25 if term is not None else .1
        tile = np.float32(img)
        shade_color = np.float32(shade_color)
        shaded_tile = tile * (1 - weight) + np.float32(shade_color) * weight
        img = np.uint8(shaded_tile)

        # Draw win lines, connecting a row of 3.
        if size in ['micro',  'tiny']:  # Skip for tiny images
            return img
        win_line_color = {Result.DRAW: COLOR_DRAW,
                          Result.X_WIN: COLOR_X,
                          Result.O_WIN: COLOR_O}[term] if term is not None else COLOR_LINES
        win_lines = self.get_win_lines(game)
        for line in win_lines:
            self.draw_win_line(img, line, win_line_color)

        return img

    def get_win_lines(self, game):
        """
        Get the winning lines for a game state.
        :param game: Game, the game state to check for winning lines
        :return: list of lines, each line is a dict with keys:
            'orient': one of 'h','v','d'
            'c1': (i,j) cell coordinates (0, 1 2)
            'c2': (i,j) cell coordinates
        """
        # Check for terminal state, and return empty list if not terminal.
        term = game.check_endstate()
        if term in [None, Result.DRAW]:
            return []
        winner_mark = WIN_MARKS[term]
        lines = []
        for i in range(3):
            # check rows
            if np.all(game.state[i, :] == winner_mark):
                lines.append({'orient': 'h', 'c1': (i, 0), 'c2': (i, 2)})
            # check columns
            if np.all(game.state[:, i] == winner_mark):
                lines.append({'orient': 'v', 'c1': (0, i), 'c2': (2, i)})
        # check diagonals
        if np.all(np.diag(game.state) == winner_mark):
            lines.append({'orient': 'd', 'c1': (0, 0), 'c2': (2, 2)})
        if np.all(np.diag(np.fliplr(game.state)) == winner_mark):
            lines.append({'orient': 'd', 'c1': (0, 2), 'c2': (2, 0)})
        return lines

    def draw_win_line(self, img, line, color):
        """
        Draw a line through three cells.
        :param img: np.array, image to draw on (single 3x3 game)
        :param dims: image dimensions dict, output of Game.get_image_dims(), used to create img
        :param line: element of list from Game._get_win_lines(), a dict with
            'orient': one of 'h','v','d'
            'c1': (i,j) cell coordinates (0, 1 2)
            'c2': (i,j) cell coordinates
        :param color: tuple, (r,g,b) color of the line
        """
        space_size = self._space_size

        cell1 = line['c1']
        cell2 = line['c2']
        cell_span1 = self.dims['cells'][cell1[0]][cell1[1]]
        cell_span2 = self.dims['cells'][cell2[0]][cell2[1]]

        size_type = GameStateArtist.get_size(space_size)
        aa = cv2.LINE_AA if size_type == 'normal' else cv2.LINE_4

        if line['orient'] == 'h':
            x0 = int((cell_span1['x'][0]) * _SHIFT)
            x1 = int((cell_span2['x'][1]) * _SHIFT)
            y = int(((cell_span1['y'][0] + cell_span2['y'][1]-1)/2) * _SHIFT)
            cv2.line(img, (x0, y), (x1, y), color, self.dims['line_t'], lineType=aa, shift=_SHIFT_BITS)
        elif line['orient'] == 'v':
            x = int(((cell_span1['x'][0] + cell_span1['x'][1]-1)/2) * _SHIFT)
            y0 = int((cell_span1['y'][0]) * _SHIFT)
            y1 = int((cell_span2['y'][1]) * _SHIFT)
            cv2.line(img, (x, y0), (x, y1), color, self.dims['line_t'], lineType=aa, shift=_SHIFT_BITS)
        elif line['orient'] == 'd':
            # Need to take different corners of the cells to get the diagonal
            if cell1[1] < cell2[1]:
                x0 = int((cell_span1['x'][0]) * _SHIFT)
                x1 = int((cell_span2['x'][1]) * _SHIFT)
                y0 = int((cell_span1['y'][0]) * _SHIFT)
                y1 = int((cell_span2['y'][1]) * _SHIFT)
            else:
                x0 = int((cell_span1['x'][1]) * _SHIFT)
                x1 = int((cell_span2['x'][0]) * _SHIFT)
                y0 = int((cell_span1['y'][0]) * _SHIFT)
                y1 = int((cell_span2['y'][1]) * _SHIFT)

            cv2.line(img, (x0, y0), (x1, y1), color, self.dims['line_t'], lineType=aa, shift=_SHIFT_BITS)
        return img

    def _add_marker(self, img, row, col, marker):
        """
        :param img: np.array, image to draw on
        :param dims: image dimensions dict, output of Game.get_image_dims()
        :param loc: tuple, (i,j) location in the 3x3 grid, (0,1,2) for each
        :param marker: str, "X", "O"
        """
        cell_span = self.dims['cells'][row][col]
        padding = self._space_size * .4
        size = GameStateArtist.get_size(self._space_size)

        def _draw_line(p0, p1, color, thickness):
            cv2.line(img,
                     (int(p0[0]*self._SHIFT), int(p0[1]*self._SHIFT)),
                     (int(p1[0]*self._SHIFT), int(p1[1]*self._SHIFT)),
                     color, thickness, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        def _circle_at(c, rad, color, thickness):
            cv2.circle(img, (int(c[0]*self._SHIFT), int(c[1]*self._SHIFT)),
                       int(rad*self._SHIFT), color, thickness=thickness,
                       lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        def get_X_points(pad_l):
            x0, x1 = cell_span['x'][0]+pad_l/2, cell_span['x'][1]-pad_l/2-1
            y0, y1 = cell_span['y'][0]+pad_l/2, cell_span['y'][1]-pad_l/2-1
            return x0, x1, y0, y1

        if size == 'normal':  # draw Normal

            if marker == Mark.X:
                x0, x1, y0, y1 = get_X_points(padding)
                x_thickness = int(self.dims['line_t'])

                _draw_line((x0, y0), (x1, y1), MARKER_COLORS[marker], x_thickness)
                _draw_line((x1, y0), (x0, y1), MARKER_COLORS[marker], x_thickness)

            else:
                center = get_cell_center(self.dims, (row, col))
                rad = (self._space_size/2)-padding/2 + (self._space_size//9)
                rad_inner = rad - (self.dims['line_t']*1)
                _circle_at(center, rad, MARKER_COLORS[marker], -1)
                _circle_at(center, rad_inner, COLOR_BG, -1)

        elif size == 'small':
            border = 1  # if self._space_size < 10 else 2
            # fill the cell with the color, leaving a border
            x_span, y_span = cell_span['x'], cell_span['y']
            img[y_span[0]+border:y_span[1]-border, x_span[0]+border:x_span[1]-border] = MARKER_COLORS[marker]

        else:  # size == 'micro'
            # fill the cell with the color
            x_span, y_span = cell_span['x'], cell_span['y']
            img[y_span[0]:y_span[1], x_span[0]:x_span[1]] = MARKER_COLORS[marker]

        return img

    def _get_o_points(self, size, line_width):

        if size not in self._point_cache:
            self._point_cache[size] = get_annulus_polyline(size, line_width, n_points=int(size*2))
        return self._point_cache[size]


def test_terminals():
    from tic_tac_toe import Game

    def show_strs(strs):
        game = Game.from_strs(strs)
        artist = GameStateArtist(50)
        img = artist.get_image(game)

        cv2.imshow("Tic Tac Toe", img[:, :, ::-1])
        cv2.waitKey(0)

    show_strs(["XOO",
               "OXX",
               "XOO"])
    show_strs(["XOO",
               "XXX",
               "XOO"])
    show_strs(["XOO",
               " OX",
               "XOO"])
    show_strs(["XXO",
               "XOX",
               "OOO"])
    show_strs(["XXO",
               "X X",
               "O O"])

    cv2.destroyAllWindows()


def test_img():
    from tic_tac_toe import Game
    game = Game()
    game.move(Mark.X, (0, 0))
    game.move(Mark.O, (1, 1))
    game.move(Mark.X, (2, 2))
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    game_imgb = game.get_img(space_size=11)
    # cv2.imshow("Tic Tac Toe", game_imgb[:,:,::-1])
    # cv2.waitKey(0)

    sizes = np.array([10, 20, 50])
    padding = 10
    widths = []
    for i, size in enumerate(sizes):
        x_offset = np.ceil(np.sum(widths) + i * padding).astype(int)
        game_img = game.get_img(space_size=size)
        w = game_img.shape[1]
        img[:w, x_offset:x_offset + w,] = game_img

        widths.append(w)

    cv2.imwrite("game.png", img)
    cv2.imshow("Tic Tac Toe", img[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_image():
    from tic_tac_toe import Game
    game = Game()
    game = Game.from_strs(["XOX",
                           "O X",
                           "OXO"])
    print(game)

    def print_dim_test(img_size):
        s_size = GameStateArtist.get_space_size(img_size)
        dims = GameStateArtist(s_size).dims
        print(f"Image size: {img_size}, space size: {s_size}, line_t: {dims['line_t']}")

    [print_dim_test(i) for i in [7, 25, 30, 35, 40, 45, 50, 100, 200, 500]]

    artist = GameStateArtist(20)
    print("Artist category:", artist.get_size(artist._space_size))
    print("Artist space size:", artist._space_size)
    img = artist.get_image(game,draw_box=True)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # test_img()
    # test_terminals()
    # get_draw()

    make_image()
