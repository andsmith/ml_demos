import numpy as np
from game_base import Mark, get_cell_center
from util import get_annulus_polyline
import cv2
from colors import COLOR_X, COLOR_O, COLOR_BG

_SHIFT_BITS = 5
_SHIFT = 1 << _SHIFT_BITS


def draw_win_line(img, dims, line, color):
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
    space_size = dims['space_size']

    cell1 = line['c1']
    cell2 = line['c2']
    cell_span1 = dims['cells'][cell1[0]][cell1[1]]
    cell_span2 = dims['cells'][cell2[0]][cell2[1]]

    if line['orient'] == 'h':
        x0 = int((cell_span1['x'][0]) * _SHIFT)
        x1 = int((cell_span2['x'][1]) * _SHIFT)
        y = int(((cell_span1['y'][0] + cell_span2['y'][1]-1)/2) * _SHIFT)
        cv2.line(img, (x0, y), (x1, y), color, dims['line_t'], lineType=cv2.LINE_AA, shift=_SHIFT_BITS)
    elif line['orient'] == 'v':
        x = int(((cell_span1['x'][0] + cell_span1['x'][1]-1)/2) * _SHIFT)
        y0 = int((cell_span1['y'][0]) * _SHIFT)
        y1 = int((cell_span2['y'][1]) * _SHIFT)
        cv2.line(img, (x, y0), (x, y1), color, dims['line_t'], lineType=cv2.LINE_AA, shift=_SHIFT_BITS)
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

        cv2.line(img, (x0, y0), (x1, y1), color, dims['line_t'], lineType=cv2.LINE_AA, shift=_SHIFT_BITS)
    return img


_CUTOFFS = {'min_normal': 7,
            'min_small': 8}  # no small if > min_normal
def get_size(space_size):
    if space_size >= _CUTOFFS['min_normal']:
        return 'normal'
    elif space_size >= _CUTOFFS['min_small']:
        return 'small'
    else:
        return 'tiny'


class MarkerArtist(object):
    """
    Draw X O Markers on the image:
        * Normal sized markers:  X and O with scaled thickness  (space size >= 10)
        * Small markers: X and O with single-pixel lines (5 <= space_size < 10)
        * Tiny markers: fill the cell completely with the color   (space_size < 5)

    Marker size is determined by the space size it's being drawn in.
    This value should be in dims['space_size'].

    Normal sized markers will be padded by empty space, determined by dims['marker_padding_frac'].

    """

    def __init__(self, color_x=COLOR_X, color_o=COLOR_O):
        self._font = cv2.FONT_HERSHEY_COMPLEX
        self.colors = {Mark.O: color_o,
                       Mark.X: color_x}
        self._SHIFT_BITS = 5
        self._SHIFT = 1 << self._SHIFT_BITS
        self._point_cache = {}  # The "O" polyline to fill, for each value of dims['space_size']

    def add_marker(self, img, dims, loc, marker):
        """
        :param img: np.array, image to draw on
        :param dims: image dimensions dict, output of Game.get_image_dims()
        :param loc: tuple, (i,j) location in the 3x3 grid, (0,1,2) for each
        :param marker: str, "X", "O"
        """
        cell_span = dims['cells'][loc[0]][loc[1]]
        space_size = dims['space_size']
        padding_px = int(space_size * dims['marker_padding_frac'])

        size = get_size(space_size)

        if size=='normal':  # draw Normal
            if marker == Mark.X:

                x0, x1 = cell_span['x'][0]+padding_px//2, cell_span['x'][1]-padding_px//2-1
                y0, y1 = cell_span['y'][0]+padding_px//2, cell_span['y'][1]-padding_px//2-1
                cv2.line(img, (x0, y0), (x1, y1), self.colors[marker], dims['line_t'], lineType=cv2.LINE_AA)
                cv2.line(img, (x1, y0), (x0, y1), self.colors[marker], dims['line_t'], lineType=cv2.LINE_AA)

            else:
                center = get_cell_center(dims, loc)
                x, y = center
                rad = (space_size/2)-padding_px/2 + (space_size//9)
                rad_inner = rad - (dims['line_t']*1)
                cv2.circle(img, (int(x*self._SHIFT), int(y*self._SHIFT)), int(rad*self._SHIFT),
                           self.colors[marker], thickness=-1, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)
                cv2.circle(img, (int(x*self._SHIFT), int(y*self._SHIFT)), int(rad_inner*self._SHIFT),
                           COLOR_BG, thickness=-1, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        elif size=='small':
            # Draw single-pixel wide lines with 1- or 2-pixel  margin.

            m_pix = 1 if dims['space_size'] < (_CUTOFFS['min_normal']+_CUTOFFS['min_small'])/2 else 2

            center = get_cell_center(dims, loc)
            x, y = center
            if marker == Mark.X:
                x0, x1 = cell_span['x'][0]+m_pix, cell_span['x'][1]-1-m_pix
                y0, y1 = cell_span['y'][0]+m_pix, cell_span['y'][1]-1-m_pix

                cv2.line(img, (x0, y0), (x1, y1), self.colors[marker], 2)
                cv2.line(img, (x1, y0), (x0, y1), self.colors[marker], 2)

            else:
                center = get_cell_center(dims, loc)
                x, y = center
                rad = (space_size/2)-m_pix
                cv2.circle(img, (int(x*self._SHIFT), int(y*self._SHIFT)), int(rad*self._SHIFT),
                           self.colors[marker], shift=self._SHIFT_BITS, lineType=cv2.LINE_AA)

        else:  # size == 'tiny'
            # fill the cell with the color
            x_span, y_span = cell_span['x'], cell_span['y']
            img[y_span[0]:y_span[1], x_span[0]:x_span[1]] = self.colors[marker]
        return img

    def _get_o_points(self, size, line_width):

        if size not in self._point_cache:
            self._point_cache[size] = get_annulus_polyline(size, line_width, n_points=int(size*2))
        return self._point_cache[size]


def test_terminals():
    from tic_tac_toe import Game

    def show_strs(strs, bbox):
        game = Game.from_strs(strs)
        img = game.get_img(space_size=50, draw_box=bbox)
        cv2.imshow("Tic Tac Toe", img[:, :, ::-1])
        cv2.waitKey(0)

    show_strs(["XOO",
               "OXX",
               "XOO"], bbox=True)
    show_strs(["XOO",
               "OXX",
               "XOO"], bbox=False)
    show_strs(["XOO",
               "XXX",
               "XOO"], bbox=True)
    show_strs(["XOO",
               "XXX",
               "XOO"], bbox=False)
    show_strs(["XOO",
               " OX",
               "XOO"], bbox=True)
    show_strs(["XOO",
               " OX",
               "XOO"], bbox=False)
    show_strs(["XXO",
               "XOX",
               "OOO"], bbox=True)
    show_strs(["XXO",
               "XOX",
               "OOO"], bbox=False)

    show_strs(["XXO",
               "X X",
               "O O"], bbox=True)
    show_strs(["XXO",
               "X X",
               "O O"], bbox=False)

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


if __name__ == "__main__":
    # test_img()
    test_terminals()
    # get_draw()
