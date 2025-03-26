import numpy as np
from game_base import Mark
from util import get_annulus_polyline
import cv2
from colors import MPL_BLUE_RGB, MPL_GREEN_RGB, MPL_ORANGE_RGB

class MarkerArtist(object):
    """
    Draw "X", "O", circle, square, or "D" (for draw) on an image.
    Scale appropriately.
    """

    def __init__(self, color_x=MPL_BLUE_RGB, color_o=MPL_ORANGE_RGB, color_d=MPL_GREEN_RGB):
        self._font = cv2.FONT_HERSHEY_COMPLEX
        self._small_cutoff_cell_size = 15
        self.color_o = color_o
        self.color_x = color_x
        self.color_d = color_d
        self._SHIFT_BITS = 5
        self._SHIFT = 1 << self._SHIFT_BITS
        self._point_cache = {}  #

    def add_marker(self, img, center, space_size, marker, pad_frac=0.1):
        """
        :param img: np.array, image to draw on
        :param center: tuple, center of the marker
        :param space_size: int, number of pixels height/width of the grid cell (not counting lines)
        :param marker: str, "X", "O", "D"
        :param pad_frac: float, fraction of space_size to pad around the marker
        """
        x, y = center
        pad_size = int(space_size * pad_frac)
        marker_size = space_size - 2 * pad_size
        center_int = (int(x), int(y))

        if marker == "X":
            color = self.color_x
            dot_shape = 'x' if space_size >= self._small_cutoff_cell_size else 'square'
        elif marker == "O":
            color = self.color_o
            dot_shape = 'o' if space_size >= self._small_cutoff_cell_size else 'circle'
        elif marker == "D":
            dot_shape = 'd'
            color = self.color_d
            font_scale = space_size / 50
            font_thickness = max(1, int(space_size / 20))

        # half the width of the different markers:
        circle_rad = np.max([2, marker_size//1.6]).astype(int)
        box_rad = np.max([1, marker_size//2]).astype(int)
        x_rad = np.max([1, marker_size//2]).astype(int)
        line_width = np.max([1, marker_size//4]).astype(int)

        if dot_shape == 'square':
            x, y = int(x), int(y)
            img[y-box_rad:y+box_rad, x-box_rad:x+box_rad] = color

        elif dot_shape == 'circle':
            cv2.circle(img, center_int, circle_rad, color, -1)

        elif dot_shape == 'x':
            x, y = int(x), int(y)
            cv2.line(img, (x - x_rad, y - x_rad), (x + x_rad, y + x_rad), color, line_width)
            cv2.line(img, (x + x_rad, y - x_rad), (x - x_rad, y + x_rad), color, line_width)

        elif dot_shape == 'o':
            r_inner = circle_rad - line_width
            points = (self._get_o_points(circle_rad, r_inner) + np.array(center).T) * self._SHIFT
            cv2.fillPoly(img, [points.astype(np.int32)], color, lineType=cv2.LINE_AA, shift=self._SHIFT_BITS)

        elif dot_shape == 'd':
            (width, height), baseline = cv2.getTextSize("D", self._font, font_scale, font_thickness)
            t_x = center[0] - width//2
            t_y = center[1] + height//2
            pos = int(t_x), int(t_y)
            cv2.putText(img, "D", pos, self._font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    def _get_o_points(self, size, line_width):
        if size not in self._point_cache:
            self._point_cache[size] = get_annulus_polyline(size, line_width)
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
               "XOO"], bbox = True)
    show_strs(["XOO",
               "OXX",
               "XOO"], bbox = False)
    show_strs(["XOO",
               "XXX",
               "XOO"], bbox = True)
    show_strs(["XOO",
               "XXX",
               "XOO"], bbox = False)
    show_strs(["XOO",
               " OX",
               "XOO"], bbox = True)
    show_strs(["XOO",
               " OX",
               "XOO"], bbox = False)
    show_strs(["XXO",
               "XOX",
               "OOO"], bbox = True)
    show_strs(["XXO",
               "XOX",
               "OOO"], bbox = False)
    
    show_strs(["XXO",
               "X X",
               "O O"], bbox = True)
    show_strs(["XXO",
               "X X",
               "O O"], bbox = False)
    
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
        game_img = game.get_img(space_size=size, marker_padding_frac=0.15)
        w = game_img.shape[1]
        img[:w, x_offset:x_offset + w,] = game_img
        print(game_img.dtype)
        widths.append(w)

    cv2.imwrite("game.png", img)
    cv2.imshow("Tic Tac Toe", img[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    #test_img()
    test_terminals()
    # get_draw()
