import numpy as np
from game_base import Mark, Result, WIN_MARKS, get_cell_center
from util import get_annulus_polyline, float_rgb_to_int
import cv2
from colors import COLOR_SCHEME
import matplotlib.pyplot as plt
from tic_tac_toe import Game

MARKER_COLORS = {Mark.X: COLOR_SCHEME['color_x'],
                 Mark.O: COLOR_SCHEME['color_o'],
                 Mark.EMPTY: COLOR_SCHEME['bg']}

_SHIFT_BITS = 5
_SHIFT = 1 << _SHIFT_BITS


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

        * "micro" sized images: (size < 5)
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

    _SHIFT_BITS = 5
    _SHIFT = 1 << _SHIFT_BITS

    @staticmethod
    def get_size(space_size):
        if space_size >= 11:
            return 'normal'
        elif space_size >= 5:
            return 'small'
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

    def get_action_dist_image(self, action_dist, player_mark, cmap, highlight_choice=None, alpha=1.0, highlight_color=None):
        """
        instead of orange/blue, for nonzero probability actions, use the colormap to render
        the mark by the color determined by the action probability.
        :param action_dist: list of tuples (action, probability), where action is a tuple (row, col)
        :param player_mark: Mark, the player's mark (X or O)
        :param cmap: colormap, a matplotlib colormap to use for rendering the action probabilities
        :param highlight_choice: index into action_dist, which action was chosen, to outlined.
        :returns: list of (bounding_box, probability) tuples for the color key mouseover action.
        """
        highlight_color = COLOR_SCHEME['highlight'] if highlight_color is None else highlight_color

        size = GameStateArtist.get_size(self._space_size)

        img = self._get_blank()
        self._draw_grid_lines(img, term=None)

        action_to_prob = {action: (prob, i) for i, (action, prob) in enumerate(action_dist)}
        bboxes = []
        # Draw markers
        for i in range(3):
            for j in range(3):
                action = (i, j)
                if action not in action_to_prob:
                    continue
                prob, action_ind = action_to_prob[action]
                color = float_rgb_to_int(np.array(cmap(1-prob))**alpha)  # get the RGB color from the colormap
                highlight = (highlight_choice == action_ind) if highlight_choice is not None else False
                h_col = None if (not highlight or len(action_dist) == 1) else highlight_color
                bbox = self._add_marker(img, row=i, col=j, marker=player_mark, color=color,
                                        highlight_color=h_col)
                bboxes.append((bbox, prob))

        return img, bboxes

    def _draw_grid_lines(self, img, term):
        grid_line_color = COLOR_SCHEME['lines']

        size = GameStateArtist.get_size(self._space_size)

        thickness = self.dims['line_t']
        img_s = self.dims['img_size']
        # Draw grid lines
        if size != 'micro':
            for i in [1, 2]:
                # always dark or draw color
                line_color = grid_line_color if not (term is not None and term ==
                                                     Result.DRAW) else COLOR_SCHEME['color_draw']
                z_0 = i * (self._space_size + thickness)
                z_1 = z_0 + thickness

                w0 = thickness
                w1 = img_s - thickness

                img[z_0:z_1, w0:w1] = line_color
                img[w0:w1, z_0:z_1] = line_color

    def _get_blank(self):
        img_s = self.dims['img_size']

        # Create the image
        img = np.zeros((img_s, img_s, 3), dtype=np.uint8)
        img[:, :] = COLOR_SCHEME['bg']

        return img

    def get_image(self, game, highlight_cell=None, alpha=1.0, highlight_color=None):
        """
        Return an image of the game board & its dimension dictionary.

        :param dims: dict, output of get_image_dims
        :param draw_box: draw a bounding box around the grid (with (non)terminal color)
            if None, only draw the box around terminal states

        """
        highlight_color = COLOR_SCHEME['highlight'] if highlight_color is None else highlight_color

        size = GameStateArtist.get_size(self._space_size)

        img = self._get_blank()
        term = game.check_endstate()
        self._draw_grid_lines(img, term)

        # Draw markers
        for i in range(3):
            for j in range(3):
                no_marker = False
                if game.state[i, j] == Mark.EMPTY:
                    no_marker = True
                h_col = highlight_color if highlight_cell is not None and (i, j) == highlight_cell else None
                self._add_marker(img, row=i, col=j, marker=game.state[i, j], highlight_color=h_col, no_marker=no_marker)

        # shade the whole image (including the bbox area) is averaged with the winner/draw color.
        shade_color = COLOR_SCHEME['color_shade'] if term is None else {Result.DRAW: COLOR_SCHEME['color_draw_shade'],
                                                                        Result.X_WIN: COLOR_SCHEME['color_x'],
                                                                        Result.O_WIN: COLOR_SCHEME['color_o']}[term]
        weight = 0.25 if term is not None else .1
        tile = np.float32(img)
        shade_color = np.float32(shade_color)
        shaded_tile = tile * (1 - weight) + np.float32(shade_color) * weight
        img = np.uint8(shaded_tile)

        # Draw win lines, connecting a row of 3.
        if size in ['micro',  'tiny']:  # Skip for tiny images
            return img
        win_line_color = {Result.DRAW: COLOR_SCHEME['color_draw'],
                          Result.X_WIN: COLOR_SCHEME['color_x'],
                          Result.O_WIN: COLOR_SCHEME['color_o']}[term] if term is not None else COLOR_SCHEME['lines']
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

    def _add_marker(self, img, row, col, marker, color=None, highlight_color=None, no_marker=False):
        """
        :param img: np.array, image to draw on
        :param dims: image dimensions dict, output of Game.get_image_dims()
        :param loc: tuple, (i,j) location in the 3x3 grid, (0,1,2) for each
        :param marker: str, "X", "O"
        returns: bounding box of cell marker was drwan in.
        """
        cell_span = self.dims['cells'][row][col]
        color = MARKER_COLORS[marker] if color is None else color
        line_t = self.dims['line_t']
        GameStateArtist.draw_mark(img, cell_span, line_t, marker, color,
                                  highlight_color=highlight_color, no_marker=no_marker)
        return cell_span

    @staticmethod
    def draw_mark(img, bbox, line_t, marker, color, highlight_color=None, no_marker=False):
        space_size = bbox['x'][1] - bbox['x'][0]
        y_size = bbox['y'][1] - bbox['y'][0]
        space_size = min(space_size, y_size)  # use the smaller dimension
        padding = space_size * .4

        size = GameStateArtist.get_size(space_size)

        def _circle_at(c, rad, color, thickness):
            cv2.circle(img, (int(c[0]*GameStateArtist._SHIFT), int(c[1]*GameStateArtist._SHIFT)),
                       int(rad*GameStateArtist._SHIFT), color, thickness=thickness,
                       lineType=cv2.LINE_AA, shift=GameStateArtist._SHIFT_BITS)

        def get_X_points(pad_l):
            x0, x1 = bbox['x'][0]+pad_l/2, bbox['x'][1]-pad_l/2-1
            y0, y1 = bbox['y'][0]+pad_l/2, bbox['y'][1]-pad_l/2-1
            return x0, x1, y0, y1

        def _draw_line(p0, p1, color, thickness):
            cv2.line(img,
                     (int(p0[0]*GameStateArtist._SHIFT), int(p0[1]*GameStateArtist._SHIFT)),
                     (int(p1[0]*GameStateArtist._SHIFT), int(p1[1]*GameStateArtist._SHIFT)),
                     color, thickness, lineType=cv2.LINE_AA, shift=GameStateArtist._SHIFT_BITS)

        if size == 'normal':  # draw Normal

            x_thickness = int(line_t)
            if x_thickness > 2:
                x_thickness = x_thickness - 1

            circle_thickness = x_thickness*1.4  # if x_thickness>2 else x_thickness*1.35
            center = (bbox['x'][0] + (bbox['x'][1]-1)) / 2, (bbox['y'][0] + (bbox['y'][1]-1)) / 2
            rad = (space_size/2)-padding/2 + (space_size/17)
            rad_inner = rad - circle_thickness

            rad_highlight = max(2, rad_inner-1)

            if not no_marker:
                if marker == Mark.X:
                    x0, x1, y0, y1 = get_X_points(padding)
                    _draw_line((x0, y0), (x1, y1), color, x_thickness)
                    _draw_line((x1, y0), (x0, y1), color, x_thickness)
                else:
                    _circle_at(center, rad, color, -1)
                    _circle_at(center, rad_inner, COLOR_SCHEME['bg'], -1)

            if highlight_color is not None:
                if size == 'normal':
                    # draw a highlight circle around the marker
                    _circle_at(center, rad_highlight, highlight_color, -1)
                else:
                    raise ValueError("Highlight color not supported for normal images.")

        elif size == 'small':
            border = 1  # if self._space_size < 10 else 2
            # fill the cell with the color, leaving a border
            x_span, y_span = bbox['x'], bbox['y']
            if not no_marker:
                img[y_span[0]+border:y_span[1]-border, x_span[0]+border:x_span[1]-border] = color
            if highlight_color is not None:
                img[y_span[0]+border:y_span[1]-border, x_span[0]+border:x_span[1]-border] = highlight_color

        else:  # size == 'micro'
            # fill the cell with the color
            x_span, y_span = bbox['x'], bbox['y']
            if not no_marker:
                img[y_span[0]:y_span[1], x_span[0]:x_span[1]] = color
            if highlight_color is not None:
                img[y_span[0]:y_span[1], x_span[0]:x_span[1]] = highlight_color

    def _get_o_points(self, size, line_width):

        if size not in self._point_cache:
            self._point_cache[size] = get_annulus_polyline(size, line_width, n_points=int(size*2))
        return self._point_cache[size]

def place_string(pos, string, font, scale, thickness=1, incl_baseline=True, t_dims=None):
    """
    A string is to be added to an image below and right of 'pos'.
    Determine the text's position, and where the new "top" of the image is.
    :param pos: (x, y) position to place the string.
    :param string: string to place.
    :param font: cv2 font constant.
    :param scale: font scale.
    :param thickness: thickness of the text.
    :param incl_baseline: whether to include the baseline in the new "top" of the image.
    :param t_dims: if given, use these dimensions instead of recalculating them:
       (width, height), baseline  # return value of cv2.getTextSize
       NOTE:  If using, font, scale,thickness can be None
    :returns: (x, y): position of the text, 
               (y_top, x+width): new top y position, position right of text
    """
    if t_dims is None:
        (text_width, text_height), baseline = cv2.getTextSize(string, font, scale, thickness)
    else:
        (text_width, text_height), baseline = t_dims

    x_pos = pos[0]
    y_pos = pos[1] + text_height
    y_top = y_pos + (baseline if incl_baseline else 0)
    return (x_pos, y_pos), (y_top, x_pos + text_width)

def test_place_string():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:] = COLOR_SCHEME['bg']

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    string = "T'j(}"

    y_top = 20
    pos = (20, y_top)
    test_string = string + "no baseline"
    (x_pos, y_pos), (y_top, _) = place_string(pos, test_string, font, scale, incl_baseline=False)
    cv2.putText(img, test_string, (x_pos, y_pos), font, scale, COLOR_SCHEME['lines'], 1, cv2.LINE_AA)
    cv2.rectangle(img, (pos[0], pos[1]), (pos[0]+200, y_top), COLOR_SCHEME['lines'], 1, cv2.LINE_AA)

    y_top += 12
    pos = (20, y_top)
    test_string = string + "w/baseline"
    (x_pos, y_pos),  (y_top, _) = place_string(pos, test_string, font, scale, incl_baseline=True)
    cv2.putText(img, test_string, (x_pos, y_pos), font, scale, COLOR_SCHEME['lines'], 1, cv2.LINE_AA)
    cv2.rectangle(img, (pos[0], pos[1]), (pos[0]+200, y_top), COLOR_SCHEME['lines'], 1, cv2.LINE_AA)
    
    cv2.imshow('Test Place String', img[:,:,::-1])
    cv2.waitKey(0)

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


def make_single_image(game=None, space_size=20):
    artist = GameStateArtist(space_size)
    game = Game.from_strs(["XOX",
                           "O X",
                           "OXO"]) if game is None else game
    print("Generating image for game:\n%s" % game)
    img = artist.get_image(game)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    test_place_string()
    test_terminals()
    make_single_image()
