"""
visualizations for RL steps (state/epoch/policy) that are portions of a game tree

 +------------------+--------------+
 |   TITLE          |  ALGORITHM   |
 |                  |  STATUS      |
 |  Turn: 3         |  AREA  (key) |
 |  +-----+         +--------------+
 |  |state| V(s)                   |
 |  +-----+                        |
 |   | |                           |  # separation space
 |   | |        p(a1|s)            |
 |   | |       +-------+           |
 |   | +-------|action1|           |
 |   |         +-------+           |
 |   |         /   |   \           |
 |   |        /    |    \          |  # child arrow space
 |   |       /     |     \         |
 |   |  p(s1|a,s) p(...)  p(...)   |
 |   |  +-----+  +-----+  +-----+  |
 |   |  |s1   |  |s'2  |  |s'3  |  |
 |   |  +-----+  +-----+  +-----+  |   # caption space
 |   |  V(s'1)   V(s'2)   V(s'3)   |
 |   |                             |  # sep space
 |   |          p(a2|s)            |   }
 |   |         +-------+           |   }  {action unit, 0-9 of them exist per state}
 |   +---------|action2|           |   }
 |             +-------+           |   }
 |               /    \            |   }
 |              /      \           |   }
 |             /        \          |   }
 |        p(s'1|a,s)  p(...)       |   }
 |         +-----+   +-----+       |   }
 |         |s'1  |   |O-WIN|       |   }
 |         +-----+   +-----+       |   }
 |         V(s'1)    R=-1.0        |   }
 |                                 |  # sep space
 +---------------------------------+
 implementation plan:
 1. calculate title size, get State placement
 2. lowest(alg status area, title area) -> top of action units, determin action unit size and separation space size
 3. parent arrows extend from MIDDLE of state for the top action, each subsequent action's arrow from the LEFT of the previous.
 4. Action units:
     a)  determine vertical space taken by:
          - states (fixed)
          - captions (scaled)
          - arrows (scaled)
 """

import logging
import numpy as np
import cv2
from tic_tac_toe import Game, Mark, Result
from game_base import TERMINAL_REWARDS
from colors import COLOR_SCHEME, MPL_CYCLE_COLORS
from drawing import GameStateArtist, place_string
import abc
from tab_content import TabContentPage
from layout import LAYOUT, SHIFT_BITS, SHIFT_MUL

_DEFAULT_PARAMS = {'space_sizes': {
    'state': 35,
    'action': 25,
    'next_state': 30},
    'rel_dims': {'pad_frac': (0.01, 0.05),  # W, H
                 'sep_y_space': .1,  # fraction of action unit height
                 'caption_y_space': .15},  # fraction of action unit height
    'string_v_spacing': 1.5,  # mult of str height, move strings DOWN this much (1.0=no spacing)

}


class CaptionedTile(object):
    """
    Small image with optional text above/below,
    Anchor points for connecting arrows,
    Bounding box for mouseover capacity.
    """

    def __init__(self, tile_img, above_cap=None, below_cap=None, pad_px=2, font_scale=None, str_dims=None):
        """
        :param tile_img: image to use as the tile.
        :param above_cap: text to draw above the tile.
        :param below_cap: text to draw below the tile.
        :param pad_px: padding around the tile image.
        """
        self.tile_img = tile_img
        self.tile_size = tile_img.shape[1], tile_img.shape[0]  # W, H
        self.above_cap = above_cap
        self.below_cap = below_cap
        self._pad_px = pad_px
        self._font = LAYOUT['cv2_fonts']['state_captions']['font']
        self._font_scale = font_scale if font_scale is not None else LAYOUT['cv2_fonts']['state_captions']['scale']
        test_str = above_cap if above_cap is not None else (below_cap if below_cap is not None else 'X')
        self._str_dims = cv2.getTextSize(test_str, self._font, self._font_scale, 1) if str_dims is None else str_dims
        self._v_spacing = int(self._str_dims[0][1]*LAYOUT['cv2_fonts']['state_captions']['v_spacing'])
        print("Font scale:  %f, v_spacing: %i" % (self._font_scale, self._v_spacing))
        self.size, self._tile_bbox, self._captions, self._y_bottom = self._calc_dims()

    def get_attach_points(self, pos=(0, 0), n=8, loc='bottom-left'):
        """
        Equally distributed in specified location:
          - bottom-left:  from the left edge to the middle (for the "state" tile).
          - bottom: left to right of bottom edge. (for action tiles)
          - left:  top to bottom of the left edge. (action tiles)
          - top: middle, above upper caption.  (next-state tiles)
        :param loc: position to attach to, one of 'bottom-left', 'bottom', 'left', 'top'.
        :param n: number of points to return.
        :param pos: (x, y) offset to apply to the points.
        :returns: list of (x, y) points. 
        """
        if loc == 'bottom-left':
            x = np.linspace(self._tile_bbox['x'][0], self._tile_bbox['x']
                            [1], n+2)[1:-1]  # skip the first and last point
            y = np.full(n,self.size[1])
        elif loc == 'bottom':
            x = np.linspace(self._tile_bbox['x'][0], self._tile_bbox['x']
                            [1], n+2)[1:-1]  # skip the first and last point
            y = np.full(n, self.size[1])
        elif loc == 'left':
            x = np.full(n, self._tile_bbox['x'][0])
            y = np.linspace(self._tile_bbox['y'][0], self._tile_bbox['y']
                            [1], n+2)[1:-1]  # skip the first and last point
        elif loc == 'top':
            x = np.linspace(self._tile_bbox['x'][0], self._tile_bbox['x']
                            [1], n+2)[1:-1]  # skip the first and last point
            y = np.full(n, 0)
        else:
            raise ValueError(f"Invalid attach position: {loc}")
        return list(zip(x+pos[0], y+pos[1]))

    def _calc_dims(self):
        """
        Determine sizes. 
        set x/y coordinates for everything. 
        """
        w = self.tile_size[0] + self._pad_px * 2
        h = self.tile_size[1] + self._pad_px * 2
        y_top = self._pad_px + self._v_spacing 
        x_left = self._pad_px
        captions = []
        # import ipdb; ipdb.set_trace()
        
        if self.above_cap is not None:
            cap_pos, (y_top, _) = place_string((x_left, y_top), self.above_cap, None, None,
                                               None, incl_baseline=True, t_dims=self._str_dims)
            captions.append((cap_pos, self.above_cap))
            y_top += self._v_spacing //2
        img_pos = (x_left, y_top)
        y_top += self.tile_size[1] + self._v_spacing 
        self._img_bottom = y_top
        if self.below_cap is not None:
            cap_pos, (y_top, _) = place_string((x_left, y_top), self.below_cap, None, None,
                                               None, incl_baseline=True, t_dims=self._str_dims)
            captions.append((cap_pos, self.below_cap))
            y_top += self._v_spacing
            
        self._bottom_space = 0# self._v_spacing //2
        size = (w, y_top +self._bottom_space)
        tile_bbox = {'x': (img_pos[0], img_pos[0] + self.tile_size[0]),
                     'y': (img_pos[1], img_pos[1] + self.tile_size[1])}
        return size, tile_bbox, captions, y_top

    def draw(self, img, pos):
        """
        Add the captioned tile to the image at the given position.
        :param img: image to draw on.
        :param pos: (x, y) position to draw the tile.
        :returns: bounding box of tile-image:
           tile: {'x': (x0, x1), 'y': (y0, y1)},
        """
        def _offset_pos(p_xy):
            return (p_xy[0] + pos[0], p_xy[1] + pos[1])

        def _offset_bbox(bbox):
            return {'x': (bbox['x'][0] + pos[0], bbox['x'][1] + pos[0]),
                    'y': (bbox['y'][0] + pos[1], bbox['y'][1] + pos[1])}

        for cap_pos, cap_str in self._captions:
            cap_pos = _offset_pos(cap_pos)
            cv2.putText(img, cap_str, cap_pos, self._font, self._font_scale,
                        COLOR_SCHEME['text'], 1, cv2.LINE_AA)

        tile_bbox = _offset_bbox(self._tile_bbox)
        img[tile_bbox['y'][0]:tile_bbox['y'][1],
            tile_bbox['x'][0]:tile_bbox['x'][1]] = self.tile_img

        return tile_bbox, pos[1]+self._y_bottom + self._bottom_space


def test_captioned_tile():
    from tic_tac_toe import Game
    from game_base import Mark
    from drawing import GameStateArtist
    from colors import COLOR_SCHEME
    from resize_test import ResizingTester

    game = Game.from_strs(["XOX",
                           " OX",
                           "X O"])

    def frame_factory(size):
        pad = 20
        test_img_size = (size[0]-pad*2, size[1]-pad*2)
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = COLOR_SCHEME['bg']

        space_size = test_img_size[0]//8

        artist = GameStateArtist(space_size=space_size)
        ct = CaptionedTile(artist.get_image(game), above_cap='P(s): 1.0', below_cap=None)#'v(s): 0.24253')

        n_lower_attach = (size[0]//30 % 7+1)  # Change width to try different numbers of lower attach points

        # draw the tile
        pos = (pad, pad)
        tile_bbox, y_bottom = ct.draw(img, pos)

        # draw some attachment points
        def _draw_attach(points, color):
            for point in points:
                pt =int(point[0]), int(point[1]) # (int(point[0]*SHIFT_MUL), int(point[1]*SHIFT_MUL))
                cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)#, shift=SHIFT_BITS)

        upper_attach = ct.get_attach_points(pos, n=1, loc='top')
        lower_attach = ct.get_attach_points(pos, n=n_lower_attach, loc='bottom')
        left_attach = ct.get_attach_points(pos, n=1, loc='left')
        _draw_attach(upper_attach, (0, 255, 0))
        _draw_attach(lower_attach, (0, 0, 255))
        _draw_attach(left_attach, (255, 0, 0))
        # Draw bbox around tile part
        p0 = tile_bbox['x'][0], tile_bbox['y'][0]
        p1 = tile_bbox['x'][1], tile_bbox['y'][1]
        cv2.rectangle(img, p0, p1, (255, 0, 0), 1)

        
        x_left, x_right = pos[0], pos[1] + ct.size[0]
        p0 = x_left, pos[1]
        p1 = x_right, pos[1] + ct.size[1]
        cv2.line(img, (0, y_bottom), (size[0], y_bottom), (0, 255, 0), 1)
        cv2.line(img, (0, pos[1]), (size[0], pos[1]), (0, 255, 0), 1)
        cv2.rectangle(img, p0, p1, (0, 255, 0), 2)

        return img

    rt = ResizingTester(frame_factory, (649, 480))
    rt.start()


class ValFuncViz(object):
    """
    When mousing over a state, this image shows the value function & any current update
    for the moused-over state.

    This class generates the image.

    """

    def __init__(self, env, policy, values, size, key_size=None, title='Value Function', draw_params=None):
        """
        Show value function, actions & child states for the given state.
        :param state: The state to visualize.
        :param size: Size of the image to generate.
        :param key_size: (w,h), reserve this much area (expect to be overwritten) in the top right.
        """
        self._fonts = LAYOUT['cv2_fonts']
        self.size = size
        self.env = env
        self.policy = policy
        self.values = values
        self._title = title
        self.key_size = key_size if key_size is not None else (0, 0)
        self._par = ValFuncViz._DEFAULT_PARAMS.copy()
        if draw_params is not None:
            self._par.update(draw_params)
        self._dims, self._bboxes = self._calc_dims()

    def _calc_dims(self):
        """
        Calculate what we can ahead of time (before knowing the state we're drawing):
            - State area:   Title, value(s), turn #

        returns: {'state': (bbox),
                  'action_bbox': (bbox),}
        """
        x_marg, y_marg = LAYOUT['margin_rel'] * np.array(self.size)
        x_left, x_right = x_marg, self.size[0] - x_marg - self.key_size[0]
        y_top, y_bottom = y_marg, self.size[1] - y_marg
        artists = {kind: GameStateArtist(space_size=self._par['space_sizes'][kind])
                   for kind in ['state', 'action', 'next_state']}

        # Title area
        title_right = min(x_right, self.key_size[0] + x_marg)
        title_left = x_left
        title_top = y_top + int(y_marg * (self._par['string_v_spacing'] - 1))
        title_pos, y_top = place_string((title_left, title_top), self._title,
                                        self._fonts['title']['font'], self._fonts['title']['scale'],
                                        incl_baseline=True)
        # State icon (CaptionedTile)
        tile_sizes = {kind: artist.dims['img_size'] for kind, artist in artists.items()}

        state_tile = CaptionedTile(artists['state'].get_image(None), above_cap='(set later)', below_cap=None)

        # everything but arrows must be left of the right-most arrow possible
        left_edge_x = state_tile.get_attach_points(n=8, pos='bottom-left')[-1][0]

        #

    def draw(self, state):
        self.artist = GameStateArtist(state, size=self.size, key_size=self.key_size)


def test_val_func_viz():
    img_size = (630, 950)
    key_size = (300, 100)
    from reinforcement_base import Environment
    from baseline_players import HeuristicPlayer
    from game_base import Mark
    from mouse_state_manager import MouseBoxManager
    agent_policy = HeuristicPlayer(mark=Mark.X, n_rules=1)
    opponent_policy = HeuristicPlayer(mark=Mark.O, n_rules=1)
    env = Environment(opponent_policy=opponent_policy, player_mark=Mark.X)

    terminals, nonterminals = env.get_terminal_states(), env.get_nonterminal_states()
    values = {state: TERMINAL_REWARDS[state.result] for state in terminals}
    values.update({state: np.random.randn() for state in nonterminals})

    # s val_func_viz = ValFuncViz(state=test_games[0],

    test_games = [Game.from_strs(["XOX",
                                  " OX",
                                 "X O"]),

                  Game.from_strs(["   ",
                                  "   ",
                                 "   "]),

                  Game.from_strs(["XOX",
                                  " OX",
                                 "X O"])]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #
    test_captioned_tile()
