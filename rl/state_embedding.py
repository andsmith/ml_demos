import logging
import numpy as np
import cv2
from layout import LAYOUT
from game_util import get_box_placer, sort_states_into_layers
from util import get_font_scale, tk_color_from_rgb
from mouse_state_manager import MouseBoxManager
from state_key import StateKey
from color_key import ColorKey, ProbabilityColorKey
from drawing import GameStateArtist
from tic_tac_toe import Game
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
from gameplay import ResultSet, Match
from colors import COLOR_SCHEME
from layer_optimizer import SimpleTreeOptimizer
from gui_base import Key, KeySizeTester

SPACE_SIZES = [7, 2, 2, 2, 2, 3]  # Sizes for the state embedding layers


class StateEmbedding(object):
    """
    Determine a good location for all reachable (RL) game states for player X given the image size.

    Resize this when the tab panel changes size.
    StateArtists (and subclasses) look up size from this class to draw images.

    """

    def __init__(self, env, key_size=(0, 0)):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)
        :param key_size:  (width, height) tuple, space reserved in the upper right corner for keys.

        """
        self.size = None
        self._env = env
        self.box_placer = None
        self._layout = LAYOUT['state_embedding']
        self.key_size = key_size
        self.terminal_states = env.get_terminal_states()
        self.updatable_states = env.get_nonterminal_states()
        self.all_states = self.updatable_states + self.terminal_states
        self.states_by_layer = sort_states_into_layers(self.all_states)
        self.layers_per_state = {state['id']: i for i, layer in enumerate(self.states_by_layer) for state in layer}
        self.box_sizes, self.artists = self._get_box_sizes()

        # Avoid recalculating full state embedding.
        self._size_cache = {}
        logging.info(f"StateEmbedding initialized with {len(self.all_states)} states")

    def set_size(self, new_size):
        # cache the box layouts.
        if new_size != self.size or new_size not in self._size_cache:
            logging.info(f"StateImageManager resized to {new_size}")
            self.size = new_size
            if new_size in self._size_cache:
                self.box_placer = self._size_cache[new_size]
            else:
                self.box_placer = self._calc_dims()
                self._size_cache[new_size] = self.box_placer
            return True
        return False

    def _calc_dims(self):
        """
        Calculate state layout.
        """
        # Place all game states into the image region
        box_placer, _, _ = get_box_placer(self.size,
                                          self.all_states,
                                          box_sizes=self.box_sizes,
                                          layer_vpad_px=1,
                                          layer_bar_w=1,
                                          player=self._env.player, key_size=self.key_size)

        # Swap state positions so wins are on the right, losses on the left, draws in the middle, etc.
        terminal_lut = {state: state.check_endstate() for state in self.all_states}
        tree_opt = SimpleTreeOptimizer(image_size=self.size,
                                       states_by_layer=self.states_by_layer,
                                       state_positions=box_placer.box_positions,
                                       terminal=terminal_lut)
        new_positions = tree_opt.get_new_positions()
        box_placer.box_positions = new_positions

        return box_placer

    def _get_box_sizes(self):
        artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in SPACE_SIZES]
        box_sizes = [artists[layer_no].get_image(Game()).shape[0] for layer_no in range(len(artists))]
        return box_sizes, artists


def test_state_embedding():
    img_size = (1200, 980)
    from reinforcement_base import Environment
    from baseline_players import HeuristicPlayer
    from game_base import Mark
    from mouse_state_manager import MouseBoxManager
    agent = HeuristicPlayer(mark=Mark.X, n_rules=1)
    opponent = HeuristicPlayer(mark=Mark.O, n_rules=2)
    env = Environment(opponent_policy=opponent, player_mark=Mark.X)
    embed = StateEmbedding(env, key_size=(300, 70))
    embed.set_size(img_size)
    mouse = MouseBoxManager(None)
    mouse.set_boxes(embed.box_placer.box_positions)

    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    terminals, nonterminals = env.get_terminal_states(), env.get_nonterminal_states()
    all_states = terminals + nonterminals
    # for state in all_states:
    mouse.render_state(img, nonterminals, 1)
    mouse.render_state(img, terminals, 2)

    cv2.imshow("State Embedding Test", img[:, :, ::-1])  # Flip the image horizontally
    cv2.waitKey(0)


class StateEmbeddingKey(Key):
    """
    For display at the top.
    Shows the three main game states: X-WIN, DRAW, O-WIN.
    #                                         } (pad before line)
    +---------------------------------------+
    |   X-WIN       DRAW        O-WIN       | } v_pad
    |   +-------+   +-------+   +-------+   | } t_pad       
    |   | X     |   | X O O |   | O     |   |      
    |   |   X   |   | O X X |   |   O   |   |
    |   |     X |   | O X O |   |     O |   |
    |   +-------+   +-------+   +-------+   |
    |   R = 1.0     R = -0.5    R = -1.0    | } t_pad
    +---------------------------------------+ } v_pad   
    """

    def __init__(self, size, x_offset=None):

        super().__init__(size, x_offset)

        self._dims, self._content = self._make_key()

    def _make_key(self):
        # each list is a row, of cells, etc.  This default has 1 row, 3 columns.
        content = {'titles': [['X-WIN', 'DRAW', 'O-WIN']],
                   'states': [[Game.from_strs(["X  ", " X ", "  X"]),
                              Game.from_strs(["XOO", "OXX", "OXO"]),
                              Game.from_strs(["O  ", " O ", "  O"])]],
                   'captions': [["R: 1.0", "R: -0.5", "R: -1.0"]],  # Only 1 line for each, for now.
                   'font': cv2.FONT_HERSHEY_SIMPLEX,
                   'text_color': COLOR_SCHEME['text']}

        dims = {}

        # TODO: Move to LAYOUT
        v_pad_frac = 0.1  # fraction of TOTAL vertical space used for top & bottom padding
        h_pad_frac = 0.15  # fraction of TOTAL horizontal space used for left & right padding, and between columns

        t_pad_frac = 0.07  # fraction of TOTAL unused space used for text_padding
        v_img_frac = 0.66  # fraction of key height used for the game state images
        n_cols = len(content['titles'][0])  # number of columns in the key
        n_rows = len(content['titles'])  # number of rows in the key

        width, height = self.size

        horiz_pad_size = int(width * h_pad_frac / (1+n_cols))
        vert_pad_size = int(height * v_pad_frac / (1 + n_rows))

        horiz_size = width - (1+n_cols) * horiz_pad_size  # vertical size available for the key
        vert_size = height - (1+n_rows) * vert_pad_size  # horizontal size available for the key

        # set grid:
        col_w = int(horiz_size / n_cols)
        row_h = int(vert_size / n_rows)
        dims['columns'] = []
        x_left = horiz_pad_size
        for col in range(n_cols):
            x_right = x_left + col_w
            dims['columns'].append((x_left, x_right))
            x_left = x_right + horiz_pad_size
        dims['rows'] = []
        y_top = vert_pad_size
        for row in range(n_rows):
            y_bottom = y_top + row_h
            dims['rows'].append((y_top, y_bottom))
            y_top = y_bottom + vert_pad_size

        # set image size and get game artists:
        side_len = row_h  # square images
        test_img_h = int(side_len * v_img_frac)

        space_size = GameStateArtist.get_space_size(test_img_h)
        artist = GameStateArtist(space_size=space_size)
        content['state_images'] = [[artist.get_image(state) for state in state_row] for state_row in content['states']]
        img_h = artist.dims['img_size']
        dims['state_image_size'] = img_h

        vert_text_pad_size = int(img_h * t_pad_frac)  # vertical space for text
        vert_text_size = vert_size - img_h - 2 * vert_text_pad_size  # vertical space for text

        text_h = (vert_text_size)//2  # number of text_rows

        font_scale = get_font_scale(font=content['font'],
                                    max_height=text_h,
                                    incl_baseline=False)

        # set text and image positions:
        cells = []

        for row in range(n_rows):
            y_top, y_bottom = dims['rows'][row]
            cells.append([])
            for col in range(n_cols):
                
                x_left, x_right = dims['columns'][col]
                
                title_top, title_bottom = y_top, y_top + text_h
                img_top = title_bottom+vert_text_pad_size
                img_bottom = img_top + img_h
                caption_top = img_bottom + vert_text_pad_size
                caption_bottom = caption_top + text_h

                x_center = (x_left + x_right) // 2
                img_center_x_span = (x_center - img_h // 2, x_center - img_h // 2 + img_h)

                (width, height), baseline = cv2.getTextSize(content['titles'][row][col],
                                                            content['font'],
                                                            fontScale=font_scale,
                                                            thickness=1)
                txt_center_x = x_center - width // 2

                cells[-1].append({'title_y_span': (title_top, title_bottom),
                                 'caption_y_span': (caption_top, caption_bottom),
                                 'img_y_span': (img_top, img_bottom),
                                 'img_x_span': img_center_x_span,
                                 'text_pos': (txt_center_x, title_bottom +baseline),
                                 'bbox': {'x': (x_left, x_right),
                                          'y': (y_top, y_bottom)}})
        dims['cells'] = cells
        dims['n_rows'] = n_rows
        dims['n_cols'] = n_cols
        dims['font_scale'] = font_scale

        return dims, content

    def draw(self, img, indicate_value=None):
        """
        Draw the key on the given image.
        :param img: The image to draw on.
        """
        offset = np.array(self._get_draw_pos(img))
        if False:
            # Outline grid cells & padding:
            for col in range(len(self._dims['columns'])):

                x_left, x_right = self._dims['columns'][col]
                img[offset[1]: offset[1] + self.size[1], x_left, :] = 0
                img[offset[1]: offset[1] + self.size[1], x_right, :] = 0
                continue
            for row in range(len(self._dims['rows'])):
                y_top, y_bottom = self._dims['rows'][row]
                img[y_top, offset[0]: offset[0] + self.size[0], :] = 0
                img[y_bottom, offset[0]: offset[0] + self.size[0], :] = 0

        def _dot_at(xy, size=2, color=0):
            xy = xy[0]+offset[0], xy[1]+offset[1]
            img[xy[1]-size:xy[1]+size+1, xy[0]-size:xy[0]+size+1, :] = color

        for row in range(self._dims['n_rows']):
            for col in range(self._dims['n_cols']):
                cell=self._dims['cells'][row][col]


                x_left, x_right = self._dims['columns'][col]

                state_image = self._content['state_images'][row][col]
                img_y_span = cell['img_y_span']
                img_x_span = cell['img_x_span']
                img_pos = (img_x_span[0], img_y_span[0] + offset[1])
                img[offset[1] + img_pos[1]:offset[1] + img_pos[1]+state_image.shape[0],
                    offset[0] + img_pos[0]:offset[0] + img_pos[0]+state_image.shape[1], :] = state_image
                # _dot_at(img_pos,size=4)
                title_y_span = cell['title_y_span']
                title_x_span = cell['text_pos']
                #_dot_at((title_x_span[0], title_y_span[0]), size=4)
                #_dot_at((title_x_span[0], title_y_span[1]), size=4)

                pos = (title_x_span[0]+offset[0], title_y_span[1]+offset[1])
                cv2.putText(img, self._content['titles'][row][col],
                            pos, self._content['font'],self._dims['font_scale'],
                            self._content['text_color'], thickness=1, lineType=cv2.LINE_AA)

                cap_y_span = cell['caption_y_span']
                cap_x = offset[0] + img_pos[0]
                cap_y = cap_y_span[1] + offset[1]
                #_dot_at((cap_x, cap_y_span[0]), size=4, color=128)
                #_dot_at((cap_x, cap_y_span[1]), size=4, color=128)
                pos = (cap_x, cap_y)
                print("Writing caption at", pos, self._content['captions'][row][col])
                cv2.putText(img, self._content['captions'][row][col],pos,
                            self._content['font'], self._dims['font_scale'],
                            self._content['text_color'], thickness=1, lineType=cv2.LINE_AA)

                # img[offset[1]+title_span[0],10:200,:] =

        x0, y0 = offset
        x1, y1 = x0 + self.size[0], y0 + self.size[1]
        cv2.rectangle(img, (x0, y0), (x1, y1), 0, thickness=1)


def test_state_embedding_key():

    key_size = (500, 160)

    def key_factory(size):
        return StateEmbeddingKey(size=size)
    tester = KeySizeTester(key_size, key_factory)
    tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_state_embedding()
    test_state_embedding_key()
    # Example usage:
    # app = MyApp()  # Replace with your application instance
    # env = Environment()  # Replace with your environment instance
    # embedding = StateEmbedding(env)
    # embedding.set_size((800, 600))
    # print(embedding.box_placer.box_positions)  # Check the box positions
