"""
Plot a trace (sequence of game states, actions, rewards), as output from Match.play_trace()
"""
import numpy as np
import cv2
import logging
from drawing import GameStateArtist
from util import write_lines_in_bbox, get_font_scale
from game_base import Mark
import matplotlib.pyplot as plt
from layout import LAYOUT


class TraceArtist(object):
    """
    Show the sequences of (state, action, rewards) for a game played by the RL agent.
    TODO:  subclass to plot in the format of an Alternating Markov Game (p1, p2, p1, ...) instead 
           of a RL agent (agent, environment, agent, ...).

    General format to plot a trace in a bounding box:


        # find maximum state image size to fit w/spacing & text rows, etc.

        Agent X goes first & last:
        +---------------------------+
        |                           |
        |        Header text        |  # Use header_txt="" to reserve room, or not show the header & move
        |                           |  # everything up, use header_txt=None.
        |    state - s    pi(s)     |
        |  +---------+ +---------+  |
        |  |         | | Actn.   |  |
        |  |         | | Dist. 1 |  |
        |  |         | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X      | | Actn.   |  |
        |  |    O    | | Dist. 2 |  |
        |  |         | |         |  |
        |  +---------+ +---------+  | 
        |  +---------+ +---------+  |
        |  |  X   O  | | Actn.   |  |
        |  |    O    | | Dist. 1 |  |
        |  |    X    | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X X O  | | Actn.   |  |
        |  |    O    | | Dist. 1 |  |
        |  |    X O  | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X X O  | | Actn.   |  |
        |  |  O O X  | | Dist. 1 |  |
        |  |    X O  | |         |  |
        |  +---------+ +---------+  |
        |              +---------+  |
        |              |  X X O  |  |  # final move was X, so on the right
        |              |  O O X  |  |
        |              |  X X O  |  |
        |              +---------+  |
        |                           |
        |     Return:  10.234       |
        |                           | # Justify everything to the top.
        |                           |
        +---------------------------+   

        Agent X goes Second (1 state shorter), O finishes.
        +---------------------------+
        |                           |
        |        Header text        |
        |                           |
        |    state - s    pi(s)     |
        |  +---------+ +---------+  |
        |  |         | | Actn.   |  |
        |  |    O    | | Dist. 1 |  |
        |  |         | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X   O  | | Actn.   |  |
        |  |    O    | | Dist. 2 |  |
        |  |         | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X   O  | | Actn.   |  |
        |  |  O O    | | Dist. 1 |  |
        |  |  X      | |         |  |
        |  +---------+ +---------+  |
        |  +---------+ +---------+  |
        |  |  X O O  | | Actn.   |  |
        |  |  O O X  | | Dist. 1 |  |
        |  |  X      | |         |  |
        |  +---------+ +---------+  |
        |  +---------+              |  # final move was O, so on the left
        |  |  X O O  |              |
        |  |  O O X  |              |
        |  |  X X O  |              |
        |  +---------+              |
        |                           |
        |     Return:  10.234       |
        |                           |
        |                           |
        |                           |
        |                           |
        +---------------------------+

    """

    def __init__(self, size_wh, player, params={}, sample_header=None):
        """
        :param size_wh:  The size of the bounding box for the trace, in pixels
        :param player:  The player for the agent.  Mark.X or Mark.O
        :param params:  Optional parameters for the trace (see _DEFAULT_PARAMS)
        """
        self._size = size_wh
        self._player = player  # the player for whom the trace is drawn
        self._params = LAYOUT['results_viz']['trace_params'].copy()
        self._params.update(params)
        self._cmap = plt.get_cmap('gray')  # colormap for the player colors
        self._sample_header = sample_header
        self.dims = self._calc_dims()
        #logging.info("TraceArtist initialized with size %s, state_img_size %i, for player %s" %
        #             (self._size, self.dims['img_size'], self._player.name))

    @staticmethod
    def _fmt_return(ret_val):
        return "R: %.3f" % ret_val

    def _calc_dims(self):
        """
        Calculate image positions, text positions and line coordinates for the two 
        kinds of traces (agent first, agent second)

        1. determine the font scales, determine text heights, padding, etc.
        2. determine state image size based on remaining space
        3. calculate the bounding boxes for the state images and text

        returns: dict with dimensions for plotting the trace.
        """
        n_state_rows = 6
        col_titles = ["s", "p(a|s)"]
        n_state_cols = 2
        w, h = self._size

        # if we are width-limited:
        n_h_pads = n_state_cols + 1
        n_squares = n_state_cols + n_h_pads * self._params['pad_frac']
        max_img_w = int(w / n_squares)

        # if we are height-limited:
        text_square_frac = self._params['header_font_frac'] + \
            self._params['return_font_frac']+self._params['col_title_frac']
        n_v_pads = n_state_rows + 4  # 3 for header, col_titles (and half-space), return, and bottom padding
        n_txt_pads = 5
        n_squares = (n_state_rows+text_square_frac) + n_v_pads * \
            self._params['pad_frac'] + n_txt_pads * self._params['txt_spacing_frac']
        max_img_h = int(h / n_squares)

        s_size = GameStateArtist.get_space_size(min(max_img_h, max_img_w), bar_w_frac=.2)
        artist = GameStateArtist(space_size=s_size, bar_w_frac=.2)
        img_s = artist.dims['img_size']

        pad_y_px = int(img_s * self._params['pad_frac'])
        txt_pad_y_px = int(img_s * self._params['txt_spacing_frac'])
        interior_pad_x = pad_y_px  # use the same padding for x and y, since we are top-justified
        exterior_total_pad_x = w - (interior_pad_x * (n_state_cols-1) + img_s * n_state_cols)

        exterior_pad_x = exterior_total_pad_x // 2

        if max_img_w < max_img_h:
            # we are width-limited, extra room below
            x_left = interior_pad_x
        else:
            x_left = exterior_pad_x
        # x_lefts = np.arange(n_state_cols) * (img_s + interior_pad_x) + x_left
        x_center = w//2
        x_left = x_center - img_s - interior_pad_x//2
        x_right = x_center + img_s + interior_pad_x//2
        x_lefts = [x_left, x_right - img_s]  # left and right columns

        # start placing things
        y = txt_pad_y_px  # start at the top, with padding

        # header
        header_h = int(img_s * self._params['header_font_frac'])
        header_y = y
        y += header_h
        y += txt_pad_y_px

        # column titles
        col_title_h = int(img_s * self._params['col_title_frac'])
        col_title_y = y
        y += col_title_h + pad_y_px

        turns = []
        for _ in range(n_state_rows):

            state_img_box = {'x': (x_lefts[0], x_lefts[0] + img_s),
                             'y': (y, y + img_s)}
            action_img_box = {'x': (x_lefts[1], x_lefts[1] + img_s),
                              'y': (y, y + img_s)}

            col_text_w = (state_img_box['x'][1] - state_img_box['x'][0])  # same for both

            turns.append({'left_bbox': state_img_box,
                          'right_bbox': action_img_box})
            y += img_s + pad_y_px

        return_h = int(img_s * self._params['return_font_frac'])

        y_bottom = y + return_h + txt_pad_y_px + txt_pad_y_px

        # calculate font scales
        text_w = w - 2 * exterior_pad_x  # width for the text, minus padding

        header_font_scale = get_font_scale(
            self._params['font'], max_height=header_h, max_width=text_w,
            incl_baseline=True, text_lines=["Header text" if self._sample_header is None else self._sample_header])

        return_w = int(text_w*.85)
        return_font_scale = get_font_scale(
            self._params['font'], max_height=return_h, max_width=return_w,
            incl_baseline=True, text_lines=[self._fmt_return(-3.587)])

        col_font_scale = get_font_scale(
            self._params['font'], max_height=col_title_h, max_width=col_text_w,
            incl_baseline=True, text_lines=col_titles[1:])

        dims = {'artist': artist,
                'img_size': img_s,
                'trace_size': self._size,
                'cell_size': s_size,
                'center_x': x_center,
                'header_y': header_y,
                'col_xs': x_lefts,
                'x_span': (x_left, x_right),
                'col_title_y': col_title_y,
                'exterior_pad_x': exterior_pad_x,
                'interior_pad_x': interior_pad_x,
                'txt_pad_y': txt_pad_y_px,
                'pad_y_px': pad_y_px,
                'header_font_scale': header_font_scale,
                'return_font_scale': return_font_scale,
                'col_font_scale': col_font_scale,
                'col_titles': col_titles,
                'turns': turns,
                'y_bottom': y_bottom,
                'col_title_h': col_title_h}
        return dims

    def _align_trace_RL(self, trace):
        """
        1. determine if agent goes first or second
        2. Use odd or even numbered turn's states as the agent's states.
        3. Determine the terminal state & return value.
        """
        state_imgs, action_imgs, action_cell_bboxes = [], [], []
        next_state_ind = 0

        while True:
            trace_state = trace['game'][next_state_ind]
            next_state_ind += 1

            player = trace_state['player']
            state = trace_state['state']
            next_state = trace_state['next_state']

            action_dist = trace_state['action_dist']
            action_taken_ind = trace_state['action_ind']
            action_taken = action_dist[action_taken_ind][0]

            if player != self._player:
                if next_state_ind == len(trace['game']):
                    state_imgs.append(self.dims['artist'].get_image(next_state))
                    term_pos = 'left'
                    total_return = trace_state['reward']
                    break
                continue

            state_imgs.append(self.dims['artist'].get_image(state))  # , highlight_cell = action_taken)

            action_img, action_bboxes = self.dims['artist'].get_action_dist_image(action_dist, self._player,
                                                                                  cmap=self._cmap, highlight_choice=action_taken_ind)
            action_imgs.append(action_img)
            action_cell_bboxes.append(action_bboxes)

            # It's the agent's turn:
            if next_state_ind == len(trace['game']):
                # If we reached the end of the trace on our turn,
                state_imgs.append(self.dims['artist'].get_image(next_state))
                term_pos = 'right'
                total_return = trace_state['reward']
                break

        return state_imgs, action_imgs, term_pos, total_return, action_cell_bboxes

    def draw_trace(self, img, trace, pos_xy, header_txt=""):

        state_imgs, action_imgs, term_pos, total_return, action_cell_bboxes = self._align_trace_RL(trace)
        pad_y_px = self.dims['pad_y_px']
        pad_x_left = self.dims['exterior_pad_x']
        pad_x_int = self.dims['interior_pad_x']
        y_bottom = self.dims['y_bottom']
        txt_pad_y_px = self.dims['txt_pad_y']
        y_top_rel = pad_y_px if header_txt is not None else txt_pad_y_px
        center_x, x_span_both = self.dims['center_x'], self.dims['x_span']


        def _put_text(text, y_pos, x_span, font_scale, incl_baseline=True, justify='center'):
            """
            Put text in the image at the given position (spec offset to (0, 0)).
            :param text:  The text to put in the image.
            :param pos:  The position (x, y) to put the text.
            :return:  The height of the text in pixels.
            """
            (text_width, text_height), baseline = cv2.getTextSize(text, self._params['font'], font_scale, 1)
            text_height += baseline if incl_baseline else 0
            y_bottom = y_pos + text_height + pos_xy[1]
            center_x = int((x_span[0] + x_span[1]) / 2)
            if justify == 'center':
                x_left = pos_xy[0] + center_x - text_width // 2
            elif justify == 'left':
                x_left = pos_xy[0] + x_span[0]

            txt_pos = (x_left, y_bottom)

            cv2.putText(img, text, txt_pos, self._params['font'], font_scale,
                        self._params['colors']['text'], 1, cv2.LINE_AA)
            return text_height, baseline

        # Draw the header text
        if header_txt is not None:
            header_y = self.dims['header_y']
            h_height = _put_text(header_txt, header_y, x_span_both, self.dims['header_font_scale'], justify='center')[0]
            y_top_rel += _put_text(header_txt, header_y, x_span_both,
                                   self.dims['header_font_scale'], justify='center')[0]
            y_top_rel += txt_pad_y_px

        # Draw the column titles
        for col_ind, col_title in enumerate(self.dims['col_titles']):
            x_left = self.dims['col_xs'][col_ind]
            x_right = x_left + self.dims['img_size']
            x_span = (x_left, x_right)
            col_title_y = self.dims['col_title_y']
            col_h, col_base = _put_text(col_title, col_title_y, x_span,
                                        self.dims['col_font_scale'], justify='center', incl_baseline=False)
            y_top_rel += pad_y_px

        y_top_rel = col_title_y + col_h + col_base + txt_pad_y_px
        # import ipdb; ipdb.set_trace()
        for row in range(len(action_imgs)):

            state_bbox = self.dims['turns'][row]['left_bbox']
            action_bbox = self.dims['turns'][row]['right_bbox']

            # Draw the state image
            x_left = pos_xy[0] + state_bbox['x'][0]
            y_top = pos_xy[1] + state_bbox['y'][0]
            img[y_top:y_top + self.dims['img_size'],
                x_left:x_left + self.dims['img_size'], :] = state_imgs[row]

            # Draw the action image
            x_right = pos_xy[0] + action_bbox['x'][0]
            img[y_top:y_top + self.dims['img_size'],
                x_right:x_right + self.dims['img_size'], :] = action_imgs[row]

            y_top_rel = state_bbox['y'][1] + pad_y_px

        if term_pos == 'left':
            term_bbox = self.dims['turns'][-1]['left_bbox']
        else:
            term_bbox = self.dims['turns'][-1]['right_bbox']

        # Draw the terminal state image

        x_left = pos_xy[0] + term_bbox['x'][0]
        y_top = pos_xy[1] + y_top_rel

        img[y_top:y_top + self.dims['img_size'],
            x_left:x_left + self.dims['img_size'], :] = state_imgs[-1]

        y_top_rel += self.dims['img_size'] + txt_pad_y_px

        # Write the return value
        return_txt = self._fmt_return(total_return)
        # return_txt_pos = ( self.dims['center_x'], y_top )
        _put_text(return_txt, y_top_rel, x_span_both,
                  self.dims['return_font_scale'], justify='center', incl_baseline=True)

        return img
    
    def set_size(self, new_size_wh):
        self._size = new_size_wh
        self.dims=self._calc_dims() 


class ResizableTester(object):
    def __init__(self, traces, size, shape, margin=0):
        self._traces = traces
        self._size = size
        self._margin = margin
        self._shape = shape

    def get_frame(self, size):
        n_traces = self._shape[0] * self._shape[1]
        trace_width = (size[0]-self._margin*(self._shape[1]+1)) // self._shape[1]
        trace_height = (size[1]-self._margin*(self._shape[0]+1)) // self._shape[0]

        ta = TraceArtist((trace_width, trace_height), player=Mark.X, sample_header="Test Game He")
        true_trace_height = ta.dims['y_bottom']

        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        t_ind = 0
        for row in range(self._shape[0]):
            for col in range(self._shape[1]):
                if t_ind >= len(self._traces):
                    break

                x_left = self._margin + col * (trace_width + self._margin)
                y_top = self._margin + row * (true_trace_height + self._margin)

                # Draw bkg color
                trace_size = ()
                img[y_top:y_top + true_trace_height, x_left:x_left + trace_width, :] = ta._params['colors']['bg']

                ta.draw_trace(img, self._traces[t_ind], pos_xy=(x_left, y_top), header_txt="Test Game %i" %
                              (col+1) if np.random.rand() > 0.5 else "")
                t_ind += 1

        return img

    def start(self):
        """
        Start the OpenCV window and display the traces.
        """
        cv2.namedWindow("Trace", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trace", self._size[0], self._size[1])
        # Set the resize callback to update the image size

        while True:
            x, y, width, height = cv2.getWindowImageRect("Trace")
            img = self.get_frame((width, height))
            cv2.imshow("Trace", img[:, :, ::-1])
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()


def test_plot_trace():
    from gameplay import get_test_trace, Match
    test_shape = (2, 7)
    n_trials = test_shape[0] * test_shape[1]
    img_size = (900, 820)

    test_traces = [get_test_trace() for _ in range(n_trials)]
    for trace in test_traces:
        print("First player: %s" % (trace['first player'].name))
        Match.print_trace(trace)
    resizable_tester = ResizableTester(test_traces, size=img_size, shape=test_shape, margin=10)
    resizable_tester.get_frame(img_size)
    resizable_tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_plot_trace()
