"""
Plot a trace (sequence of game states, actions, rewards), as output from Match.play_trace()
"""
import numpy as np
import cv2
import logging
from gameplay import get_test_trace, Match
from drawing import GameStateArtist
from util import write_lines_in_bbox, get_font_scale
from game_base import Mark, Result, WIN_MARKS
from colors import COLOR_BG, COLOR_LINES, COLOR_TEXT
import matplotlib.pyplot as plt


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

    _DEFAULT_PARAMS = {'header_font_frac': 0.4,  # fraction of image side length for the header text
                       'return_font_frac': 0.3,  # same for the return text
                       'col_title_frac': 0.3,  # fraction of image side length for the column titles
                       "pad_frac": 0.05,  # fraction of image side length to use as padding between images and text, etc.
                       'font': cv2.FONT_HERSHEY_SIMPLEX,
                       'colors': {'bg': COLOR_BG,
                                  'lines': COLOR_LINES,
                                  'text': COLOR_TEXT}
                       }

    def __init__(self, size_wh, player, params={}, sample_header=None):
        """
        :param size_wh:  The size of the bounding box for the trace, in pixels
        :param player:  The player for the agent.  Mark.X or Mark.O
        :param params:  Optional parameters for the trace (see _DEFAULT_PARAMS)
        """
        self._size = size_wh
        self._player = player  # the player for whom the trace is drawn
        self._params = self._DEFAULT_PARAMS.copy()
        self._params.update(params)
        self._cmap = plt.get_cmap('gray')  # colormap for the player colors
        self.dims = self._calc_dims(sample_header)

    @staticmethod
    def _fmt_return(ret_val):
        return "R: %.3f" % ret_val

    def _calc_dims(self, sample_header):
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
        n_squares = n_state_cols  + n_h_pads * self._params['pad_frac']
        max_img_w = int(w / n_squares)

        # if we are height-limited:
        text_square_frac =  self._params['header_font_frac'] + self._params['return_font_frac']+self._params['col_title_frac']
        n_v_pads = n_state_rows + 4.5  # 3 for header, col_titles (and half-space), return, and bottom padding
        n_squares = (n_state_rows+text_square_frac) + n_v_pads * self._params['pad_frac']
        max_img_h = int(h / n_squares)

        s_size = GameStateArtist.get_space_size(min(max_img_h, max_img_w), bar_w_frac=.2)
        artist = GameStateArtist(space_size=s_size, bar_w_frac=.2)
        img_s = artist.dims['img_size']

        pad_y_px = int(img_s * self._params['pad_frac'])
        interior_pad_x = pad_y_px  # use the same padding for x and y, since we are top-justified
        exterior_total_pad_x =   w - (interior_pad_x * (n_state_cols-1) + img_s * n_state_cols) 

        exterior_pad_x = exterior_total_pad_x // 2


        if max_img_w < max_img_h:
            # we are width-limited, extra room below
            x_left = interior_pad_x
        else:
            x_left = exterior_pad_x
        #x_lefts = np.arange(n_state_cols) * (img_s + interior_pad_x) + x_left
        x_center = w//2
        x_left = x_center - img_s - interior_pad_x//2
        x_right = x_center + img_s + interior_pad_x//2
        x_lefts = [x_left, x_right - img_s]  # left and right columns


        # start placing things
        y = pad_y_px  # put them on/below this pixel row

        # header
        header_h = int(img_s * self._params['header_font_frac'])
        header_y = pad_y_px
        y += header_h + int(pad_y_px*.5)

        # column titles
        col_title_h = int(img_s * self._params['col_title_frac'])
        col_title_y = y
        y += col_title_h + int(pad_y_px*.5)


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
        return_h =  int(img_s * self._params['return_font_frac'])

        print(header_h, return_h, col_title_h)

        # calculate font scales
        text_w = w - 2 * exterior_pad_x  # width for the text, minus padding

        header_font_scale = get_font_scale(
            self._params['font'], max_height=header_h, max_width=text_w,
            incl_baseline=True, text_lines=["Header text" if sample_header is None else sample_header])
        
        return_w = int(text_w*.85)
        return_font_scale = get_font_scale(
            self._params['font'], max_height=return_h, max_width=return_w,
            incl_baseline=False, text_lines=[self._fmt_return(-3.587)])
        

        col_font_scale = get_font_scale(
            self._params['font'], max_height=col_title_h, max_width=col_text_w,
            incl_baseline=True, text_lines=col_titles[1:])
        print("Header font scale: %.2f, return font scale: %.2f, col font scale: %.2f" %    
                (header_font_scale, return_font_scale, col_font_scale))
        dims = {'artist': artist,
                'img_size': img_s,
                'cell_size': s_size,
                'center_x': x_center,
                'header_y': header_y,
                'col_xs': x_lefts,
                'x_span': (x_left, x_right),
                'col_title_y': col_title_y,
                'exterior_pad_x': exterior_pad_x,
                'interior_pad_x': interior_pad_x,
                'pad_y_px': pad_y_px,
                'header_font_scale': header_font_scale,
                'return_font_scale': return_font_scale,
                'col_font_scale': col_font_scale,
                'col_titles': col_titles,
                'turns': turns}
        
        import pprint
        pprint.pprint(dims)

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

            if player != self._player:
                if next_state_ind == len(trace['game']):
                    state_imgs.append(self.dims['artist'].get_image(next_state))
                    term_pos = 'left'
                    total_return = trace_state['reward']
                    break
                continue

            action_dist = trace_state['action_dist']
            action_taken_ind = trace_state['action_ind']

            state_imgs.append(self.dims['artist'].get_image(state))

            action_img, action_bboxes = self.dims['artist'].get_action_dist_image(action_dist, self._player,
                                                                                  cmap=self._cmap,
                                                                                  highlight_choice=action_taken_ind)
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

        y_top_rel = pad_y_px

        center_x, x_span_both = self.dims['center_x'], self.dims['x_span']

        

        def _put_text(text, y_pos,x_span, font_scale, justify='center'):
            """
            Put text in the image at the given position (spec offset to (0, 0)).
            :param text:  The text to put in the image.
            :param pos:  The position (x, y) to put the text.
            :return:  The height of the text in pixels.
            """
            (text_width, text_height), baseline = cv2.getTextSize(text, self._params['font'], font_scale, 1)
            y_bottom = y_pos + text_height + pos_xy[1]
            center_x = int((x_span[0] + x_span[1]) / 2)
            if justify == 'center':
                x_left = pos_xy[0] + center_x - text_width // 2
            elif justify == 'left':
                x_left =  pos_xy[0] + x_span[0] 

            txt_pos = (x_left, y_bottom)

            cv2.putText(img, text, txt_pos, self._params['font'], font_scale,
                        self._params['colors']['text'], 1, cv2.LINE_AA)
            return text_height, baseline
        
        # Draw the header text
        if header_txt is not None:
            header_y = self.dims['header_y'] 
            y_top_rel += _put_text(header_txt, header_y,x_span_both, self.dims['header_font_scale'], justify='center')[0]
            y_top_rel += pad_y_px

                    

        # Draw the column titles
        for col_ind, col_title in enumerate(self.dims['col_titles']):
            x_left = self.dims['col_xs'][col_ind]
            x_right = x_left + self.dims['img_size']
            x_span = (x_left, x_right)
            col_title_y = self.dims['col_title_y'] 
            col_h, col_base=_put_text(col_title, col_title_y,x_span, self.dims['col_font_scale'], justify='center')
        
        y_top_rel = col_title_y + col_h + col_base + int(pad_y_px* 0.5)

        for row in range(len(action_imgs)):
            state_bbox = self.dims['turns'][row]['left_bbox']
            action_bbox = self.dims['turns'][row]['right_bbox']

            # Draw the state image
            x_left = pos_xy[0] + state_bbox['x'][0]
            y_bottom = y_top_rel + self.dims['img_size'] + pos_xy[1]
            y_top = y_top_rel + pos_xy[1]

            img[y_top:y_bottom, x_left:x_left + self.dims['img_size'], :] = state_imgs[row]

            # Draw the action image
            x_right = pos_xy[0] + action_bbox['x'][0]
            img[y_top:y_bottom, x_right:x_right + self.dims['img_size'], :] = action_imgs[row]

            y_top_rel += self.dims['img_size'] + pad_y_px

        if term_pos == 'left':
            term_bbox = self.dims['turns'][-1]['left_bbox']
        else:
            term_bbox = self.dims['turns'][-1]['right_bbox']

        # Draw the terminal state image
        x_left = pos_xy[0] + term_bbox['x'][0]
        y_top = y_top_rel + pos_xy[1]
        y_bottom = y_top + self.dims['img_size']
        img[y_top:y_bottom, x_left:x_left + self.dims['img_size'], :] = state_imgs[-1]

        y_top_rel += self.dims['img_size'] + pad_y_px

        # Write the return value
        return_txt = self._fmt_return(total_return)
        # return_txt_pos = ( self.dims['center_x'], y_top )
        _put_text(return_txt, y_top_rel, x_span_both,self.dims['return_font_scale'], justify='left')

        return img


class ResizableTester(object):
    def __init__(self, traces, size):
        self._traces = traces
        self._size = size

    def get_frame(self, size, margin=5):
        n_traces = len(self._traces)
        trace_width = (size[0]-margin*(n_traces+1)) // n_traces
        trace_height = size[1] - margin * 2

        ta = TraceArtist((trace_width, trace_height), player=Mark.X, sample_header="Test Game He")
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        for col, trace in enumerate(self._traces):

            x_left = margin + col * (trace_width + margin)
            y_top = margin

            # Draw bkg color
            trace_size = ()
            img[y_top:y_top + trace_height, x_left:x_left + trace_width, :] = ta._params['colors']['bg']

            ta.draw_trace(img, trace, pos_xy=(x_left, y_top), header_txt="Test Game %i" %
                          (col+1) if np.random.rand() > 0.5 else "")

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
            img = self.get_frame((width, height),margin=1)
            cv2.imshow("Trace", img[:, :, ::-1])
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()


def test_plot_trace():

    n_trials = 14
    img_size = (1900, 620)

    test_traces = [get_test_trace(Result.DRAW, order=1) for _ in range(n_trials)]
    for trace in test_traces:
        print("First player: %s" % (trace['first player'].name))
        Match.print_trace(trace)
    resizable_tester = ResizableTester(test_traces, size=img_size)
    resizable_tester.get_frame(img_size)
    resizable_tester.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_plot_trace()
