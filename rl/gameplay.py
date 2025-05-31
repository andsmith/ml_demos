"""
Play agents against each other.
Watch the game unfold, or run many trials.
"""
import numpy as np
from tic_tac_toe import Game
from game_base import Mark, Result, get_reward, WIN_MARKS
from baseline_players import RandomPlayer, HeuristicPlayer
from multiprocessing import Pool, cpu_count
import time
import logging
from layout import LAYOUT
from colors import COLOR_X, COLOR_O, COLOR_DRAW
# VALID_STATES = Game.enumerate_states()[1]

from plot_trace import TraceArtist


class Match(object):
    """
    One game.  Coin flip decides who goes first.
    """

    def __init__(self, player1, player2):
        """
        :param player1: Policy
        :param player2: Policy
        """
        self._players = [player1, player2]
        self._game = None

        self._self_check = False

    @staticmethod
    def print_trace(trace):

        for turn in range(len(trace['game'])):
            print("")
            action_dist = trace['game'][turn]['action_dist']
            action_ind = trace['game'][turn]['action_ind']
            action = action_dist[action_ind][0]  # get the action from the distribution
            player = trace['game'][turn]['player']
            state = trace['game'][turn]['state']
            print(state)

            print("Turn:  %d\nPlayer %s marks cell (%d, %d):" % (turn + 1, player.name, action[0], action[1]))
            print("\n")

        print("Final state:")
        print(trace['game'][-1]['next_state'])
        print("Result: %s" % trace['result'].name)

    def play_and_trace(self, order=0, verbose=False, initi_state=None, deterministic=False):
        """
        Play a random game, return the sequence of states and actions.
        :param order:  -1 (p2,p1), 0 (random), 1 (p1, p2)
        :return:  A dict with keys 'state', 'action', 'result' and values:
           'state': the game state
           'action': tuple (row, col)  (if state is not terminal),
           'result': the result of the game, one ofMark.X, Mark.O, Result.DRAW  (if state is terminal).
        """
        players = self._players[:]
        if order == -1 or (order == 0 and np.random.rand() < 0.5):
            players = self._players[::-1]

        self._game = Game() if initi_state is None else initi_state

        trace = {'first player': players[0].player,
                 'second_player': players[1].player,
                 'game': [],
                 'result': None}
        while True:
            for player in players:
                action_dist, action_ind = player.recommend_and_take_action(self._game,
                                                                           deterministic=deterministic)
                action = action_dist[action_ind][0]
                next_state = self._game.clone_and_move(action, player.player)
                trace['game'].append({'state': self._game,
                                      'action_dist': action_dist,
                                      'action_ind': action_ind,
                                      'player': player.player,
                                      'reward': 0.0,
                                      'next_state': next_state})
                self._game = next_state
                # if self._self_check:
                # if self._game not in VALID_STATES:
                #    raise ValueError("Invalid state: %s" % self._game)
                if verbose:
                    print("Player %s marks cell (%d, %d):" % (player.player, action[0], action[1]))
                    print(self._game)
                    print("\n")
                w = self._game.check_endstate()
                if w is not None:
                    trace['result'] = w
                    trace['game'][-1]['reward'] = get_reward(self._game, action, player.player)
                    return trace

    def play(self, order=0, verbose=False, initi_state=None):
        """
        Play one game , return result.
        """
        trace = self.play_and_trace(order=order, verbose=verbose, initi_state=initi_state)
        return trace['result']


def demo_match():
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=1)
    match = Match(player1, player2)
    print("Winner: %s" % match.play(verbose=True).name)


def demo_fixed_match():
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=3)
    starting_point = Game()  # .from_strs(["X O", "X  ", " O"])
    match = Match(player1, player2)

    for _ in range(10):

        print("Winner: %s" % match.play(verbose=False, initi_state=starting_point).name)
        print('\n\n\n----------------------\n')


class ResultSet(object):
    """
    Accumulate traces.  Determine wins/losses/draws, count unique traces
    """

    def __init__(self, player_mark, opponent_mark):
        self._player_mark = player_mark
        self._opp_mark = opponent_mark
        self._cur_sample = None  # dict of permutations of wins/draw/loss (index into self._traces  )
        self._n_games = 0

        # Create mapping as new states come in, use to identify unique traces.
        self._state_to_ind = {}
        self._next_new_ind = 0

        self._trace_inds = {'wins': [],  # for each trace a tuple of indices
                            'losses': [],  # for identifying unique traces
                            'draws': []}

        self._traces = {'wins': [],  # the actual traces
                        'losses': [],
                        'draws': []}

        self._counts = {'wins': [],  # number of unique traces
                        'losses': [],
                        'draws': []}

    def get_summary(self):
        """
        Return a summary of the results.
        :return: dict with keys:

           - 'wins', 'losses', 'draws': each a tuple(n total, n distinct).
           - 'games': number of games played.
        """

        return {'wins': (int(np.sum(self._counts['wins'])), len(self._counts['wins'])),
                'losses': (int(np.sum(self._counts['losses'])), len(self._counts['losses'])),
                'draws': (int(np.sum(self._counts['draws'])), len(self._counts['draws'])),
                'games': self._n_games}

    def _get_ind_seq(self, trace):
        """
        Get the indices of the states in the trace.
        :param trace: list of Game states
        :return: list of indices of the states in the trace
        """
        indices = []
        for turn_info in trace['game']:
            state = turn_info['state']
            if state not in self._state_to_ind:
                self._state_to_ind[state] = self._next_new_ind
                self._next_new_ind += 1
            indices.append(self._state_to_ind[state])

        return tuple(indices)

    def _get_outcome(self, trace):
        """
        is it a win for self.player?
        """

        if (trace['result'] == WIN_MARKS[self._player_mark]):
            return 'wins'
        elif trace['result'] == WIN_MARKS[self._opp_mark]:
            return 'losses'
        elif trace['result'] == Result.DRAW:
            return 'draws'

        raise ValueError("Unexpected result!")

    def add_trace(self, trace):
        self._n_games += 1
        outcome = self._get_outcome(trace)
        trace_ind = self._get_ind_seq(trace)
        if trace_ind not in self._trace_inds[outcome]:
            self._trace_inds[outcome].append(trace_ind)
            self._traces[outcome].append(trace)
            self._counts[outcome].append(1)
        else:
            ind = self._trace_inds[outcome].index(trace_ind)  # TODO: Fix this, use a dict instead of a list.
            self._counts[outcome][ind] += 1

    def resample(self):
        logging.info("Resampling %d games, %d wins, %d losses, %d draws" %
                     (self._n_games, len(self._trace_inds['wins']),
                      len(self._trace_inds['losses']), len(self._trace_inds['draws'])))

        self._cur_sample = {'wins': np.random.permutation(len(self._traces['wins'])),
                            'losses': np.random.permutation(len(self._traces['losses'])),
                            'draws': np.random.permutation(len(self._traces['draws']))}
        logging.info("Resampled %d games, %d wins, %d losses, %d draws" %
                     (self._n_games, len(self._cur_sample['wins']),
                      len(self._cur_sample['losses']), len(self._cur_sample['draws'])))

    def draw(self, img, y_top, trace_artist_params={}, layout_params={}):
        """
        Three boxes, "Wins" "Draws" and "Losses".
        In each, approximately the same number of traces, if possible  (equal for now)
        in each box, as many traces as will fit.
        The title for each trace is its number of occurrences.


        :param img:  image to draw on
        :param y_top:  upper-most y coordinate to draw on.
        :param trace_artist_params: dict with parameters for the trace artist, or None to use defaults.
        :param dims: dict with dimensions for the drawing (subset of layout['matches'['match_area']])
        :returns:  list of (bbox, probability) for the mousing over & updating the ColorKey.
        """
        w, h = img.shape[1], img.shape[0]
        
        
        if self._cur_sample is None:
            self.resample()

        layout = LAYOUT['results_viz']['match_area']
        layout.update(layout_params)

        h = img.shape[0] - y_top
        trace_size = layout['trace_size']
        trace_w, trace_h = trace_size
        
        ta = TraceArtist(trace_size, Mark.X, params=trace_artist_params, sample_header=" x 10   ")
        
        trace_h = ta.dims['y_bottom']

        trace_pad = int(layout['trace_pad_frac'] * trace_w)  # padding between traces
        group_pad = int(layout['group_pad_frac'] * w)
        group_bar_thickness = int(layout['group_bar_thickness_frac'] * w)

        # calculate total horiz. space for drawing traces.
        h_space = 4 * group_pad + 3*trace_pad + 6*group_bar_thickness
        n_h_traces = int((w-h_space) / (trace_w + trace_pad))  # number of traces that fit horizontally
        v_space = 2*group_pad + 2 * trace_pad + 2 * group_bar_thickness
        n_v_traces = int((h - v_space) / (trace_h + trace_pad))  # number of traces that fit vertically

        n_win_cols = min(len(self._traces['wins']), int(np.ceil(n_h_traces/3)))
        n_loss_cols = min(len(self._traces['losses']), int(np.ceil(n_h_traces/3)))
        n_draw_cols = min(len(self._traces['draws']), max(0, n_h_traces - n_win_cols - n_loss_cols))
        n_rows = n_v_traces

        #print("Drawing colums: %d wins, %d losses, %d draws" % (n_win_cols, n_loss_cols, n_draw_cols))
        #print("On a grid of %d x %d traces" % (n_h_traces, n_v_traces))

        r = 0
        no_additions = 0
        while n_win_cols + n_loss_cols + n_draw_cols < n_h_traces:
            if r % 3 == 0:
                if n_win_cols < len(self._traces['wins']):
                    n_win_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            elif r % 3 == 1:
                if n_loss_cols < len(self._traces['losses']):
                    n_loss_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            elif r % 3 == 2:
                if n_draw_cols < len(self._traces['draws']):
                    n_draw_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            if no_additions == 3:
                # no more traces to add, stop
                break
            r += 1

        n_traces = n_h_traces * n_v_traces

        n_wins = min(len(self._traces['wins']), n_v_traces*n_win_cols)
        n_losses = min(len(self._traces['losses']), n_v_traces*n_loss_cols)
        n_draws = min(len(self._traces['draws']), n_v_traces*n_draw_cols)

        

        r = 0
        #print("Drawing %d wins, %d losses, %d draws" % (n_wins, n_losses, n_draws))
        #print("From totals: %d wins, %d losses, %d draws" % (len(self._traces['wins']),
        #                                                     len(self._traces['losses']),
        #                                                     len(self._traces['draws'])))

        def _box_width(n):
            bw = n * trace_w + (n+1) * trace_pad
            return bw
        def _box_height(n):
            bh = n * trace_h + (n+1) * trace_pad
            return bh
        # draw the boxes for each outcome group
        x_left = 0
        


        bbox_top = y_top + group_pad + group_bar_thickness
        bbox_height = h - group_pad*2 -group_bar_thickness * 2 
        win_width, draw_width, loss_width = _box_width(n_win_cols), _box_width(n_draw_cols), _box_width(n_loss_cols)
        used_width = win_width + draw_width + loss_width + 6 * group_bar_thickness
        horiz_spacing = (w- used_width) // 4

        #bbox_bottom = bbox_top + group_bar_thickness + 2 * 
        win_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+win_width + horiz_spacing+group_bar_thickness),
                    'y': (bbox_top, bbox_top + bbox_height)}
        x_left = win_bbox['x'][1] +group_bar_thickness
        draw_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+draw_width+group_bar_thickness+ horiz_spacing),
                     'y': (bbox_top, bbox_top + bbox_height)}
        x_left = draw_bbox['x'][1]+group_bar_thickness
        loss_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+loss_width +group_bar_thickness+ horiz_spacing),
                     'y': (bbox_top, bbox_top + bbox_height)}

        def _draw_box_at(img, color, bbox, thickness=1):
            """
            bbox defines region INSIDE box of 'thickness' pixels.
            """
            thickness = 1 if thickness < 1 else int(thickness)
            x0, y0 = bbox['x'][0]-thickness, bbox['y'][0]-thickness
            x1, y1 = bbox['x'][1]+1, bbox['y'][1]+1
            # top
            img[y0:y0+thickness, x0:x1+thickness] = color
            # bottom
            img[y1:y1+thickness, x0:x1+thickness] = color
            # left
            img[y0:y1+thickness, x0:x0+thickness] = color
            # right
            img[y0:y1+thickness, x1:x1+thickness] = color

            return thickness

        def _draw_trace_set(group_bbox, traces, kind):
            count = self._counts
            x = group_bbox['x'][0] + trace_pad 
            y = group_bbox['y'][0] + trace_pad 
            for i, trace in enumerate(traces):
                trace = self._traces[kind][self._cur_sample[kind][i]]
                count = self._counts[kind][self._cur_sample[kind][i]]
                header_text = ("x %i" % count) if count > 1 else "X 1"
                pos = (x, y)
                ta.draw_trace(img, trace, pos, header_txt=header_text)
                x += trace_w + trace_pad

                if x + trace_w > group_bbox['x'][1]:
                    x = group_bbox['x'][0] + trace_pad
                    y += trace_h 

        _draw_box_at(img, COLOR_X, win_bbox, thickness=group_bar_thickness)
        _draw_box_at(img, COLOR_DRAW, draw_bbox, thickness=group_bar_thickness)
        _draw_box_at(img, COLOR_O, loss_bbox, thickness=group_bar_thickness)

        def _get_sample(traces, inds):
            return [traces[i] for i in inds]

        _draw_trace_set(win_bbox, _get_sample(self._trace_inds['wins'], self._cur_sample['wins'][:n_wins]), 'wins')
        _draw_trace_set(draw_bbox, _get_sample(self._trace_inds['draws'], self._cur_sample['draws'][:n_draws]), 'draws')
        _draw_trace_set(loss_bbox, _get_sample(
            self._trace_inds['losses'], self._cur_sample['losses'][:n_losses]), 'losses')
        


def test_result_set(n=100):
    player = HeuristicPlayer(Mark.X, n_rules=2)
    opponent = HeuristicPlayer(Mark.O, n_rules=2)
    rs = ResultSet(player.player, opponent.player)
    for _ in range(n):
        match = Match(player, opponent)
        rs.add_trace(match.play_and_trace(order=0, verbose=False))

    import cv2

    img_size = [(1000, 550)]

    def on_mouse(event, x, y, flagqs, param):
        """
        Mouse callback for the image window.
        """
        

    cv2.namedWindow("results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("results", img_size[0][0], img_size[0][1])
    # Set the resize callback to update the image size
    cv2.setMouseCallback("results", on_mouse)

    import pprint
    print("Test set summary:")
    pprint.pprint(rs.get_summary(), width=40)

    while True:
        _,_, width, height = cv2.getWindowImageRect("results")
        
        new_size = width, height
        if (new_size[0] != img_size[0][0] or new_size[1] != img_size[0][1]):
            img_size[0] = (new_size[0], new_size[1])
            print("New size: %s" % str(img_size[0]))
        img = np.zeros((img_size[0][1], img_size[0][0], 3), dtype=np.uint8)
        img[:] = 127
        rs.draw(img, y_top=60)
        cv2.imshow("results", img[:, :, ::-1])
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break




def _test():
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=4)
    game = Game(np.array([[Mark.O, Mark.EMPTY, Mark.EMPTY],
                          [Mark.EMPTY, Mark.X, Mark.EMPTY],
                          [Mark.O, Mark.EMPTY, Mark.X]]))
    print(game)
    print("%s recommends marking %s" % (player1.__class__, player1.take_action(game)))


'''
class Tournament(object):
    """
    Gather statistics from many games:
        Wins/losses from each player, given who went first:
    """

    def __init__(self, player1, player2, num_games, n_cpu=0):
        """
        :param player1: Policy
        :param player2: Policy
        :param num_games: int
        :param n_cpu: int, number of CPUs to use for parallel processing (0 means use all available)
        """
        self._player1=player1
        self._player2=player2
        self._num_games=num_games
        self._results={'player1_first': {'player1': 0, 'player2': 0, 'draw': 0},
                         'player2_first': {'player1': 0, 'player2': 0, 'draw': 0}}
        self._n_cpu=n_cpu if n_cpu > 0 else cpu_count() - 2

    def print_results(self):
        """
        Print results of the tournament.
        """
        print("\n\n===================================\n")
        print("\tPlayer 1: %s\n" % self._player1)
        print("\tPlayer 2: %s\n" % self._player2)

        print("\tPlayer 1 first:")
        print("\t\tPlayer 1 wins: %d" % self._results['player1_first']['player1'])
        print("\t\tPlayer 2 wins: %d" % self._results['player1_first']['player2'])
        print("\t\tDraws: %d" % self._results['player1_first']['draw'])
        print("\n")
        print("\tPlayer 2 first:")
        print("\t\tPlayer 1 wins: %d" % self._results['player2_first']['player1'])
        print("\t\tPlayer 2 wins: %d" % self._results['player2_first']['player2'])
        print("\t\tDraws: %d" % self._results['player2_first']['draw'])
        print("\n")
        print("\tTotal:")
        print("\t\tPlayer 1 wins: %d" % (self._results['player1_first']
              ['player1'] + self._results['player2_first']['player1']))
        print("\t\tPlayer 2 wins: %d" % (self._results['player1_first']
              ['player2'] + self._results['player2_first']['player2']))
        print("\t\tDraws: %d\n\n\n" % (self._results['player1_first']['draw'] + self._results['player2_first']['draw']))

    def run(self):
        """
        Run the tournament.
        """
        p1p2_matches=[Match(self._player1, self._player2) for _ in range(self._num_games//2)]
        p2p1_matches=[Match(self._player2, self._player1) for _ in range(self._num_games//2)]
        t0=time.time()
        if self._n_cpu == 1:
            print("Running in serial with %i matches..." % self._num_games)
            # p1p2_results, p2p1_results = [], []
            # for match in p1p2_matches:
            #    p1p2_results.append(_play(match))
            # for match in p2p1_matches:
            #    p2p1_results.append(_play(match))
            p1p2_results=[_play(match) for match in p1p2_matches]
            p2p1_results=[_play(match) for match in p2p1_matches]

        else:
            print("Running in parallel (%i) with %i matches..." % (self._n_cpu, self._num_games))
            with Pool(self._n_cpu) as pool:
                p1p2_results=pool.map(_play, p1p2_matches)
                p2p1_results=pool.map(_play, p2p1_matches)

        self._results['player1_first']['player1']=sum([1 for w in p1p2_results if w == self._player1.winning_result])
        self._results['player1_first']['player2']=sum([1 for w in p1p2_results if w == self._player2.winning_result])
        self._results['player1_first']['draw']=sum([1 for w in p1p2_results if w == Result.DRAW])
        self._results['player2_first']['player1']=sum([1 for w in p2p1_results if w == self._player1.winning_result])
        self._results['player2_first']['player2']=sum([1 for w in p2p1_results if w == self._player2.winning_result])
        self._results['player2_first']['draw']=sum([1 for w in p2p1_results if w == Result.DRAW])
        print("\tcompleted in %.3f seconds.\n" % (time.time() - t0))

def demo_tournament():
    player1=HeuristicPlayer(Mark.X, n_rules=6)
    # player2 = RandomPlayer(Mark.O)
    player2=HeuristicPlayer(Mark.O, n_rules=2)
    tournament=Tournament(player1, player2, 100, n_cpu=12)

    tournament.run()
    tournament.print_results()

def _play(match):
    w=match.play(randomize_players=False)
    return w
'''


def get_test_trace(required_result=None, order=0):
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=1)
    match = Match(player1, player2)
    if required_result is None:
        return match.play_and_trace(order=order, verbose=False)

    while trace['result'] != required_result:
        trace = match.play_and_trace(order=order, verbose=False)

    return trace


def demo_print_trace():
    trace = get_test_trace(required_result=Result.DRAW)
    Match.print_trace(trace)


if __name__ == "__main__":
    # test_draw_box()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # demo_match()
    test_result_set(100)
    # demo_print_trace()
    # demo_fixed_match()
    # demo_tournament()
    # _test()
