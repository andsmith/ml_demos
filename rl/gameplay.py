"""
Play agents against each other.
Watch the game unfold, or run many trials.
"""
from enum import IntEnum
import numpy as np
from tic_tac_toe import Game, GameTree
from game_base import Mark, Result, get_reward, WIN_MARKS
from baseline_players import RandomPlayer, HeuristicPlayer
from multiprocessing import Pool, cpu_count
import time
import logging
from layout import LAYOUT
from colors import COLOR_SCHEME
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

    def play_and_trace(self, order=0, verbose=False, init_state=None, deterministic=False):
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

        self._game = Game() if init_state is None else init_state

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

    def play(self, order=0, verbose=False, init_state=None):
        """
        Play one game , return result.
        """
        trace = self.play_and_trace(order=order, verbose=verbose, init_state=init_state)
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

        print("Winner: %s" % match.play(verbose=False, init_state=starting_point).name)
        print('\n\n\n----------------------\n')


class OutcomeEnum(IntEnum):
    WIN = 1
    DRAW = 0
    LOSS = -1


class ResultSet(object):
    """
    Accumulate traces.  Determine wins/losses/draws, count unique traces
    """

    def __init__(self, player_mark):
        self._player_mark = player_mark
        self._opp_mark = GameTree.opponent(player_mark)
        self._cur_sample = None  # dict of permutations of wins/draw/loss (index into self._traces  )
        self._n_games = 0
        self._show_most_freq = True  # show games repeatedly played

        # Create mapping as new states come in, use to identify unique traces.
        self._state_to_ind = {}
        self._gamestates = []  # inverse mapping

        self._game_traces = {'traces': [],  # list of game trace records (as returned by Match.play_and_trace)
                             'indsqs': [],  # list of tuples of indices into self._gamestates (each tuple is a trace)
                             'played_first': [],  # list of booleans, True if player_mark played first
                             'outcomes': []}  # list of OutcomeEnum values.

    def _get_ind_seq(self, trace):
        """
        Get the indices of the states in the trace.
        Add states not seen before to the mapping.
        :param trace: list of Game states
        :return: tuple of indices of the states in the trace
        """
        indices = []
        for turn_info in trace['game']:
            state = turn_info['state']
            if state not in self._state_to_ind:
                self._state_to_ind[state] = len(self._gamestates)
                self._gamestates.append(state)

            indices.append(self._state_to_ind[state])

        return tuple(indices)

    def get_summary(self):
        """
        Get a summary of the results.
        :return: dict with keys 'wins', 'losses', 'draws', each a dict with keys 'count' and 'traces'.
        """
        wins = np.array(self._game_traces['outcomes']) == OutcomeEnum.WIN
        losses = np.array(self._game_traces['outcomes']) == OutcomeEnum.LOSS
        draws = np.array(self._game_traces['outcomes']) == OutcomeEnum.DRAW
        went_first = np.array(self._game_traces['played_first'])
        n_games = len(self._game_traces['traces'])

        return {'wins': {'total': np.sum(wins),
                         'as_first': np.sum(wins[went_first]),
                         'as_second': np.sum(wins[~went_first])},
                'losses': {'total': np.sum(losses),
                           'as_first': np.sum(losses[went_first]),
                           'as_second': np.sum(losses[~went_first])},
                'draws': {'total': np.sum(draws),
                          'as_first': np.sum(draws[went_first]),
                          'as_second': np.sum(draws[~went_first])},
                'games': {'total': n_games,
                          'as_first': np.sum(went_first),
                          'as_second': np.sum(~went_first)}}

    def _get_outcome(self, trace):
        """
        is it a win for self.player?
        """
        if trace['result'] == Result.DRAW:
            return OutcomeEnum.DRAW
        elif (WIN_MARKS[trace['result']] == self._player_mark):
            return OutcomeEnum.WIN
        elif WIN_MARKS[trace['result']] == self._opp_mark:
            return OutcomeEnum.LOSS
        raise ValueError("Unexpected result!")

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

        layout = LAYOUT['results_viz']['match_area']
        layout.update(layout_params)

        # Calculate how many traces can fit, given the bbox, trace size, box size, etc.:
        h = img.shape[0] - y_top
        trace_size = np.array(layout['trace_size'])
        trace_w, trace_h = trace_size
        ta = TraceArtist(trace_size, Mark.X, params=trace_artist_params, sample_header=" x 10   ")
        trace_h = ta.dims['y_bottom']
        trace_pad = (np.array(layout['trace_pad_frac']) * trace_size).astype(int)
        group_pad = int(layout['group_pad_frac'] * w)
        group_bar_thickness = int(layout['group_bar_thickness_frac'] * w)

        # calculate total horiz./vert. space for drawing traces, then the number that will fit:
        h_space = 4 * group_pad + 3*trace_pad[0] + 6*group_bar_thickness
        n_h_traces = int((w-h_space) / (trace_w + trace_pad[0]))  # number of traces that fit horizontally
        v_space = 2*group_pad + 2 * trace_pad[1] + 2 * group_bar_thickness
        # n_v_traces= int((h - v_space) / (trace_h + trace_pad))  # number of traces that fit vertically
        n_v_traces = 2  # Hard code, x first above, o first below

        num_wins, num_draws, num_losses = np.sum(np.array(self._game_traces['outcomes']) == OutcomeEnum.WIN), \
            np.sum(np.array(self._game_traces['outcomes']) == OutcomeEnum.DRAW), \
            np.sum(np.array(self._game_traces['outcomes']) == OutcomeEnum.LOSS)

        n_win_cols = min(num_wins, int(np.ceil(n_h_traces/3)))
        n_loss_cols = min(num_losses, int(np.ceil(n_h_traces/3)))
        n_draw_cols = min(num_draws, max(0, n_h_traces - n_win_cols - n_loss_cols))
        n_rows = n_v_traces  # n_cols different in each box

        # Add columns to each category until the window is full, even if they'll be empty.
        r = 0
        no_additions = 0
        while n_win_cols + n_loss_cols + n_draw_cols < n_h_traces:
            if r % 3 == 0:
                if n_win_cols < num_wins:
                    n_win_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            elif r % 3 == 1:
                if n_loss_cols < num_losses:
                    n_loss_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            elif r % 3 == 2:
                if n_draw_cols < num_draws:
                    n_draw_cols += 1
                    no_additions = 0
                else:
                    no_additions += 1
            if no_additions == 3:
                # no more traces to add, stop
                break
            r += 1

        n_traces = n_h_traces * n_v_traces

        n_wins = min(num_wins, n_v_traces*n_win_cols)
        n_losses = min(num_losses, n_v_traces*n_loss_cols)
        n_draws = min(num_draws, n_v_traces*n_draw_cols)

        print("Drawing colums: %d wins, %d losses, %d draws" % (n_win_cols, n_loss_cols, n_draw_cols))
        print("On a grid of %d x %d traces" % (n_h_traces, n_v_traces))

        def _box_width(n):
            bw = n * trace_w + (n+1) * trace_pad[0]
            return bw

        # draw the boxes for each outcome group
        x_left = 0

        bbox_top = y_top + group_pad + group_bar_thickness
        bbox_height = h - group_pad*2 - group_bar_thickness * 2
        win_width, draw_width, loss_width = _box_width(n_win_cols), _box_width(n_draw_cols), _box_width(n_loss_cols)
        used_width = win_width + draw_width + loss_width + 6 * group_bar_thickness
        horiz_spacing = (w - used_width) // 4

        # bbox_bottom = bbox_top + group_bar_thickness + 2 *
        win_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+win_width + horiz_spacing+group_bar_thickness),
                    'y': (bbox_top, bbox_top + bbox_height)}
        x_left = win_bbox['x'][1] + group_bar_thickness
        draw_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+draw_width+group_bar_thickness + horiz_spacing),
                     'y': (bbox_top, bbox_top + bbox_height)}
        x_left = draw_bbox['x'][1]+group_bar_thickness
        loss_bbox = {'x': (x_left+horiz_spacing+group_bar_thickness, x_left+loss_width + group_bar_thickness + horiz_spacing),
                     'y': (bbox_top, bbox_top + bbox_height)}

        def _draw_box_at(img, color, bbox, thickness=1):
            # bbox defines region INSIDE box of 'thickness' pixels.
            thickness = 1 if thickness < 1 else int(thickness)
            x0, y0 = bbox['x'][0]-thickness, bbox['y'][0]-thickness
            x1, y1 = bbox['x'][1]+1, bbox['y'][1]+1
            img[y0:y0+thickness, x0:x1+thickness] = color
            img[y1:y1+thickness, x0:x1+thickness] = color
            img[y0:y1+thickness, x0:x0+thickness] = color
            img[y0:y1+thickness, x1:x1+thickness] = color

        def _draw_trace_set(group_bbox, trace_lists):
            """
            Draw a set of traces in the given group_bbox.
            :param group_bbox: dict with keys 'x' and 'y', each a tuple of (left, right)
            :param trace_lists: list of lists of {'trace': <trace info>, 'count': <int>}, 1 list per row
            :param kind: 'wins', 'losses', or 'draws'
            """
            x_left = group_bbox['x'][0] + trace_pad[0]
            y_top = group_bbox['y'][0] + trace_pad[1]
            for row in range(len(trace_lists)):
                y = y_top + row * (trace_h + trace_pad[1])
                for col in range(len(trace_lists[row])):
                    x = x_left + col * (trace_w + trace_pad[0])
                    trace = trace_lists[row][col]['traces'][0]
                    count = trace_lists[row][col]['count']
                    header_text = ("x %i" % count) if count > 1 else " "
                    pos = (x, y)
                    ta.draw_trace(img, trace, pos, header_txt=header_text)

        _draw_box_at(img, COLOR_SCHEME['color_x'], win_bbox, thickness=group_bar_thickness)
        _draw_box_at(img, COLOR_SCHEME['color_draw'], draw_bbox, thickness=group_bar_thickness)
        _draw_box_at(img, COLOR_SCHEME['color_o'], loss_bbox, thickness=group_bar_thickness)

        win_samp_size = {'cols': n_win_cols, 'num': n_wins, 'rows': n_rows}
        draw_samp_size = {'cols': n_draw_cols, 'num': n_draws, 'rows': n_rows}
        loss_samp_size = {'cols': n_loss_cols, 'num': n_losses, 'rows': n_rows}
        sample = self.resample(win_samp_size, draw_samp_size, loss_samp_size)

        _draw_trace_set(win_bbox, sample['wins'])
        _draw_trace_set(draw_bbox, sample['draws'])
        _draw_trace_set(loss_bbox,  sample['losses'])

    def add_trace(self, trace):
        self._n_games += 1
        outcome = self._get_outcome(trace)
        indsq = self._get_ind_seq(trace)
        self._game_traces['traces'].append(trace)
        self._game_traces['indsqs'].append(indsq)
        self._game_traces['outcomes'].append(outcome)
        played_first = trace['game'][0]['player'] == self._player_mark
        self._game_traces['played_first'].append(played_first)

    def resample(self, win_samp_info, draw_samp_info, loss_samp_info, show_most_freq_offset=0):
        """
        Get the lists of traces to show, a sample of each kind, either random or the most frequetly repeated.
        Split each list as evenly as possible into games started by the plaer and games started by the opponent.
        i.e.: for 10 wins, show the 5 most played with X starting, and 5 most played with O starting (if show_most_freq=0)

        :param *_samp_info: dict with:
            'cols': int, number of columns to show,
            'rows': int, number of rows to show, if 2, show X-starting on top, O-starting below,
            'num': int, number of traces to show,
        :param show_most_freq_offset:
           if n, then show the n-th most frequent trace, n+1, n+2, ... up to n + 'cols' per row.
           if None, show random traces.
           (Show most freq by default.)
        :return: dict for 'wins','draws','losses', each a list of lists of dicts with keys 'trace' and 'count' for plotting.
        """
        def _get_ranking(outcome, was_first=None):
            subset = [i for i in range(self._n_games) if (self._game_traces['outcomes'][i] == outcome and
                                                          (was_first is None or
                                                           was_first == self._game_traces['played_first'][i]))]
            ind_seqs = [self._game_traces['indsqs'][i] for i in subset]
            counts = {}
            for i, ind_seq in enumerate(ind_seqs):
                if ind_seq not in counts:
                    counts[ind_seq] = {'count': 0,
                                       'traces': []}
                counts[ind_seq]['count'] += 1
                counts[ind_seq]['traces'].append(self._game_traces['traces'][subset[i]])
            unique_game_seqs = list(counts.keys())
            seqences_rank = sorted(list(range(len(unique_game_seqs))),
                                   key=lambda i: counts[unique_game_seqs[i]]['count'], reverse=True)

            return {'order': seqences_rank,
                    'trace_counts': counts,
                    'unique_game_seqs': unique_game_seqs,
                    'subset': np.array(subset),
                    'n_unique': len(unique_game_seqs)}

        def _get_sample(outcome, info):

            if info['rows'] == 2:
                p_first = _get_ranking(outcome, was_first=True)
                p_second = _get_ranking(outcome, was_first=False)
                n = info['cols']
                if show_most_freq_offset is None:
                    sample_ind_rows = [np.random.choice(p_first['n_unique'], size=n, replace=False),
                                       np.random.choice(p_second['n_unique'], size=n, replace=False)]
                else:
                    sample_first = [i + show_most_freq_offset for i in range(min(n, p_first['n_unique']))]
                    sample_second = [i + show_most_freq_offset for i in range(min(n, p_second['n_unique']))]
                sample_first = [s for s in sample_first if s < p_first['n_unique']]
                sample_second = [s for s in sample_second if s < p_second['n_unique']]
                sample_ind_rows = [sample_first, sample_second
                                   ]
            elif info['rows'] != 2:
                raise NotImplemented

            sample = [[{'traces': p_first['trace_counts'][p_first['unique_game_seqs'][p_first['order'][i]]]['traces'],
                        'count': p_first['trace_counts'][p_first['unique_game_seqs'][p_first['order'][i]]]['count']}
                       for i in sample_first],

                      [{'traces': p_second['trace_counts'][p_second['unique_game_seqs'][p_second['order'][i]]]['traces'],
                        'count': p_second['trace_counts'][p_second['unique_game_seqs'][p_second['order'][i]]]['count']}
                       for i in sample_second]]

            return sample

        return {'wins': _get_sample(OutcomeEnum.WIN, win_samp_info),
                'draws': _get_sample(OutcomeEnum.DRAW, draw_samp_info),
                'losses': _get_sample(OutcomeEnum.LOSS, loss_samp_info)}


def test_result_set(n=100):
    player = HeuristicPlayer(Mark.X, n_rules=2)
    opponent = HeuristicPlayer(Mark.O, n_rules=2)
    rs = ResultSet(player.player)
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

    # import pprint
    # print("Test set summary:")
    # pprint.pprint(rs.get_summary(), width=40)

    while True:
        _, _, width, height = cv2.getWindowImageRect("results")

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
