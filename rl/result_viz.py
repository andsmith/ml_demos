"""
Play a large number of games, visuzalize the results as they are computed:

    * On a narrow band at the top of the image, show statistics and a graph:
        - on the right, show statistics:
            - n games played so far
            - n wins   (blue)
            - n draws  (green)
            - n losses (orange)
        - on the left, show a bar graph w/lines extending from the four statistics to the left.


       And show a 1/dimensional "thermomenter" type indicator showing the relative win/draw/loss rates.

    * On the botton, show as many example games as will fit in the image. 
        * Results are in three boxes, "Wins" "Draws" and "Losses"    
        * The sequence of game states in a vertical column under the outcome label.
            - The initial state is at the top, as normal.
            - Each subsequent state has the agent selected action highlighted
            - Terminal states appear at the bottom w/the final action highlighted
        * The reward for the final action is printed at the bottom.
"""
import numpy as np
from drawing import GameStateArtist
from util import get_font_scale
from tic_tac_toe import Game
from game_base import Mark
from threading import Thread
from gameplay import Match, ResultSet
import logging
from collections import OrderedDict
import cv2
from baseline_players import HeuristicPlayer
from color_key import ProbabilityColorKey
import matplotlib.pyplot as plt
from colors import COLOR_SCHEME
from reinforcement_base import Environment

from layout import LAYOUT

DEFAULT_COLORS = COLOR_SCHEME

class PolicyEvaluationResultViz(object):
    def __init__(self, player_policy, opp_policy, colors=None):
        self._player = player_policy.player
        self._opp = opp_policy.player
        self._opp_pi = opp_policy
        self._pi = player_policy
        self._colors = colors if colors is not None else DEFAULT_COLORS
        self._layout = LAYOUT['results_viz']
        self._env = Environment(opponent_policy=opp_policy, player_mark=self._player)
        self._updatable_states = self._env.get_nonterminal_states()
        self._n_states = len(self._updatable_states)
        self._n_diffs = 0#self._pi.compare(self._opp_pi, states=self._updatable_states, count=True, deterministic=True)

        self._color_key_size = (self._layout['color_key']['width'], self._layout['color_key']['height'])
        self._color_key = ProbabilityColorKey(size=self._color_key_size)
        self._cur_value = None  # mouseover
        self._cur_sample = None
        self._results = ResultSet(self._player)

    def play(self, n_games=200):
        match = Match(self._pi, self._opp_pi)

        for iter in range(n_games):
            trace = match.play_and_trace(order=((-1)**iter), verbose=False)
            self._results.add_trace(trace)

            if iter % 100==0 or iter == n_games-1:
                
                summary = self._results.get_summary()

                print("Played %i games, W: %i, D: %i, L: %i:" % (iter+1,
                                                               summary['wins']['total'],
                                                               summary['draws']['total'],
                                                               summary['losses']['total']))
                

    def _draw_summary(self, img):
        x_center = int(img.shape[1] / 2)
        w, h = img.shape[1], img.shape[0]
        # first calculate font sizes
        n_lines = 5  # len(summary_lines)
        height = self._layout['summary']['size']['h'] - self._layout['summary']['text_indent'][1] * 2
        h_per_line = int(height / n_lines)  # of text

        lefover_x_space = w - self._color_key_size[0]
        summary_width = int(lefover_x_space * self._layout['summary']['graph_width_frac'])
        total_graph_width = int(summary_width * self._layout['summary']['graph_width_frac']
                          * (1.0 - self._layout['summary']['graph_indent_frac']))

        font_scale = get_font_scale(self._layout['summary']['font'], h_per_line *
                                    self._layout['summary']['text_spacing_frac'], incl_baseline=True)

        # draw the color key
        self._color_key.draw(img, indicate_value=self._cur_value)

        def _draw_text(text, baseline_pos, color, font_scale=1.0, justify='left'):
            (txt_width, txt_height), baseline = cv2.getTextSize(line, self._layout['summary']['font'], font_scale, 1)
            y_pos = baseline_pos[1] + txt_height+baseline

            if justify == 'left':
                x_pos = baseline_pos[0]
            else:
                x_pos = baseline_pos[0] - txt_width

            pos = (x_pos, y_pos)

            print("Drawing text: '%s' at %s with font scale %f" % (text, pos, font_scale))
            cv2.putText(img, text, pos, self._layout['summary']['font'], font_scale, color, 1, cv2.LINE_AA)

            return pos, txt_height, txt_width, baseline

        # on the left, print the policy names, the number of policy differences and the number of games played.

        y0 = self._layout['summary']['text_indent'][1]
        x0 = self._layout['summary']['text_indent'][0]

        # Policy names:
        summary_lines = OrderedDict()
        summary_lines['player1'] = 'Player 1: %s' % self._pi
        summary_lines['player2'] = 'Player 2: %s' % self._opp_pi

        max_w = 0
        summary_y = y0 + h_per_line 
        for i, (line_key, line) in enumerate(summary_lines.items()):
            pos, _, w, _ = _draw_text(line, (x0, summary_y), self._colors['text'], font_scale=font_scale)
            img[pos[0], pos[1], :] = (0, 0, 255)
            summary_y += h_per_line
            max_w = max(max_w, w)
        graph_indent_x = int(max_w * self._layout['summary']['graph_indent_frac'])

        # Now print & draw the results graph:
        x_far_left = x0 + max_w + self._layout['summary']['text_indent'][0] * 2 
        x_far_right = img.shape[1] - self._layout['summary']['text_indent'][0]

        summary_lines = OrderedDict()
        count_ucount_result = self._results.get_summary()
        
        sum_result = {'games': count_ucount_result['games']['total'],
                      'wins': count_ucount_result['wins']['total'],
                      'draws': count_ucount_result['draws']['total'],
                      'losses': count_ucount_result['losses']['total']}
        
        summary_lines['first'] = " " # first is blank
        summary_lines['wins'] = '%s-Wins: %i' % (self._player.name, sum_result['wins'])
        summary_lines['draws'] = '%s-Draws: %i' % (self._player.name, sum_result['draws'])
        summary_lines['losses'] = '%s-Losses: %i' % (self._player.name, sum_result['losses'])
        summary_lines['games'] = 'Games: %i' % sum_result['games']

        x0 = x_far_left+graph_indent_x
        y0 = self._layout['summary']['text_indent'][1]
        y_text_pos = {}
        y_text_heights = {}
        width = 0
        for i, (line_key, line) in enumerate(summary_lines.items()):
            base_pos, heigt, txt_width, baseline = _draw_text(
                line, (x0, y0), self._colors['text'], font_scale=font_scale, justify='left')
            y_text_pos[line_key] = base_pos, baseline
            y_text_heights[line_key] = heigt
            y0 += h_per_line
            width = max(width, txt_width)

        # Draw the bar graphs(x-first, then x-second)
        x_left_t = x0 + width + self._layout['summary']['text_indent'][0] // 2
        x_right_t = x_left_t + total_graph_width
        x_sep =  self._layout['summary']['graph_sep']
        graph_width = (total_graph_width -x_sep)//2
        bar_max_len = graph_width
        
        x_first_left = x_left_t
        x_second_left = x_first_left + graph_width + x_sep
        x_first_right = x_first_left + graph_width
        x_second_right = x_second_left + graph_width

        def _draw_bar(color, x_span, x_frac_rel, name):
            x_left, x_right = x_span
            print("Drawing bar for %s with x_frac_rel=%f  (spanning x %i to %i)" % (name, x_frac_rel, x_left, x_right))
            x_end = x_right - int((1.0-x_frac_rel) * bar_max_len)
            x_start = x_left

            y_t_top = y_text_pos[name][0][1] - y_text_heights[name]
            y_t_bottom = y_text_pos[name][0][1]

            y_start = y_t_top
            y_end = y_t_bottom
            print('\tDrawing bar  x(%i, %i)  y(%i, %i)' % (x_start, x_end, y_start, y_end))
            img[y_start:y_end, x_start:x_end] = color

        # before drawing bars, shade area under it
        graph_y_top = y_text_pos['wins'][0][1] - y_text_heights['wins'] - y_text_pos['losses'][1]
        graph_y_bottom = y_text_pos['losses'][0][1] + y_text_pos['losses'][1]

        def gray_box(x_span):
            patch = img[graph_y_top:graph_y_bottom, x_span[0]:x_span[1]]
            img[graph_y_top:graph_y_bottom, x_span[0]:x_span[1]] = (patch * .9).astype(np.uint8)
            
        gray_box((x_second_left, x_second_right))
        gray_box((x_first_left, x_first_right))
        # draw x-first bars:
        n_first = count_ucount_result['games']['as_first']
        n_second = count_ucount_result['games']['as_second']
        _draw_bar(self._colors['color_x'], (x_first_left, x_first_right), count_ucount_result['wins']['as_first'] / n_first, 'wins')
        _draw_bar(self._colors['color_o'], (x_first_left, x_first_right), count_ucount_result['losses']['as_first'] / n_first, 'losses')
        _draw_bar(self._colors['color_draw'], (x_first_left, x_first_right), count_ucount_result['draws']['as_first'] / n_first, 'draws')
        # draw x-second bars:
        _draw_bar(self._colors['color_x'], (x_second_left, x_second_right), count_ucount_result['wins']['as_second'] / n_second, 'wins')
        _draw_bar(self._colors['color_o'], (x_second_left, x_second_right), count_ucount_result['losses']['as_second'] / n_second, 'losses')
        _draw_bar(self._colors['color_draw'], (x_second_left, x_second_right), count_ucount_result['draws']['as_second'] / n_second, 'draws')

        x_first_txt_pos = (x_first_left, graph_y_top- h_per_line//3)   
        x_second_txt_pos = (x_second_left, graph_y_top- h_per_line//3)
        cv2.putText(img, 'X-First', x_first_txt_pos, self._layout['summary']['font'],
                    font_scale * .9, self._colors['text'], 1, cv2.LINE_AA)
        cv2.putText(img, 'X-Second', x_second_txt_pos, self._layout['summary']['font'],
                    font_scale* .9, self._colors['text'], 1, cv2.LINE_AA)
        
    

    def _get_distinct_results_and_counts(self):
        """
        Find repeats, count them.
        """

        return None, None

    def draw(self, size_wh):
        img = np.zeros((size_wh[1], size_wh[0], 3), dtype=np.uint8)
        img[:] = self._colors['bg']

        self._draw_summary(img)

        cv2.line(img, (20, self._layout['summary']['size']['h']),
                 (size_wh[0]-20, self._layout['summary']['size']['h']),
                 self._colors['lines'], 2, cv2.LINE_AA)
        y_top = self._layout['summary']['size']['h']

        self._results.draw(img,y_top)
        return img


def load_value_func(n_rules_2=4):
    import pickle
    with open("value_function_iter_7.pkl", "rb") as f:
        value = pickle.load(f) 

    from policy_optim import ValueFuncPolicy
    opponent = HeuristicPlayer(n_rules=n_rules_2, mark=Mark.O)

    env = Environment(opponent_policy=opponent, player_mark=Mark.X)

    pi = ValueFuncPolicy(v=value, environment=env, player=Mark.X, old_policy=None) 

    return pi, opponent

def test_policy_eval_viz_learned(n_rules_2=2):
    img_size = (1590, 980)

    # agent = HeuristicPlayer(n_rules=6, mark=Mark.X)
    # opponent = HeuristicPlayer(n_rules=4, mark=Mark.O)
    # viz = PolicyEvaluationViz(player_policy=agent, opp_policy=opponent)

    pi, opponent = load_value_func(n_rules_2)
    viz = PolicyEvaluationResultViz(player_policy=pi, opp_policy=opponent)
    viz.play(n_games=1000)

    img = viz.draw(img_size)
    cv2.imshow("Policy Evaluation Visualization", img[:, :, ::-1])
    cv2.waitKey(0)

def test_policy_eval_viz_heuristic(n_rules_1=2, n_rules_2=4):
    img_size = (1590, 980)

    agent = HeuristicPlayer(n_rules=n_rules_1, mark=Mark.X)
    opponent = HeuristicPlayer(n_rules=n_rules_2, mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)
    viz.play(n_games=500)


    img = viz.draw(img_size)
    cv2.imshow("Policy Evaluation Visualization", img[:, :, ::-1])
    cv2.waitKey(0)

def test_perfect_vs_heuristic(n_rules_1=6):
    img_size = (1590, 980)

    agent = HeuristicPlayer(n_rules=n_rules_1, mark=Mark.X)
    #opponent = HeuristicPlayer(n_rules=n_rules_2, mark=Mark.O)

    from perfect_player import MiniMaxPolicy
    opponent = MiniMaxPolicy(player_mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)
    viz.play(n_games=1500)

    img = viz.draw(img_size)
    cv2.imshow("Policy Evaluation Visualization", img[:, :, ::-1])
    cv2.waitKey(0)

def test_perfect(n_rules_1=6):
    img_size = (1590, 980)
    from perfect_player import MiniMaxPolicy
    #opponent = HeuristicPlayer(n_rules=n_rules_2, mark=Mark.O)

    agent = MiniMaxPolicy(player_mark=Mark.X)
    opponent = MiniMaxPolicy(player_mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)
    viz.play(n_games=1500)

    img = viz.draw(img_size)
    cv2.imshow("Policy Evaluation Visualization", img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_perfect()

    import sys
    sys.exit()

    print(len(sys.argv))
    if len(sys.argv) > 2 :
        n_rules_1, n_rules_2 = int(sys.argv[1]), int(sys.argv[2])
        test_policy_eval_viz_heuristic(n_rules_1=n_rules_1, n_rules_2=n_rules_2)
    elif len(sys.argv) > 1 :
        n_rules_2 = int(sys.argv[1])
        test_policy_eval_viz_learned(n_rules_2=n_rules_2)
        
    # Uncomment the following line to run the test
    # test_value_func_policy()
