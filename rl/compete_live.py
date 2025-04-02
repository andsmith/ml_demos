"""
Register a list of policies.
Run a continual competition between them, plotting the current/past win/loss rates.
Update the policies as they learn, watch the win/loss ratios change.
"""
import numpy
import matplotlib.pyplot as plt
from game_base import Mark, Result
from tic_tac_toe import Game
from policies import Policy
from gameplay import Match
from baseline_players import HeuristicPlayer
import numpy as np


class LiveTournament(object):
    def __init__(self, matches_per_round=100):
        self._matches_per_round = matches_per_round
        self._round_no = 0
        self._policies = {}  # key by name
        # key by pairs of names (p1,p2), value is list (1 per-round) of {'P1_wins': int, 'P2_wins': int, 'draws': int}
        self._results = {}
        self._matches = []  # list of pairs of policy names, who is competing.

        # plotting
        self._artists = {}  # key by matchup, value is line-artist for result plot
        self._init_graphics()

    def get_win_history(self, result_list):
        x, y = [], []
        for round_result in result_list:
            n_games = round_result['P1_wins'] + round_result['P2_wins'] + round_result['draws']
            n_wins = round_result['P1_wins']
            draw_rate = round_result['draws'] / n_games
            win_rate = n_wins / n_games
            x.append(round_result['round_no'])
            y.append(win_rate)
        return x, y

    def update_plot(self):
        """
        Update the plot of the win/loss ratios.
        """

        # import ipdb; ipdb.set_trace()
        self._ax.clear()

        for match_up, results in self._results.items():
            round_num, win_rate = self.get_win_history(results)
            self._ax.plot(round_num, win_rate, label="%s vs %s" % (self._policies[match_up[0]], self._policies[match_up[1]]))
            #self._artists[match_up].set_data(round_num, win_rate)
        self._ax.legend(loc='upper left')
        self._ax.set_xlabel("Round number")
        self._ax.set_ylabel("Win rate")
        self._ax.set_ylim(-.05, 1) 
        self._ax.set_title("Win rate over time, %i games per round" % self._matches_per_round)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _run_round(self):
        for match_up in self._matches:
            p1, p2 = match_up
            policy1, policy2 = self._policies[p1], self._policies[p2]
            match_a = Match(self._policies[p1], self._policies[p2])
            match_b = Match(self._policies[p2], self._policies[p1])
            match_a_results = [match_a.play(randomize_players=False) for _ in range(self._matches_per_round//2)]
            match_b_results = [match_b.play(randomize_players=False) for _ in range(self._matches_per_round//2)]

            round_results = {'P1_wins': np.sum([r == policy1.winning_result for r in match_a_results]) + np.sum([r == policy2.winning_result for r in match_b_results]),
                             'P2_wins': np.sum([r == policy2.winning_result for r in match_a_results]) + np.sum([r == policy1.winning_result for r in match_b_results]),
                             'draws': np.sum([r == Result.DRAW for r in match_a_results]) + np.sum([r == Result.DRAW for r in match_b_results]),
                             'round_no': self._round_no}
            self._results[match_up].append(round_results)
        self._round_no += 1

    def _init_graphics(self):
        plt.ion()
        self._fig, self._ax = plt.subplots()

    def register(self, name, policy):
        self._policies[name] = policy
        if policy is None:
            del self._policies[name]
            self._matches = [(n1, n2) for n1, n2 in self._matches if n1 != name and n2 != name]

    def add_match(self, name1, name2):
        self._matches.append((name1, name2))
        self._results[(name1, name2)] = []
        self._artists[(name1, name2)] = None

        # add artists
        self._artists[(name1, name2)] = self._ax.plot([], [])[0]

    def start(self):
        while True:
            self._run_round()
            self.update_plot()

def test():
    t = LiveTournament(matches_per_round=100)
    p1 = HeuristicPlayer(Mark.X, n_rules=6)
    p2 = HeuristicPlayer(Mark.O, n_rules=6)
    p3 = HeuristicPlayer(Mark.O, n_rules=3)
    p4 = HeuristicPlayer(Mark.O, n_rules=0)
    t.register("p1", p1)
    t.register("p2", p2)
    t.register("p3", p3)
    t.register("p4", p4)
    t.add_match("p1", "p2")
    t.add_match("p1", "p3")
    t.add_match("p1", "p4")

    t.start()

if __name__ == "__main__":
    test()
