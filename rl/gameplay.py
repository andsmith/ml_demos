"""
Play agents against each other.
Watch the game unfold, or run many trials.
"""
import numpy as np
from tic_tac_toe import Game, Mark
from baseline_players import RandomPlayer, HeuristicPlayer
from multiprocessing import Pool, cpu_count
import time

VALID_STATES = Game.enumerate_states()[1]

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

    def play(self, randomize_players=True, verbose=False):
        """
        Play one game.
        """
        if randomize_players and np.random.rand() < 0.5:
            self._players.reverse()

        self._game = Game()

        while True:
            for player in self._players:
                action = player.recommend_action(self._game)
                self._game.move(player.player, action)
                if self._self_check:
                    if self._game not in VALID_STATES:
                        raise ValueError("Invalid state: %s" % self._game)
                if verbose:
                    print("Player %s marks cell (%d, %d):" % (player.player, action[0], action[1]))
                    print(self._game)
                    print("\n")
                w = self._game.check_terminal()
                if w is not None:
                    return w


def demo_match():
    player1 = HeuristicPlayer(Mark.X)
    player2 = HeuristicPlayer(Mark.O)
    match = Match(player1, player2)
    print("Winner: %s" % match.play(verbose=True).name)


def _test():
    hp = HeuristicPlayer(Mark.X)
    game = Game(np.array([[Mark.O, Mark.EMPTY, Mark.EMPTY],
                          [Mark.EMPTY, Mark.X, Mark.EMPTY],
                          [Mark.O, Mark.EMPTY, Mark.X]]))
    print(game)
    print("%s recommends marking %s" % (hp.__class__, hp.recommend_action(game)))


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
        self._player1 = player1
        self._player2 = player2
        self._num_games = num_games
        self._results = {'player1_first': {'player1': 0, 'player2': 0, 'draw': 0},
                         'player2_first': {'player1': 0, 'player2': 0, 'draw': 0}}
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count() - 2

    def print_results(self):
        """
        Print results of the tournament.
        """
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

        import pprint
        pprint.pprint(self._results)

    def run(self):
        """
        Run the tournament.
        """
        p1p2_matches = [Match(self._player1, self._player2) for _ in range(self._num_games//2)]
        p2p1_matches = [Match(self._player2, self._player1) for _ in range(self._num_games//2)]
        t0 = time.time()
        if self._n_cpu == 1:
            print("Running in serial mode with %i matches..." % self._num_games)
            p1p2_results = [_play(match) for match in p1p2_matches]
            p2p1_results = [_play(match) for match in p2p1_matches]

        else:
            print("Running in parallel (%i) mode with %i matches..." % (self._n_cpu, self._num_games))
            with Pool(self._n_cpu) as pool:
                p1p2_results = pool.map(_play, p1p2_matches)
                p2p1_results = pool.map(_play, p2p1_matches)

        self._results['player1_first']['player1'] = sum([1 for w in p1p2_results if w == self._player1.player])
        self._results['player1_first']['player2'] = sum([1 for w in p1p2_results if w == self._player2.player]) 
        self._results['player1_first']['draw'] = sum([1 for w in p1p2_results if w == Mark.EMPTY])
        self._results['player2_first']['player1'] = sum([1 for w in p2p1_results if w == self._player1.player])
        self._results['player2_first']['player2'] = sum([1 for w in p2p1_results if w == self._player2.player])
        self._results['player2_first']['draw'] = sum([1 for w in p2p1_results if w == Mark.EMPTY])
        print("\tcompleted in %.3f seconds.\n" % (time.time() - t0))

def _play(match):
    w = match.play(randomize_players=False)
    return w


def demo_tournament():
    player1 = HeuristicPlayer(Mark.X)
    player2 = RandomPlayer(Mark.O)
    tournament = Tournament(player1, player2, 10000, n_cpu=16)
    tournament.run()
    tournament.print_results()

if __name__ == "__main__":
    #demo_match()
    demo_tournament()
    # _test()
