"""
Play agents against each other.
Watch the game unfold, or run many trials.
"""
import numpy as np
from tic_tac_toe import Game
from game_base import Mark, Result, get_reward
from baseline_players import RandomPlayer, HeuristicPlayer
from multiprocessing import Pool, cpu_count
import time

# VALID_STATES = Game.enumerate_states()[1]



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
        if order==-1 or (order==0 and np.random.rand() < 0.5):
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
                trace['game'].append({'state':self._game,
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
    starting_point = Game()#.from_strs(["X O", "X  ", " O"])
    match = Match(player1, player2)

    for _ in range(10):

        print("Winner: %s" % match.play(verbose=False, initi_state=starting_point).name)
        print('\n\n\n----------------------\n')



def _test():
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=4)
    game = Game(np.array([[Mark.O, Mark.EMPTY, Mark.EMPTY],
                          [Mark.EMPTY, Mark.X, Mark.EMPTY],
                          [Mark.O, Mark.EMPTY, Mark.X]]))
    print(game)
    print("%s recommends marking %s" % (player1.__class__, player1.take_action(game)))


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
        p1p2_matches = [Match(self._player1, self._player2) for _ in range(self._num_games//2)]
        p2p1_matches = [Match(self._player2, self._player1) for _ in range(self._num_games//2)]
        t0 = time.time()
        if self._n_cpu == 1:
            print("Running in serial with %i matches..." % self._num_games)
            # p1p2_results, p2p1_results = [], []
            # for match in p1p2_matches:
            #    p1p2_results.append(_play(match))
            # for match in p2p1_matches:
            #    p2p1_results.append(_play(match))
            p1p2_results = [_play(match) for match in p1p2_matches]
            p2p1_results = [_play(match) for match in p2p1_matches]

        else:
            print("Running in parallel (%i) with %i matches..." % (self._n_cpu, self._num_games))
            with Pool(self._n_cpu) as pool:
                p1p2_results = pool.map(_play, p1p2_matches)
                p2p1_results = pool.map(_play, p2p1_matches)

        self._results['player1_first']['player1'] = sum([1 for w in p1p2_results if w == self._player1.winning_result])
        self._results['player1_first']['player2'] = sum([1 for w in p1p2_results if w == self._player2.winning_result])
        self._results['player1_first']['draw'] = sum([1 for w in p1p2_results if w == Result.DRAW])
        self._results['player2_first']['player1'] = sum([1 for w in p2p1_results if w == self._player1.winning_result])
        self._results['player2_first']['player2'] = sum([1 for w in p2p1_results if w == self._player2.winning_result])
        self._results['player2_first']['draw'] = sum([1 for w in p2p1_results if w == Result.DRAW])
        print("\tcompleted in %.3f seconds.\n" % (time.time() - t0))


def _play(match):
    w = match.play(randomize_players=False)
    return w


def demo_tournament():
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    #player2 = RandomPlayer(Mark.O)
    player2 = HeuristicPlayer(Mark.O, n_rules=2)
    tournament = Tournament(player1, player2, 1000, n_cpu=12)

    tournament.run()
    tournament.print_results()


def get_test_trace(required_result=Result.DRAW,order=0):
    player1 = HeuristicPlayer(Mark.X, n_rules=6)
    player2 = HeuristicPlayer(Mark.O, n_rules=1)
    match = Match(player1, player2)
    trace = None
    while trace is None or trace['result'] != required_result:
        trace = match.play_and_trace(order=order, verbose=False)
        
    return trace

def demo_print_trace():
    trace = get_test_trace(required_result=Result.DRAW) 
    Match.print_trace(trace)

if __name__ == "__main__":
    #demo_match()
    demo_print_trace()
    #demo_fixed_match()
    #demo_tournament()
    # _test()
