"""
    1. random player
    2. heuristic player
    3. minimax(d) player ?

Players are implemented as probability distributions over the next possible actions.

NOTE:  These are not policies. 

"""
import numpy as np
from tic_tac_toe import Game, Mark, Result
from policies import Policy
import logging


class RandomPlayer(Policy):
    def __init__(self, mark):
        super(RandomPlayer, self).__init__(player=mark)

    def recommend_action(self, game_state):
        actions = game_state.get_actions()
        return actions[np.random.choice(len(actions))]

    def __str__(self):
        return "RandomPlayer(%s)" % self.player.name


class HeuristicPlayer(Policy):
    """
    Apply rules in this order:
    1. If there is a winning move, take it.
    2. If the opponent has a winning move, block it.
    3. If the center is open, take it.
    4. If there is a side-center, take it.

    """

    def __init__(self, mark, n_rules=6):
        """
        :param mark:  one of Mark.X or Mark.O  Player optimizes actions for this player.
        :param n_rules:  number of rules to apply.  If n_rules=0, then just return  all actions with uniform probability.
        """
        self._n_rules = n_rules
        super(HeuristicPlayer, self).__init__(player=mark)

    def __str__(self):
        return "HeuristicPlayer(%s, n_rules=%d)" % (self.player.name, self._n_rules)

    def recommend_action(self, game_state):
        """

        :param game_state:  A Game object, it is the current player's turn.
        :returns: list of (action, probability) for all possible actions player can make.
        """

        def _make_distribution(ret_actions):
            # return uniform distribution over all actions
            probs = np.ones(len(ret_actions)) / len(ret_actions)
            return [ap for ap in zip(ret_actions, probs)]

        actions = game_state.get_actions()

        if self._n_rules == 0:
            return _make_distribution(actions)  # give up, pick any action at random

        # Rule 1, any winning moves?
        player_next_states = [game_state.clone_and_move(action, self.player) for action in actions]
        endstates = [state.check_endstate() for state in player_next_states]
        good_action_inds = [action_i for action_i, term in enumerate(endstates) if term == self.winning_result]
        if len(good_action_inds) > 0:
            good_actions = [actions[i] for i in good_action_inds]
            return _make_distribution(good_actions)

        if self._n_rules == 1:
            return _make_distribution(actions)

        # Rule 2, any blocking moves?
        opponent_next_states = [game_state.clone_and_move(action, self.opponent) for action in actions]
        opponent_endstates = [state.check_endstate() for state in opponent_next_states]
        good_action_inds = [action_i for action_i, term in enumerate(opponent_endstates) if term == self.losing_result]
        if len(good_action_inds) > 0:
            good_actions = [actions[i] for i in good_action_inds]
            return _make_distribution(good_actions)

        if self._n_rules == 2:
            return _make_distribution(actions)

        # Rule 3, center
        if (1, 1) in actions:
            good_action_inds = [i for i, action in enumerate(actions) if action == (1, 1)]
            good_actions = [actions[i] for i in good_action_inds]
            return _make_distribution(good_actions)

        if self._n_rules == 3:
            return _make_distribution(actions)

        # Rule 4: Set-up future winning move
        #  TODO

        # Rule 4, any side-center
        good_action_inds = []
        for act_i, action in enumerate(actions):
            if action in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                good_action_inds.append(act_i)
        if len(good_action_inds) > 0:
            good_actions = [actions[i] for i in good_action_inds]
            return _make_distribution(good_actions)

        return _make_distribution(actions)


def test_case():
    print("\n=====================================\n")
    case = Game.from_strs(["OXO", " XO", "XOX"])
    p1 = HeuristicPlayer(Mark.X, n_rules=4)

    rec = p1.recommend_action(case)
    print("Game state:")
    print(case)
    print("Player: %s\n" % p1.player.name)
    print("Recommended actions:")
    for action, prob in rec:
        print("Action: %s, Probability: %.4f" % (action, prob))
    action = p1.take_action(case)
    print("Action taken: %s" % (action,))
    print("Game state after action:")
    print(case.clone_and_move(action, Mark.X))


def test_baseline():
    p1 = HeuristicPlayer(Mark.X, n_rules=6)
    p2 = HeuristicPlayer(Mark.O, n_rules=3)
    game = Game()

    def _check(game, player):
        term = game.check_endstate()
        if term is not None:
            print("Game over with result: %s" % (term.name,))
            return True
        return False

    while True:
        print(game)
        if game.check_endstate() is not None:
            break

        action1 = p1.take_action(game)
        game = game.clone_and_move(action1, Mark.X)
        print("\nPlayer %s makes move %s\n\n%s\n" % (p1.player.name, action1, game))
        if _check(game, p1):
            break
        action2 = p2.take_action(game)
        game = game.clone_and_move(action2, Mark.O)
        print("\nPlayer %s makes move %s\n\n%s\n" % (p2.player.name, action2, game))
        if _check(game, p2):
            break

    print(game)
    print("Game over")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_baseline()
    # test_case()
