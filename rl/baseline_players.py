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
    4. If the opponent is in a corner, take the opposite corner.
    5. Take any open corner.
    6. Take any open side.

    Let a be the set of all possible actions for the agent given a game state.
    Let r be the lowest-numbered rule that applies to an action.
    let a_r be the set of actions that apply to rule r.
    Let a_g be the complement, a - a_r.
    let p_g be the probability of using an action not applying to any rules ("giving up") when there are rules to apply.

    Then the probability for each action in a_r  is (1 - p_g)/|a_r|, and
    the probability for each action in a_r is p_g/|a_g|.

    I.e. the majority of probability mass (1-p_g) is given to the actions that apply to the rule and 
    the rest goes to the actions that do not apply to the rule.

    """

    def __init__(self, mark, n_rules=6, p_give_up=1e-2):
        """
        :param mark:  one of Mark.X or Mark.O  Player optimizes actions for this player.
        :param n_rules:  number of rules to apply.  If n_rules=0, then just return  all actions with uniform probability.
        :param p_give_up:  probability of giving up if no rules apply.  (default=1e-6)
        """
        self._p_give_up = p_give_up
        self._n_rules = n_rules
        super(HeuristicPlayer, self).__init__(player=mark)

    def __str__(self):
        return "HeuristicPlayer(%s, n_rules=%d)" % (self.player.name, self._n_rules)

    def take_action(self, game_state, p_bad_move=0):
        recommendations = self.recommend_action(game_state)
        probs = np.array([prob for _, prob in recommendations])
        actions = np.array([action for action, _ in recommendations])
        action_inds = np.arange(len(actions))
        # sample from the distribution
        action_ind = np.random.choice(action_inds, p=probs)
        return tuple(actions[action_ind])

    def recommend_action(self, game_state):
        """

        :param game_state:  A Game object, it is the current player's turn.
        :returns: list of (action, probability) for all possible actions player can make.
        """
        actions = game_state.get_actions()

        def _give_up():
            # return uniform distribution over all actions
            return [ap for ap in zip(actions, np.ones(len(actions)) / len(actions))]

        def _get_probs(good_action_inds):
            n_good = len(good_action_inds)
            n_bad = len(actions) - n_good
            if n_good == 0 or n_bad == 0:
                return [ap for ap in zip(actions, np.ones(len(actions)) / len(actions))]
            probs = np.zeros(len(actions)) + self._p_give_up / n_bad
            probs[good_action_inds] = (1 - self._p_give_up) / n_good
            return [ap for ap in zip(actions, probs / np.sum(probs))]

        if self._n_rules == 0:
            return _give_up()

        # Rule 1, any winning moves?
        player_next_states = [game_state.clone_and_move(action, self.player) for action in actions]
        endstates = [state.check_endstate() for state in player_next_states]
        good_action_i = [action_i for action_i, term in enumerate( endstates) if term == self.winning_result]
        if len(good_action_i) > 0:
            return _get_probs(good_action_i)

        if self._n_rules == 1:
            return _give_up()

        # Rule 2, any blocking moves?
        opponent_next_states = [game_state.clone_and_move(action, self.opponent) for action in actions]
        opponent_endstates = [state.check_endstate() for state in opponent_next_states]
        good_action_i = [action_i for action_i, term in enumerate(opponent_endstates) if term == self.losing_result]
        if len(good_action_i) > 0:
            return _get_probs(good_action_i)

        if self._n_rules == 2:
            return _give_up()

        # Rule 3, center
        if (1, 1) in actions:
            ind = actions.index((1, 1))
            return _get_probs([ind])

        if self._n_rules == 3:
            return _give_up()

        # Rule 4, opposite corner
        good_action_i = []
        for act_i, action in enumerate(actions):
            if action in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                opposite = (2 - action[0], 2 - action[1])
                if game_state.state[opposite] == self.opponent:
                    good_action_i.append(act_i)

        if len(good_action_i) > 0:
            return _get_probs(good_action_i)

        if self._n_rules == 4:
            return _give_up()

        # Rule 5, any corner
        good_action_i = []
        for act_i, action in enumerate(actions):
            if action in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                good_action_i.append(act_i)
        if len(good_action_i) > 0:
            return _get_probs(good_action_i)

        # Rule 6, any side
        good_action_i = []
        for act_i, action in enumerate(actions):
            if action in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                good_action_i.append(act_i)
        if len(good_action_i) > 0:
            return _get_probs(good_action_i)

        return _give_up()

def test_case():
    print("\n=====================================\n")
    case = Game.from_strs(["OXO", " XO", "XOX"])
    p1 = HeuristicPlayer(Mark.X, n_rules=6)

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
            print("Game over with result: %s" % ( term.name,))
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
    #test_case()
