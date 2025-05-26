from baseline_players import HeuristicPlayer
from policy_optim import ValueFuncPolicy
from game_base import Mark, Result
import numpy as np
import logging
from reinforcement_base import Environment
from tic_tac_toe import Game


def test_baseline_players_different(verbose=False):
    logging.info("Testing: all heuristc players are different.")
    players = [HeuristicPlayer(Mark.X, n_rules=p) for p in [0, 1, 2, 3, 4]]
    opp = HeuristicPlayer(Mark.O, n_rules=4)
    env = Environment(opponent_policy=opp, player_mark=Mark.X)
    updatable_states = env.get_nonterminal_states()

    differences = np.zeros((len(players), len(players)), dtype=int)
    if verbose:
        print("Comparing %i baseline policies on %i states:" % (len(players), len(updatable_states)))
    for i1, p1 in enumerate(players):
        for i2, p2 in enumerate(players):
            if i2 < i1:
                continue
            differences[i1, i2] = p1.compare(p2, updatable_states, count=True, deterministic=True)
            if verbose:
                print("\t%i vs %i:  %i different choices" % (i1, i2, differences[i1, i2]))

    if verbose:
        print("Difference matrix comparing %s -- %s on %i states:" % (players[0], players[1],len(updatable_states)))
        print(differences)
    diff = differences + differences.T
    assert np.all(np.diag(diff) == 0), "Diagonal of difference matrix should be zero."
    assert np.all(diff[diff != 0]) > 0, "Off-diagonal of difference matrix should be nonzero."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_baseline_players_different(verbose=True)
    logging.info("All tests passed.")
