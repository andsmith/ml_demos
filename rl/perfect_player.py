"""
Given: the perfect strategy for tic tac toe never loses.

Problem:  Find pi(s), such that following pi(s) from the start ("weak solution") leads to a win or draw.
"""
import numpy as np
import logging
from game_base import Mark, Result, TERMINAL_REWARDS
from tic_tac_toe import Game, get_game_tree_cached
import pickle
from reinforcement_base import Environment
import os
from policies import Policy


class MiniMaxPolicy(Policy):
    cache_file = {Mark.X: "minimax_policy_cache_X.pkl",
                  Mark.O: "minimax_policy_cache_O.pkl"}

    def __init__(self, player_mark=Mark.X, clear_cache=False):
        """
        Initialize the MiniMax policy for tic tac toe.
        :param player_mark: The mark of the player for which this policy is created.
        """
        self.player = player_mark
        self.opponent_mark = Mark.O if player_mark == Mark.X else Mark.X

        if clear_cache or not os.path.exists(self.cache_file[player_mark]):
            logging.info("MiniMaxPolicy:  Calculating optimal policy for player %s..." % player_mark.name)
            self._pi = self._calc_opt_policy(player_mark)
            # save the policy to a file
            with open(self.cache_file[player_mark], 'wb') as f:
                pickle.dump(self._pi, f)
        else:
            logging.info("MiniMaxPolicy:  Loading optimal policy for player %s from cache..." % player_mark.name)
            with open(self.cache_file[player_mark], 'rb') as f:
                self._pi = pickle.load(f)
        dist_lens = [len(self._pi[state]) for state in self._pi]

        print("MiniMaxPolicy:  Loaded action distributions for %d states, mean dist len:  %.4f." %
              (len(self._pi), np.mean(dist_lens)))

    def _calc_opt_policy(self, player_mark):

        policy = {}  # (for player_mark) state -> [(action, prob), ...]
        mm_values = {}  # (state, player) -> value

        other_guy = {Mark.X: Mark.O, Mark.O: Mark.X}

        term_values = {Mark.X: {Result.X_WIN: 1., Result.O_WIN: -1., Result.DRAW: 0.},
                       Mark.O: {Result.X_WIN: -1., Result.O_WIN: 1., Result.DRAW: 0.}}

        def _minimax(state, player):
            """
            Minimax algorithm to find the optimal policy for the given state.
            :param state: The current state of the game.
            :param player: The current player (Mark.X or Mark.O).
            :returns: The best action and its value.
            """
            if (state, player) in mm_values:
                return mm_values[(state, player)]
            term = state.check_endstate()

            if term is not None:
                val = term_values[player_mark][term]
                mm_values[(state, player)] = val
                return val

            actions = state.get_actions()
            next_states = [state.clone_and_move(action, player) for action in actions]
            values = [_minimax(next_state, other_guy[player]) for next_state in next_states]
            if player != player_mark:
                top_val = min(values)
            else:
                top_val = max(values)
                # Policy is action w/best value. For ties, make a uniform distribution from them.
                best_inds = np.where(np.array(values) == top_val)[0]
                n_best_actions = len(best_inds)
                action_dist = [(actions[i], 1.0/n_best_actions) for i in best_inds]
                if state in policy:
                    raise Exception("State %s already in policy!" % state)
                policy[state] = action_dist
                if len(policy) % 100 == 0:
                    logging.info("Processed %d states." % len(policy))
            mm_values[(state, player)] = top_val
            return top_val

        # get the game tree
        init_state = Game()
        logging.info("Calculating optimal policy for player %s going first..." % player_mark.name)
        _minimax(init_state, player_mark)
        opp_actions = init_state.get_actions()
        logging.info("Calculating optimal policy for player %s going first..." % other_guy[player_mark].name)
        for opp_action in opp_actions:
            intermediate_state = init_state.clone_and_move(opp_action, other_guy[player_mark])
            _minimax(intermediate_state, player_mark)
        return policy

    def recommend_action(self, state):
        if state not in self._pi:
            print(state)
            print("State not in policy")
            import ipdb
            ipdb.set_trace()
            raise Exception()
        return self._pi[state]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    player_mark = Mark.X
    pi = MiniMaxPolicy(player_mark=player_mark)
