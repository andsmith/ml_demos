"""
Given: the perfect strategy for tic tac toe never loses.

Problem:  Find pi(s), such that following pi(s) from the start ("weak solution") leads to a win or draw.
"""
import numpy as np
import logging
from game_base import Mark, Result, TERMINAL_REWARDS
from tic_tac_toe import Game, get_game_tree_cached
from reinforcement_base import Environment
from policies import Policy


def extract_dynamics(self):
    """
    Find the distribution of the next REINFORCEMENT states, (not game states) for every action, for every state.
    (the set of states the agent can see on their next turn.)

    For every state, take all available actions (resulting in the "intermediate state"),
    and then use the opponent's policy to find the distribution of next states.


    :returns:  dict {state: [(action_a, [(rl_state_a1, prob_a1), (rl_state_a2, prob_a2), ...)),
                                (action_b, [(rl_state_b1, prob_b1), (rl_state_b2, prob_b2), ...)), 
                                ...]
                        }  for all nonterminal states
    """
    decision_states = self.get_nonterminal_states()
    terminal_states = {state: None for state in self.get_terminal_states()}  # for faster lookup
    policy = {}
    logging.info("Computing state transition probabilities from opponent policy & game dyanmics: ")
    for s_i,  state in enumerate(decision_states):
        agent_actions = Game.get_actions(state)
        action_dist = {}  # action -> next state distribution
        for agent_action in agent_actions:
            intermediate_state = state.clone_and_move(agent_action, self.player)
            outcome = None
            if intermediate_state in terminal_states:
                outcome = self.outcome_codes[terminal_states[intermediate_state]]
                action_dist[agent_action] = [(intermediate_state, 1.0, outcome)]  # terminal state, no next states
            else:
                opp_action_dist = self.pi_opp.recommend_action(intermediate_state)
                next_states = [(intermediate_state.clone_and_move(opp_action, self.opponent), prob)
                               for opp_action, prob in opp_action_dist]
                # Add outcome codes to the next state distribution
                next_state_dist = [(next_state, prob, self.outcome_codes[next_state.check_endstate()])
                                   for next_state, prob in next_states]

                action_dist[agent_action] = next_state_dist
        policy[state] = action_dist
    logging.info("\tfound transitions for %d states." % len(policy))

    return policy


class MiniMaxPolicy(Policy):
    def __init__(self, player_mark=Mark.X):
        """
        Initialize the MiniMax policy for tic tac toe.
        :param player_mark: The mark of the player for which this policy is created.
        """
        self.player = player_mark
        self.opponent_mark = Mark.O if player_mark == Mark.X else Mark.X
        self._pi = self._calc_opt_policy(player_mark)

    def _calc_opt_policy(self, player_mark):

        policy = {}  # (for player_mark) state -> [(action, prob), ...]
        mm_values = {}  # (state, player) -> value

        other_guy = {Mark.X: Mark.O, Mark.O: Mark.X}

        term_values = {Mark.X: {Result.X_WIN: 1., Result.O_WIN: -1., Result.DRAW: 0.},
                       Mark.O: {Result.X_WIN: -1., Result.O_WIN: 1., Result.DRAW: 0.}}
        #import ipdb; ipdb.set_trace()

        def _minimax(state, player):
            """
            Minimax algorithm to find the optimal policy for the given state.
            :param state: The current state of the game.
            :param player: The current player (Mark.X or Mark.O).
            :returns: The best action and its value.
            """
            #print("Player: %s\n%s\n" % (player, state))
            if (state,player) in mm_values:
                return mm_values[(state,player)]
            term = state.check_endstate()

            if term is not None:
                val =  term_values[player_mark][term]
                mm_values[(state,player)] = val
                return val
            

            actions = state.get_actions()
            next_states = [state.clone_and_move(action, player) for action in actions]
            values =[_minimax(next_state, other_guy[player]) for next_state in next_states]
            if player == player_mark:
                top_val = max(values)

                # remember the best actions
                best_inds = np.where(np.array(values) == top_val)[0]
                n_best_actions = len(best_inds)
                action_dist = [(actions[i], 1.0/n_best_actions) for i in best_inds]
                if state in policy:
                    raise Exception("State %s already in policy!" % state)
                policy[state] = action_dist
                if len(policy) % 100 == 0:
                    logging.info("Processed %d states." % len(policy))
            else:
                top_val = min(values)

            mm_values[(state,player)] = top_val
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
            print("State not in policy" )
            raise Exception()
        return self._pi[state]
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    player_mark = Mark.X
    pi = MiniMaxPolicy(player_mark=player_mark)
    