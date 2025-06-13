"""
Base classes and functions for RL demos.
"""
from tic_tac_toe import Game, get_game_tree_cached, GameTree
from game_base import Mark, Result
from enum import IntEnum
import numpy as np
import logging


class PIPhases(IntEnum):
    VALUE_F_OPT = 0
    POLICY_OPT = 1


class Environment(object):
    """
    The Environment class encapsulates the rules of Tic Tac Toe as well as the 
    policies of the opponent our RL agent is learning to play against.

    Each timestep, the agent takes an action (makes a move) based on the current state.
    The environment responds with a new state (the game board after the move AND the opponent's move distribution),
    and a reward.

    Two-player zero sum games are more properly modeled as an Alternating Markov Game. 
    This class shoehorns the game into a Markov Decision Process (MDP) by assuming the opponent is a stochastic policy.
    """

    def __init__(self, opponent_policy=None, player_mark=Mark.X):
        """
        :param opponent_policy:  policy of the opponent, Policy object initialized with the opposite player mark.  Will be 
            used to define the "environment" for the agent.
        :param player_mark: The player for the agent.  The opponent is the other player.
        """
        self.player = player_mark
        self.opponent = GameTree.opponent(player_mark)

        self.winning_result = Result.X_WIN if player_mark == Mark.X else Result.O_WIN
        self.losing_result = Result.O_WIN if player_mark == Mark.X else Result.X_WIN
        self.draw_result = Result.DRAW
        self.outcome_codes = {self.winning_result: 1.0,
                              self.losing_result: -1.0,
                              self.draw_result: 0.0,
                              None: None}

        self.pi_opp = opponent_policy
        self.tree = get_game_tree_cached(player=player_mark, verbose=True)
        self.terminal, self.children, self.parents, self.initial = self.tree.get_game_tree()

    def opp_move_dist(self, state, action):
        """
        Given player taking the action from the state, return the distribution of next states. 
        THis is found by evaluating the opponents policy on the state after the action is taken.
        :param state: The current state of the game.
        :param action: The action taken by the player.
        :returns: a list of (opp_action, prob) tuples, from the opponent's policy, or
            None if player's action results in a terminal state.
        """
        # Get the next state after the player takes the action.
        next_state = state.clone_and_move(action, self.player)

        # Check if the game is over.  If so, return a terminal state with no next states.
        result = next_state.check_endstate()
        if result is not None:
            return None

        # Get the distribution of next states from the opponent's policy.
        opp_moves = self.pi_opp.recommend_action(next_state)

        # Filter out zero probability moves
        return opp_moves

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

    def set_opp_policy(self, policy):
        """
        Set the opponent's policy to a new policy.
        :param policy: The new opponent policy.
        """
        self.pi_opp = policy

    def _state_valid(self, state):
        """
        Check if it's really agent's turn.
        """
        return np.sum(state.state == self.player) <= np.sum(state.state == self.opponent)

    def get_terminal_states(self):
        return [state for state in self.terminal if self.terminal[state] is not None]

    def get_nonterminal_states(self):
        # states player can see in a game (when it's their turn)
        return [state for state in self.terminal if (self.terminal[state] is None and self._state_valid(state))]


def test_environment():
    """
    Test the Environment class.
    """
    from baseline_players import HeuristicPlayer
    from game_base import Mark

    agent_policy = HeuristicPlayer(mark=Mark.X, n_rules=1)
    opponent_policy = HeuristicPlayer(mark=Mark.O, n_rules=1)
    env = Environment(opponent_policy=opponent_policy, player_mark=Mark.X)

    terminals, nonterminals = env.get_terminal_states(), env.get_nonterminal_states()
    values = {state: 1.0 for state in terminals}  # dummy values for testing
    values.update({state: np.random.randn() for state in nonterminals})

    logging.info("Terminal states: %d, Nonterminal states: %d", len(terminals), len(nonterminals))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_environment()
    # Uncomment the next line to run the environment test
    # test_environment()
