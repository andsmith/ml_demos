from policies import Policy
from game_base import get_reward, Mark, TERMINAL_REWARDS
import numpy as np


class ValueFuncPolicy(Policy):
    """
    The greedy policy for a given value function recommends the action with best expected reward + discounted value.

    This class can be used as the value function itself by referring to DPValueFunc.v[state] and also as a
      Policy object acting so the next state has highest expected value according to v(), by calling the inherited
      method ValueFuncPolicy.recommend_action(state).

    The action recommended by the policy function is the action that
    maximizes the value function of the resulting state, i.e.
        Policy(s) = argmax_a V(T(s,a)), where
        T(s,a) = state resulting from taking action a in state s.
        (equation 4.9 in Sutton and Barto, 2nd edition)

    If more than one next state has the maximal values, all are returned with 
    uniform probability, or one is selected arbitrarily if deterministic.
    """

    def __init__(self, v, environment, gamma=1.0, player=Mark.X):
        """
        :param v: dict, hash of Game (state) to value.  The value function.
        :param environment:  The environment for the agent.  This is used to calculate the next state distribution.
        :param player: Mark.X or Mark.O  The player for whom the policy is optimized.
        :return: None
        """
        super().__init__(player=player)  # default player is X
        self._v = v  # hash of Game (state) to value
        self._pi = self._optimize()
        self._env = environment
        self._gamma = gamma

    def _optimize(self):
        """
        For all nonterminal states, get the optimal action (or multiple if they tie)

        Each avaliable action will lead to one opponent (non RL) state.
        Each post-action state will have a distribution of next-states. 

        :returns: dict, hash of Game (state) to [(action, prob), ...] 
        """
        pi = {}
        for state in self._env.get_nonterminal_states():

            actions = state.get_actions()
            best = {'actions': [],
                    'rewards': []}  # in case of ties, multiple are "best"

            for action in actions:
                next_state = state.clone_and_move(action, self._player)
                next_result = next_state.check_endstate()
                if next_result == self.winning_result:
                    reward_term = TERMINAL_REWARDS[self.winning_result]
                elif next_result == self.draw_result:
                    reward_term = TERMINAL_REWARDS[self.draw_result]
                else:
                    # opponent's turn, get the probability distribution of next states
                    opp_moves = self._env.pi_opp.recommend_action(next_state)
                    # take the weighted sum over every action the opponent might make of v(s')
                    reward_term = 0.0
                    for opp_action, prob in opp_moves:
                        next_next_state = next_state.clone_and_move(opp_action, self._env.opponent)
                        next_next_result = next_next_state.check_endstate()
                        if next_next_result == self.losing_result:
                            reward_term += prob * TERMINAL_REWARDS[self.losing_result]
                        elif next_next_result == self.draw_result:
                            reward_term += prob * TERMINAL_REWARDS[self.draw_result]
                        else:
                            reward_term += prob * self._v[next_next_state] * self._gamma
                if len(best['actions']) == 0 or reward_term > max(best['rewards']):
                    best['actions'].append(action)
                    best['rewards'].append(reward_term)

            # now determine if there's a winner or distribution of actions
            highest_reward = max(best['rewards'])
            if len(best['actions']) == 1:
                pi[state] = [(best['actions'][0], 1.0)]
            else:
                num_best = np.sum(np.array(best['rewards']) == highest_reward)
                best_actions = [act for i, act in enumerate(best['actions']) if best['rewards'][i] == highest_reward]
                prob = 1.0 / num_best
                pi[state] = [(best_action, prob) for best_action in best_actions]

        return pi

    def recommend_action(self, state):
        """
        Return actions to take given the current state.

        The greedy policy selects the action 
        """
        return self._pi[state]
