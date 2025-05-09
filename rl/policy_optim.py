from policies import Policy
from game_base import get_reward, Mark, TERMINAL_REWARDS
import numpy as np
from step_visualizer import PIStateStep
import logging
from tic_tac_toe import Game


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

    def __init__(self,  v, environment, old_policy, gamma=1.0, player=Mark.X):
        """
        :param v: dict, hash of Game (state) to value.  The value function.
        :param environment:  The environment for the agent.  This is used to calculate the next state distribution.
        :param player: Mark.X or Mark.O  The player for whom the policy is optimized.
        :return: None
        """
        super().__init__(player=player)  # default player is X
        #self._app = learn_app  # game_learn.PolicyImprovementDemo object
        #self._gui = learn_app.get_gui() if learn_app is not None else None
        self._v = v  # hash of Game (state) to value
        self._env = environment
        self._gamma = gamma
        self._pi_old = old_policy  #
        self._pi = self._optimize()

    def equals(self, other):
        states = self._env.get_nonterminal_states()
        return super().compare(other, states, count=False, deterministic=True)

    def _optimize(self):
        """
        For all nonterminal states, get the optimal action (or multiple if they tie)

        Each avaliable action will lead to one opponent (non RL) state.
        Each post-action state will have a distribution of next-states. 

        :returns: dict, hash of Game (state) to [(action, prob), ...] 
        """
        pi = {}
        
        for s_ind,state in enumerate(self._env.get_nonterminal_states()):
            if s_ind % 100 == 0:
                logging.info("Optimizing policy for state %i of %i" % (s_ind, len(self._env.get_nonterminal_states())))

            # if state == Game.from_strs(["XOO", "X  ", "   "]):
            #    import ipdb
            #    ipdb.set_trace()

            old_action_dist = self._pi_old.recommend_action(state)

            def get_reward_term_and_next_states(action):
                next_state = state.clone_and_move(action, self.player)
                next_result = next_state.check_endstate()
                next_states = []
                if next_result is not None:
                    # Terminal states have zero value so just use the reward.
                    reward_term = TERMINAL_REWARDS[next_result]
                else:
                    # Nonterminal means opponent's turn, get the probability distribution of next states:
                    opp_moves = self._env.opp_move_dist(state, action)
                    # Filter out zero probability moves
                    # Take the expected value over opponent moves of v(s')
                    reward_term = 0.0
                    for opp_action, prob in opp_moves:
                        if prob == 0:
                            # TODO: Don't include zero probability actions in the next state distribution.
                            continue
                        next_next_state = next_state.clone_and_move(opp_action, self._env.opponent)
                        next_next_result = next_next_state.check_endstate()

                        next_states.append((next_next_state, prob))

                        if next_next_result == self.losing_result:
                            reward_term += prob * TERMINAL_REWARDS[self.losing_result]
                        elif next_next_result == self.draw_result:
                            reward_term += prob * TERMINAL_REWARDS[self.draw_result]
                        else:
                            reward_term += prob * self._v[next_next_state] * self._gamma

                return reward_term, next_states

            actions = state.get_actions()
            reward_terms, next_states = [], []
            for action in actions:
                r_term, next_state_dist = get_reward_term_and_next_states(action)
                reward_terms.append(r_term)
                next_states.append(next_state_dist)

            best_ind = np.argmax(reward_terms)
            best_action = actions[best_ind]
            best_reward = reward_terms[best_ind]

            # now determine if there's a winner or distribution of actions
            if len(actions) == 1:
                pi[state] = [(actions[0], 1.0)]  # only one action available
            else:
                num_best = np.sum(np.array(reward_terms) == best_reward)
                best_actions = [act for i, act in enumerate(actions) if reward_terms[i] == best_reward]
                prob = 1.0 / num_best
                pi[state] = [(best_action, prob) for best_action in best_actions]

            #if self._app is not None:
                #    PIStateStep(demo, gui, state, actions, next_states, rewards, old_action, new_action):

            #    vis = PIStateStep(self._app, self._gui, state, actions, next_states, reward_terms, old_actions=old_action_dist,
             #                     new_actions=pi[state])

            #self._app.maybe_pause('state-update', vis)

        return pi

    def recommend_action(self, state):
        """
        Return actions to take given the current state.

        The greedy policy selects the action 
        """
        return self._pi[state]


def test_value_func_policy():
    """
    Test the ValueFuncPolicy class.

    :return: None
    """
    # Create a test environment and value function
    from reinforcement_base import Environment
    from baseline_players import HeuristicPlayer
    seed_policy = HeuristicPlayer(n_rules=2, mark=Mark.X)  # Replace with your seed policy
    opponent_policy = HeuristicPlayer(n_rules=6, mark=Mark.O)  # Replace with your opponent policy

    env = Environment(opponent_policy=opponent_policy, player_mark=Mark.X)
    v = {state: np.random.rand() for state in env.get_nonterminal_states()}

    # Create a ValueFuncPolicy instance
    policy = ValueFuncPolicy(v=v, environment=env, player=Mark.X)

    # Test the recommend_action method
    state = env.get_nonterminal_states()[0]
    action = policy.recommend_action(state)

    print(f"Recommended action for state {state}: {action}")


if __name__ == "__main__":
    test_value_func_policy()
