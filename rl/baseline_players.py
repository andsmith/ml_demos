"""
    1. random player
    2. heuristic player
    3. minimax(d) player ?

Players are implemented as Policy subclasses.
"""
import numpy as np
from tic_tac_toe import Game, Mark, Result
from policies import Policy


class RandomPlayer(Policy):
    def __init__(self, mark):  
        # deterministic=True means that the player will always return ONE action, not the same each time it sees state s.
        super(RandomPlayer, self).__init__(player=mark, deterministic=True)  

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
    """

    def __init__(self, mark):
        super(HeuristicPlayer, self).__init__(player=mark, deterministic=True)
        


    def recommend_action(self, game_state):
        actions = game_state.get_actions()

        # Randomize order of actions:
        np.random.shuffle(actions)

        # Rule 1, any winning moves?
        for action in actions:
            new_state = game_state.copy()
            new_state.move(self.player, action)
            if new_state.check_terminal() == self.winning_result:
                return action
        # Rule 2, any blocking moves?
        for action in actions:
            new_state = game_state.copy()
            new_state.move(self.opponent, action)
            term = new_state.check_terminal()
            if term in [self.opponent, Result.DRAW]:
                return action
        # Rule 3, center
        if (1, 1) in actions:
            return (1, 1)
        # Rule 4, opposite corner
        for action in actions:
            if action in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                opposite = (2 - action[0], 2 - action[1])
                if game_state.state[opposite] == self.opponent:
                    return action
        # Rule 5, any corner
        for action in actions:
            if action in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                return action
        # Rule 6, any side
        for action in actions:
            if action in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                return action
