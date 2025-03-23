"""
Calculate the value function for a given policy using dynamic programming.
Use the HeuristicPolicy as the policy to learn.
"""
import numpy as np
from tic_tac_toe import Game, Mark, Result
from policies import Policy, ValueFuncPolicy
from abc import ABC, abstractmethod

class DPValueFunc(ValueFuncPolicy):
    """

    Calculate the value function from some other policy.

    

    Give terminal winning states a value of 1, losing/draw states a value of 0.
    Inductive step:
        For all states s whose value v(s) has been assigned:
            Let Parents(s) be the set of states in which the policy will recommend acting to reach s.
            For all s' in Parents(s):
                if v(s') has not been assigned:
                    v(s') = v(s)*gamma
            

    
    """
    def __init__(self, policy, player, gamma=1.0):
        """
        :param policy: Policy object
        :param player: Mark.X or Mark.O
        :param gamma: discount factor, in (0, 1]
        """
        self._policy = policy
        self._gamma = gamma
        self._player = player




        