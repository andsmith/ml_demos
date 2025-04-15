"""
Optimize the placement of state nodes within a layer.  

In general the graph should have as vertical edges as possible, states favoring one player on a consistent side.
   
   States stay in their zones:
    * within a layer, each state is confined to its zone, can move freely within it.
    * Odd-numered layers have states with an extra X on the left, the extra O states on the right.
    * Even numbered layers have 3 zones, left (for terminal x-win), middle (for non-terminal) and right (for terminal o-win).
    * Odd numbered layers have 4 zones, left and right as above, and middle-left, middle-right for non-terminal states
      with more Xs than O's or vice versa, respectively.


Approach:

    Initialize by moving terminal and odd-marked states to their required locations.

    1. Define the quality of the layout as the sum of edge lengths from each state to its children.

    2. Run until no changes for N_BIG iterations (can be tuned with cooling schedule):

        2.a)  Select a random state s, and a random neighbor* s_n whose place s is allowed to occupy without violating the rules and vice-versa.
        2.b)  Let delta_q be the change in quality resulting from the swap.
        2.c)  Accept the change with probability exp(-delta_q / T), where T is the current temperature.

"""
import numpy as np
from game_base import Mark, Result
from tic_tac_toe import Game
import logging

class SimpleTreeOptimizer(object):
    """
    Move draw states to the center, terminal-x to the left, terminal-y to the right in each layer.
    """

    def __init__(self, image_size, states_by_layer, state_positions, terminal):
        self._terminal_LUT = terminal
        self._size = image_size
        # if in {'id','state'} format, get rid of ids, use index into one of these lists.
        #import ipdb; ipdb.set_trace()
        if isinstance(states_by_layer[0][0], dict):
            self._states_per_layer = [[state_info['state'] for state_info in layer] for layer in states_by_layer]
        else:
            self._states_per_layer = states_by_layer
        # <-- Point of this class is to find a better bijection between states and positions.
        self._xy_position_LUT = state_positions
        logging.info("Optimizing state layout for game tree with %i layers." % len(self._states_per_layer))
        self._opt_xy_position_LUT = self._optimize()

    def get_new_positions(self):
        return self._opt_xy_position_LUT

    def _get_state_zone(self, state):
        if self._terminal_LUT[state] == Result.DRAW:
            return 'middle'
        if self._terminal_LUT[state] == Result.X_WIN:
            return 'left'
        if self._terminal_LUT[state] == Result.O_WIN:
            return 'right'
        n_x = np.sum(state.state == Mark.X)
        n_o = np.sum(state.state == Mark.O)
        if n_x > n_o:
            return 'middle_left'
        elif n_x < n_o:
            return 'middle_right'
        return 'middle'

    def _optimize_layer(self, layer):
        """
        Place each state in the appropriate zone.
        """
        positions = [self._xy_position_LUT[state] for state in layer]
        position_ordered = sorted(positions, key=lambda pos: pos['x'][0]+np.random.rand()/1000)

        state_zones = {state: self._get_state_zone(state) for state in layer}
        zone_list = list(set([k for k in state_zones.values()]))
        
        zone_lut = {zone:[state for state in layer if state_zones[state] == zone] for zone in zone_list}

        new_positions = {}
        next_pos = [0]

        def _add_zone(zone_name):
            if zone_name in zone_lut:
                for state in zone_lut[zone_name]:
                    new_positions[state] = position_ordered[next_pos[0]]
                    next_pos[0] += 1
                logging.info("\t\tadded %i states to zone %s." % (len(zone_lut[zone_name]), zone_name))
        _add_zone('left')
        _add_zone('middle_left')
        _add_zone('middle')
        _add_zone('middle_right')
        _add_zone('right')
        assert next_pos[0] == len(layer), "Not all states were assigned a position!"
        # Check that no two states ended up in the same position.
        xy_positions = [pos['x']+pos['y'] for pos in new_positions.values()]
        if len(xy_positions) != len(set(xy_positions)):
            raise Exception("Two states ended up in the same position!")

        return new_positions

    def _optimize(self):
        new_positions = {}
        
        for layer in self._states_per_layer:
            logging.info("\toptimizing layer with %i states." % len(layer))
            new_layer_positions = self._optimize_layer(layer)
            new_positions.update(new_layer_positions)
        return new_positions
