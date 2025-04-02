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
        # get rid of ids, use index into one of these lists.
        self._states_per_layer = [[state_info['state'] for state_info in layer] for layer in states_by_layer]
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

'''
class TreeOptimizer(SimpleTreeOptimizer):
    def __init__(self, image_size, states_by_layer, state_positions, terminal):
        """
        Run the TreeOptimizer.

        :param image_size: tuple of (width, height) for the image size.
        :param states_by_layer: list of L-list for the L layers, each layer's list is of {'state': Game obj., 'id': state_id} dicts.
        :param state_positions: dict of state (Game object) to the (x,y) coordinate (floats) for each state.
        :param terminal: dict of state_id to the Result (or None) for each state.

        """
        super(TreeOptimizer, self).__init__(image_size, states_by_layer, state_positions, terminal)

        self._opt_xy_position_LUT = self._get_optimal_positions()

    def _get_optimal_positions(self):
        """
        Combine the assingment lists and position lists to get final positions for everything.
        Check to make sure no two states ended up in the same position.
        :returns: a permutation of xy_position_LUT with things in better places.
          i.e. dict {state: {'x':(), 'y':()} for all states}
        """
        opt_xy_positions = {}
        for l_ind, layer in enumerate(self._states_per_layer):
            # Get the positions for this layer.
            positions = self._position_lists[l_ind]
            # Assign the states to the positions.
            assignment_list = self._assignment_lists[l_ind]
            for state_ind, state in enumerate(layer):
                pos_ind = assignment_list[state_ind]
                opt_xy_positions[state] = positions[pos_ind]

        # Check that no two states ended up in the same position.
        xy_positions = [pos['x']+pos['y'] for pos in opt_xy_positions.values()]
        if len(xy_positions) != len(set(xy_positions)):
            raise Exception("Two states ended up in the same position!")

        return opt_xy_positions

    def _set_zones_positions(self, zone_layers):
        """
        For each layer, 
            - figure out which XY positions go in which zones.
            - sort the positions by x-coordinate.
            - Assign the first |zone_layers[layer]['left']| states to the right most positions, etc.

        Sets:
           * self._position_lists[layer] = [{'x':(),'y':()}, ...]    (list of all the positions in the layer)
           * self._assignment_lists[layer] = [a_0, a_1, ...]  (state in self._state_per_layer[layer][a_i] goes to position self._position_lists[layer][a_i])

        :param zone_layers: list of dicts, keys are zone names, values are dicts(state_id: state, ...).
        """
        self._position_lists = [None] * len(self._states_per_layer)
        self._assignment_lists = [None] * len(self._states_per_layer)
        # self._state_assign_inds = {}
        for l_ind, layer in enumerate(self._states_per_layer):
            # Get the positions for this layer.
            positions = [self._xy_position_LUT[state] for state in layer]
            # Sort the positions by x-coordinate.
            positions.sort(key=lambda pos: pos['x'][0]+np.random.rand()/1000)
            self._position_lists[l_ind] = positions
            # Assign the states to the positions.
            # state in self._assignment_lists[l_ind][i] goes to position self._position_lists[l_ind][i]
            self._assignment_lists[l_ind] = []
            next_pos = [0]

            def _add_zone(zone_name):
                if zone_name in zone_layers[l_ind]:
                    for state in zone_layers[l_ind][zone_name]:
                        self._assignment_lists[l_ind].append(next_pos[0])
                        # self._state_assign_inds[state_id] = len(self._assignment_lists[l_ind]) - 1
                        next_pos[0] += 1
            _add_zone('left')
            _add_zone('middle_left')
            _add_zone('middle')
            _add_zone('middle_right')
            _add_zone('right')

    def _get_layer_zones(self, layer):
        """
        Partition each layer into zones.  (see above for details)

        :returns: dict('zone_name': [position_index, ...], ...)  (where the XY position is in pos_per_layer[l_ind][position_index])
        """
        # Count terminal states in the layer.
        zones = [self._get_state_zone(state) for state in layer]
        zone_list = list(set(zones))
        zone_dict = {zone: [] for zone in zone_list}
        for zone, state in zip(zones, layer):
            zone_dict[zone].append(state)

        return zone_dict

    def _optimize(self, n_iter=100000, t_start=100.0):
        """
        1.  Figure out which states can go into which positions within each layer. (zones are one of (left, right, middle, all) for each layer)
        2.  Place everything in some arbitrary initial position (respecting the layers/zones).
        2.  Iterate the simulated annealing.
        """
        # 1. Set each state's zone.
        zone_layers = [self._get_layer_zones(layer) for layer in self._states_per_layer]

        # 2. do assignment
        self._set_zones_positions(zone_layers)

        # 3.  Run the simulated annealing.
        # Cooling schedule, find multiple m such that
        n_swapped = 0
        total_delta_q = 0
        update_interval = 10000

        empty_state = Game.from_strs(["   ", "   ", "   "])

        sample_states = [state for layer in self._states_per_layer for state in layer if state != empty_state]

        def _get_random_state(other_state=None):
            if other_state is None:
                return np.random.choice(sample_states, size=1, replace=False)[0]
            # get one from the same zone:
            layer_ind = other_state.n_marked()
            zone = self._get_state_zone(other_state)
            swappable = zone_layers[layer_ind][zone]
            if len(swappable) < 2:
                raise Exception("Shouldn't be swapping layers with only 1 state...")
            swap_copy = [state for state in swappable if state != other_state]

            return np.random.choice(swap_copy, size=1, replace=False)[0], (layer_ind, zone)

        for iter in range(0):
            # Select a random state s, and a random neighbor* s_n whose place s is allowed to occupy without violating the rules and vice-versa.
            # Let delta_q be the change in quality resulting from the swap.
            # Accept the change with probability exp(-delta_q / T), where T is the current temperature.
            t = t_start * (1 - iter / n_iter)  # linear cooling schedule?

            state_1 = _get_random_state()
            state_2, (layer_ind, _) = _get_random_state(state_1)

            delta_q = np.random.randn(1)  # self._get_delta_q(state_1, state_2)
            if delta_q < 0 or np.random.rand() < np.exp(-delta_q / t):
                # swap the states
                # Look up their indices in the assignment list.

                state_1_ind = self._states_per_layer[layer_ind].index(state_1)
                state_2_ind = self._states_per_layer[layer_ind].index(state_2)

                # Swap the indices in the assignment list.
                self._assignment_lists[layer_ind][state_1_ind], self._assignment_lists[layer_ind][state_2_ind] = \
                    self._assignment_lists[layer_ind][state_2_ind], self._assignment_lists[layer_ind][state_1_ind]

                n_swapped += 1
                total_delta_q += delta_q

            if iter % update_interval == 0:
                print(
                    f"Iteration {iter}, swapped {n_swapped} times since last update, mean delta_q per swap:  {total_delta_q / n_swapped if n_swapped > 0 else 0:.2f}")
                total_delta_q = 0
                n_swapped = 0

            pass
'''