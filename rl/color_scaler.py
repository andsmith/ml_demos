"""
Determine range of colors corresponding to range of values.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

class ColorScaler(object):

    def __init__(self, state_value_map, cmap_name='viridis'):
        """
        Balance colors so a range of values spans the whole range of the colormap.

        Update by accumulating new values, rebalance as required.

        :param state_value_map:  The mapping of states to values (dict)
        :param cmap_name:  The name of the colormap to use.
        """
        logging.info("Initializing ColorScaler with %i states." % len(state_value_map))
        self._cmap = plt.get_cmap(cmap_name)
        self._state_value_map = state_value_map
        self.balance()

    def update(self, state, new_value):
        """
        Value changed, calculate new color, update LUT.
        :param state:  The state to update.
        :param new_value:  The new value for the state.
        :return: The color for the new value.
        """
        self._state_value_map[state] = new_value
        color = self.scale_color(new_value)
        self.color_LUT[state] = color
        return color

    def scale_value(self, value):
        return (value - self._min) / self._range

    def scale_color(self, value):
        v_scaled = self.scale_value(value)
        c_float = np.array(self._cmap(v_scaled)).reshape(-1)[:3]
        return int(255*c_float[0]), int(255*c_float[1]), int(255*c_float[2])

    def balance(self):
        """
        Recalculate so everything is in range.
        """
        values = np.array([self._state_value_map[state] for state in self._state_value_map])
        self._min, self._max = np.min(values), np.max(values)
        if self._min == self._max:
            self._min = 0.0
            self._max = 1.0
        self._range = self._max - self._min

        self.color_LUT = {state: self.scale_color(value) for state, value in self._state_value_map.items()}

