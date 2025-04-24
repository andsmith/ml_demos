"""
Determine range of colors corresponding to range of values.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

class ColorScaler(object):

    def __init__(self, values, cmap_name='gray'):
        """
        Balance colors so a range of values spans the whole range of the colormap.

        Update by accumulating new values, rebalance as required.

        :param initial values: array of all initial values to be colored.
        :param cmap_name:  The name of the colormap to use.
        """
        logging.info("Initializing ColorScaler with %i values." % len(values))
        self._cmap = plt.get_cmap(cmap_name)
        values = np.array(values)
        self._min = values.min()
        self._max = values.max()
        if self._min == self._max:
            self._min = 0.0
            self._max = 1.0
        self._range = self._max - self._min

    def get_min_max_colors(self):
        return self.scale_color(self._min), self.scale_color(self._max)

    def update_and_get_color(self, new_value):
        remapped = False
        if new_value < self._min:
            self._min = new_value
            remapped = True
        if new_value > self._max:
            self._max = new_value
            remapped = True
        if remapped:
            self._range = self._max - self._min
            logging.info("ColorScaler updated: min=%f, max=%f" % (self._min, self._max))
        return self.scale_color(new_value)

    def scale_value(self, value):
        """
        Scale a value to unit interval.
        """
        return (value - self._min) / self._range

    def scale_color(self, value):
        """
        Scale a value to a color in the colormap.
        """
        v_scaled = self.scale_value(value)
        c_float = np.array(self._cmap(v_scaled)).reshape(-1)[:3]
        return int(255*c_float[0]), int(255*c_float[1]), int(255*c_float[2])


    def get_LUT(self, state_value_map):
        return {state: self.scale_color(value) for state, value in state_value_map.items()}

