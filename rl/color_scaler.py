"""
Determine range of colors corresponding to range of values.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

from colors import SKY_BLUE

class ColorScaler(object):

    def __init__(self, values, cmap_name='gray', undef_val_color=SKY_BLUE, as_delta=False):
        """
        Balance colors so a range of values spans the whole range of the colormap.

        Update by accumulating new values, rebalance as required.

        :param initial values: array of all initial values to be colored.
        :param cmap_name:  The name of the colormap to use.
        """
        self._as_delta = as_delta
        distinct_values = np.unique(values) 
        #logging.info("Initializing ColorScaler with %i values." % len(distinct_values))
        # logging.info("ColorScaler values: %s" % str(distinct_values))

        #if len(distinct_values)==1:
        #    import ipdb; ipdb.set_trace()
        #if len(distinct_values) < 2:
        #    raise ValueError("ColorScaler requires at least two distinct values.")
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
            raise Exception("ColorScaler range updated: min=%f, max=%f" % (self._min, self._max))
            logging.info("ColorScaler updated: min=%f, max=%f" % (self._min, self._max))
        return self.scale_color(new_value)

    def scale_value(self, value,alpha=1.0):
        """
        Scale a value to unit interval.
        """
        if self._as_delta and value == 0.0:
            #import ipdb; ipdb.set_trace()
            return None
        scaled= (value - self._min) / self._range
        scaled = np.clip(scaled, 0.0, 1.0)
        return scaled ** alpha

    def scale_color(self, value):
        """
        Scale a value to a color in the colormap.
        """
        v_scaled = self.scale_value(value)
        if v_scaled is None:
            return None
        c_float = np.array(self._cmap(v_scaled)).reshape(-1)[:3]
        return int(255*c_float[0]), int(255*c_float[1]), int(255*c_float[2])


    def get_LUT(self, state_value_map):
        return {state: self.scale_color(value) for state, value in state_value_map.items()}

