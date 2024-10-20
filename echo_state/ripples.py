"""
Simplified 2-d (X, depth) simulation of wave propagation, as on the surface of still water.

Can an echo state network learn it?

Simulation:

    Define the height h of the surface of the water over position x and simulate its evolution over time h(x, t).
    Discretize time into T timestep t_step, and space X positions with spacing x_step, x_i - x_{-1} = x_step.

    For each timestep t=i, there can be zero or more "droplets", each a pair (x, a), where
    x is the center of the drop, a is the amplitude (corresponding to the K.E. of the drop).

    The entire input/output sequence is characterized by dimensions of the simulation and the sequence of
    droplets.

    The simulation is interacted with using a description (discretized vector) of the physical forces on the water
    surface at each timestep, i.e. zero if there are no droplets, else the sum of the wave functions W(x, a, x_d) for
    the forces at position x given a droplet of amplitude a at position x_d.

        W(x, a, x_d) = a - p(x - x_d) - p(x_d - x),

        where p(x) = x if x > 0, else 0 (the "positive part" function).

    I.e., an impulse's forces create a triangular wave centered at x_d with amplitude a and a 90-degree
    bend at the top/bottom (depending on the sign of a).

    Each droplet d_i creates two waves, one to the left and one to the right, propagating at a speed
    that is a function of the amplitude, (see _get_wave_speed method).  Waves disappear when no part
    of their wave function is above a certain threshold (see _get_wave_amplitude method), or when they
    propagate off the edge of the pond, unless reflection is enabled.

    Amplitudes decay exponentially with time, by multiplying by c_decay each iteration, i.e.
      a(t) = a * c_decay ^ i, where i is the number of timesteps since the droplet was released.


INPUTS:   The input vector at time t is the sum of each force vector for every droplet released at time t. (i.e.
    each only appears momentarily, in the single timestep it is released).

OUTPUTS: The height of the water at each position x and time t.
"""

import numpy as np
import matplotlib.pyplot as plt


def _positive_part(x):
    return np.maximum(x, 0)


class Wave(object):
    """
    Class to represent a wave pair created by the impact of a drop on the surface of the pond.
    """

    def __init__(self, x_0, a, x_max, speed_factor, decay_factor=0.95, reflect=(True, True), amp_thresh=0.01):
        """
        Initialize a wave pair.
        :param x_0: x-coordinate of the drop.
        :param a: amplitude of the wave.
        :param x_max: maximum x-coordinate of the pond.
        :param speed_factor: Constant in speed calculation, see _get_wave_speed
        :param reflect: Tuple of booleans, whether the wave should reflect off the left and right edges.
        """
        self._x_max = x_max
        self._speed_factor = speed_factor
        self._decay_factor = decay_factor
        self._reflect = reflect
        self._thresh = amp_thresh

        # wave states are
        self._w1 = {'x': x_0,
                    'amp': a,
                    'v': -self._get_wave_speed(a)}  # initially going left

        self._w2 = {'x': x_0,
                    'amp': a,
                    'v': self._get_wave_speed(a)}  # and right.

    def __str__(self):
        return "Wave: x1=%s, x2=%s, a1=%s, a2=%s" % (self._w1['x'], self._w2['x'], self._w1['amp'], self._w2['amp'])

    def is_active(self):
        if self._reflect[0] and self._reflect[1]:
            # enough to check amplitudes are both above threshold
            return self._w1['amp'] > self._thresh and self._w2['amp'] > self._thresh
        else:
            def _wave_visible(w):
                """
                A wave is active its center is within [0, x_max] or if its center is out of bounds but the
                closer in-bounds pixel would be above threshold.
                """
                if self._reflect[0] and self._reflect[1]:
                    return True
                if not self._reflect[0] and w['x'] < 0:
                    v = Wave._get_wave_shape(w, 0)
                    return v > self._thresh
                if not self._reflect[1] and w['x'] > self._x_max:
                    v = Wave._get_wave_shape(w, self._x_max)
                    return v > self._thresh
                return True

            return _wave_visible(self._w1) or _wave_visible(self._w2)

    def tick(self, dt):
        """
        Update wave state by one timestep:
            * Move waves according to current speed.
            * Decay amplitude.
        returns: True if wave is still active
        """
        def move_and_bounce(w, x_max):
            x, v = w['x'], w['v']
            x += v * dt

            # amp should decay by decay_factor in 1 second, so by decay_factor^dt in dt seconds.
            amp = w['amp'] * (self._decay_factor ** dt)
            if x < 0 and self._reflect[0]:
                x = -x
                v = -v
            elif x > x_max and self._reflect[1]:
                x = 2*x_max - x
                v = -v
            return {'x': x, 'v': v, 'amp': amp}

        self._w1 = move_and_bounce(self._w1, self._x_max)
        self._w2 = move_and_bounce(self._w2, self._x_max)
        return self.is_active()

    def _get_wave_speed(self, a):
        # speed is proportional to sqrt(a), according to some physics on wikipedia.
        return self._speed_factor * np.sqrt(a)

    @staticmethod
    def _get_wave_shape(w, positions):
        return _positive_part(w['amp']-np.abs(positions - w['x']))

    def get_amplitudes(self, x):
        """
        Calculate the impulse function due to this wave.
        """
        i1 = Wave._get_wave_shape(self._w1, x)
        i2 = Wave._get_wave_shape(self._w2, x)
        return i1 + i2


def get_natural_raindrops(n, t_max, x_max, amp_mean=10):
    """
    Assume raindrop sizes have a gamma distribution, k=2.0, theta = 3.0.
    """
    k = 2.0
    theta = 3.0
    mean = k*theta
    scale = amp_mean/mean

    times = (np.random.rand(n)*t_max)
    positions = (np.random.rand(n)*x_max)
    sizes = np.random.gamma(k, theta, n) * scale + 1
    return [{'t': t, 'x': p, 'mass': s} for t, p, s in zip(times, positions, sizes)]


class Pond(object):
    def __init__(self, n_x,  x_max=100., decay_factor=0.95):
        """
        Initialize a rippling pond simulation with:
        :param n_x: number of positions in the x dimension (spatial discretization)
        :param speed_factor: Constant in speed calculation, see _get_wave_speed
        :param decay_factor: Exponential decay factor for wave amplitudes over time
        :param threshold: fraction of dx, minimum threshold for wave amplitudes
        """
        self._n_x = n_x
        self._speed_factor = .1
        self._decay_factor = decay_factor
        self._threshold = .1
        self._max_x = x_max

        self._x = np.linspace(0, self._max_x, n_x)

    def get_stimulus(self, new_waves):
        """
        The "stimulus" is the sum of the new waves' amplitudes in this iteration.
        """
        h = np.zeros(self._n_x)
        for w in new_waves:
            h += w.get_amplitudes(self._x)
        return h

    def simulate(self, raindrops, t_max=1000., iter=1001):
        """
        Simulate the pond's evolution over time (until no drops are left, or max_iter is reached).
        :param raindrops: A list of raindrops, each a dict with 'times','x','mass' keys.
        """
        print("Simulating %i raindrops in x=[0, %.1f], t=[0, %.1f] over %i iterations" %
              (len(raindrops), self._max_x, t_max, iter))
        waves = []
        heights = []  # output, to be predicted by the ESN
        interactions = []  # input, to drive the ESN
        dt = t_max/(iter-1)
        times = np.linspace(0, t_max, iter)

        # get time index for each drop
        drop_schedule = np.array([int(np.floor(drop['t']/dt)) for drop in raindrops])

        for i in range(iter):
            # Upddate old waves, forgetting if they are no longer active,
            waves = [w for w in waves if w.tick(dt)]

            # add new waves,
            dropping_now = [drop for d_i, drop in enumerate(raindrops) if drop_schedule[d_i] == i]
            new_waves = [Wave(x_0=drop['x'], a=drop['mass'], x_max=self._max_x, speed_factor=self._speed_factor,
                         decay_factor=self._decay_factor, amp_thresh=self._threshold) for drop in dropping_now]
            waves.extend(new_waves)
            interactions.append(self.get_stimulus(new_waves))

            # finally, update the pond's height.
            h = np.zeros(self._n_x)
            for w in waves:
                h += w.get_amplitudes(self._x)

            heights.append(h)

        return np.array(heights)

