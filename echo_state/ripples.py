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
                    v= Wave._get_wave_shape(w,0)
                    return v > self._thresh
                if not self._reflect[1] and w['x'] > self._x_max:
                    v= Wave._get_wave_shape(w,self._x_max) 
                    return v> self._thresh
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
            amp = w['amp'] * self._decay_factor
            if x < 0 and self._reflect[0]:

                x = -x
                v *= -1
            elif x > x_max and self._reflect[1]:
                x = 2*x_max - x
                v *= -1
            return {'x': x, 'v': v, 'amp': amp}

        self._w1 = move_and_bounce(self._w1, self._x_max)
        self._w2 = move_and_bounce(self._w2, self._x_max)
        return self.is_active()

    def _get_wave_speed(self, a):
        return self._speed_factor * np.sqrt(a)
    
    @staticmethod
    def _get_wave_shape(w, positions):
        return _positive_part(w['amp']-np.abs(positions - w['x']))

    def get_amplitudes(self, x):
        """
        Calculate the impulse function due to this wave.
        """
        left = Wave._get_wave_shape(self._w1, x)
        right = Wave._get_wave_shape(self._w2, x)
        return left + right


def test_wave():
    n_steps = 300
    wave = Wave(x_0=35, a=20.,  x_max=101., speed_factor=.1, decay_factor=.9995, amp_thresh=1,reflect=(True, True) )
    amps=[]
    x = np.linspace(0, 100, 101)
    for iter in range(n_steps):
        amps.append(wave.get_amplitudes(x))
        print(wave)
        if not wave.tick(dt=1.0):
            break
    img = np.array(amps)
    plt.imshow(img.T, aspect='equal', cmap='hot', interpolation='nearest')
    plt.xlabel('t');plt.ylabel('x')
    plt.title("Decayed to %.6f after %i iterations" % (np.min(img[-1,:]), iter))
    plt.axis('equal')
    plt.colorbar()
    plt.show()


class Pond(object):
    def __init__(self, n_x, speed_factor, decay_factor, dx=1., dt=1., threshold=0.1):
        """
        Initialize a rippling pond simulation with:
        :param n_x: number of positions in the x dimension (spatial discretization)
        :param speed_factor: Constant in speed calculation, see _get_wave_speed
        :param decay_factor: Exponential decay factor for wave amplitudes over time
        :param dx: spatial discretization step size
        :param dt: time discretization step size
        :param threshold: fraction of dx, minimum threshold for wave amplitudes
        """
        self._n_x = n_x
        self._dx = dx
        self._dt = dt
        self._speed_factor = speed_factor
        self._decay_factor = decay_factor
        self._threshold = threshold

    def _get_wave_speed(self, a):
        """
        Calculate the wave speed for a wave of amplitude a.
        """
        return self._speed_factor * np.sqrt(a)

    def _get_wave_amplitude(self, a, t):
        """
        Calculate the wave amplitude at time t for a wave of amplitude a.
        """
        return a * self._decay_factor ** t

    # Methods for creating specific simulation scenaiors (droplet placements)

    def make_it_trickle(self, n_steps, pos_rel=0.5, rate=10):
        """
        Simulate drops regularly falling in one position.
        :param n_steps: number of timesteps to simulate
        :param pos_rel: relative position of the droplet in the pond wrt endpoints.
        :param rate: number of timesteps between drops.
        :returns: list of (x_ind, a) tuples, where x_ind is the index of the droplet's position in 
           the spatial discretization, and a is the amplitude of the droplet's wave.
        """
        self.h = np.zeros((self.n_x, 1))
        self.droplets = [(int(self.n_x * pos_rel), 1.0)]
        for i in range(n_steps):
            if i % rate == 0:
                self.droplets.append((int(self.n_x * pos_rel), 1.0))
            self._step()

    def make_it_rain(self, precipitation_level, n_steps):
        """

        """

        self.h = np.zeros((n_x, 1))


if __name__ == "__main__":
    test_wave()
