from ripples import Pond, Wave, get_natural_raindrops
import cv2
import numpy as np

WIN_WIDTH = 500
LAYOUT = {'win_size': (WIN_WIDTH, 1000),
          'midline': 0.3,  # upper portion for live heights & interaction, lower for history of heights
          'win_name': 'Pond'}


class InteractivePond(Pond):
    """
    A Pond that can be interacted with in realtime.
    """

    # def __init__(self, n_x=500, x_max=100, decay_factor=.98,wave_scale=3, a_max=30):
    def __init__(self, n_x=WIN_WIDTH, x_max=100, decay_factor=.97, wave_scale=2., a_max=30, *args, **kwargs):
        super(InteractivePond, self).__init__(n_x, x_max, decay_factor,
                                              wave_scale=wave_scale, speed_factor=1.0, *args, **kwargs)
        self._new_drops = []  # dropped since last frame
        self._win_size = LAYOUT['win_size']
        self._midline = int(LAYOUT['midline'] * self._win_size[1])
        self._win_name = LAYOUT['win_name']
        self._y_scale = 3*a_max
        self._min_y_scale = 3*a_max
        self._n_underscaled_frames = 0

        self._rain_rate = 0.0

        self._a_max = a_max
        self._n_hist_disp = self._win_size[1] - self._midline

        self._mouse_pos = None
        self._blank_row = np.zeros(self._n_x)
        self._blank_frame = np.zeros((self._win_size[1], self._win_size[0], 3), dtype=np.uint8)

        # animation
        self._dt = 0.1  # time step in seconds

        # state:
        self._waves = []  # list of active Wave objects
        self._interactions = []  # list of input vectors, to drive the ESN
        self._heights = []  # outputs, to be predicted by the ESN / plotted
        self._h_history = []  # history of height vectors

    def _mouse_pos_to_xa(self, x, y):
        """
        Convert mouse position to x and amplitude for the new drop.
        x is in [0, x_max], a is in [0, a_max].
        (input position is when y is above the midline)
        """
        x = int(x / self._win_size[0] * self._max_x)
        a = int((1 - y / self._midline) * self._a_max)
        if y > self._midline:
            return None, None
        return x, a

    def _mouse(self, event, x, y, flags, param):
        drop_x, drop_a = self._mouse_pos_to_xa(x, y)
        if drop_x is not None:
            self._mouse_pos = (x, y)  # remember for rendering
            if event == cv2.EVENT_LBUTTONDOWN:
                self._new_drops.append({'x': drop_x, 'mass': drop_a})
                print("New wave: %i at x=%i, a=%i" % (len(self._waves), drop_x, drop_a))

        else:
            self._mouse_pos = None

    def render(self, cur_heights):
        img = self._blank_frame.copy()
        img[:self._midline, :, :] = self._render_heights(cur_heights)
        img[self._midline:, :, :] = self._render_history()
        return img

    def _render_heights(self, state_vec):
        """
        Scale vertically to fit the window and plot as a line (i.e. a vertical slice of the pond).
        """
        img = np.zeros((self._midline, self._n_x, 3), dtype=np.uint8)
        h = state_vec
        highest = np.max(h)
        if highest < self._y_scale/2:
            self._n_underscaled_frames += 1
            if self._n_underscaled_frames > 30:
                if self._y_scale < self._min_y_scale:
                    self._y_scale = self._y_scale/2
                    self._n_underscaled_frames = 0
        elif highest > self._y_scale:
            while highest > self._y_scale:
                self._y_scale *= 2

        def _h_to_y(h):
            return (1 - (h/self._y_scale)) * self._midline
        y = _h_to_y(h)

        line_xy = np.vstack([np.arange(self._n_x), y]).T * 2**6

        cv2.polylines(img, [line_xy.astype(np.int32)], isClosed=False, color=(
            255, 255, 255), thickness=2, lineType=cv2.LINE_AA, shift=6)
        if self._mouse_pos is not None:
            cv2.circle(img, self._mouse_pos, 15, (255, 255, 255), -1)
        return img

    def _render_history(self):
        n_pad = self._n_hist_disp - len(self._h_history)
        if n_pad > 0:
            hist = self._h_history[::-1]+[self._blank_row]*n_pad
        else:
            hist = self._h_history[-self._n_hist_disp:][::-1]
        h = np.array(hist)
        h = h/np.max(h) if np.max(h) > 0 else h
        h = h ** (1/2.)  # gamma-adjust brightness up
        img = (h * 255).astype(np.uint8) if np.max(h) > 0 else h.astype(np.uint8)
        return np.concatenate([img[:, :, np.newaxis]]*3, axis=2)

    def _get_drops(self, dt):
        """
        Returns the drops that have fallen since the last frame and any raindrops.
        The number of raindrops is a poisson process with rate self._rain_rate.
        """
        drops = self._new_drops
        self._new_drops = []
        if self._rain_rate > 0:
            n_drops = np.random.poisson(self._rain_rate * dt)
            raindrops = get_natural_raindrops(n_drops, self._max_x, self._a_max)
            drops.extend(raindrops)
            print("Raining %i drops." % n_drops)
        return drops

    def _add_h_hist(self, h):
        self._h_history.append(h)
        if len(self._h_history) > self._n_hist_disp:
            self._h_history.pop(0)

    def simulate_interactive(self):
        """
        Simulate the pond's evolution over time (until no drops are left, or max_iter is reached).
        :param raindrops: A list of raindrops, each a dict with 'times','x','mass' keys.
        """
        cv2.namedWindow(self._win_name)
        cv2.setMouseCallback(self._win_name, self._mouse)
        print("Simulating live raindrops in x=[0, %.1f]." % (self._max_x,))

        while True:
            dropping_now = self._get_drops(self._dt)
            self._waves, h, _ = self._step_sim(self._waves, dropping_now, self._dt)


            img = self.render(h)
            cv2.imshow('Pond', img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                self._waves = []
                self._h_history = []
                print("Resetting pond.")
            elif k == ord('r'):
                self._rain_rate = self._rain_rate * 2 if self._rain_rate > 0 else .01
                print("Rain rate set to %.2f drops per second." % self._rain_rate)
            elif k == ord('f'):
                self._rain_rate = self._rain_rate / 2 if self._rain_rate > .01 else 0.
                print("Rain rate set to %.2f drops per second." % self._rain_rate)


if __name__ == "__main__":
    pond = InteractivePond()
    pond.simulate_interactive()
