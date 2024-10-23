import logging
from ripples import Pond, Wave, get_natural_raindrops, SIGNED_WAVES
import cv2
import numpy as np
import time

LAYOUT = {'line_height': 300,
          'hist_height': 450,
          'min_win_width': 500,
          'win_name': 'Pond (space to reset, q to quit, r/f to adjust rain-rate)', }

class InteractivePond(Pond):
    """
    A Pond that can be interacted with in realtime.
    """

    def __init__(self, n_x=500, x_max=100., a_max=30, dt=0.1, *args, **kwargs):

        super(InteractivePond, self).__init__(n_x, x_max, *args, **kwargs)

        self._new_drops = []  # dropped since last frame
        self._n_x = n_x
        self._a_max = a_max

        self._init_layout(a_max)

        # animation
        self._t_sim = 0.0
        self._dt = dt  # time step in seconds
        self._mouse_pos = None
        self._blank_row = np.zeros(self._n_x)
        self._blank_frame = np.zeros((self._win_size[1], self._win_size[0], 3), dtype=np.uint8)
        self._win_name = LAYOUT['win_name']
        self._y_scale = 3*a_max
        self._min_y_scale = 3*a_max
        self._n_underscaled_frames = 0

        # state:
        self._waves = []  # list of active Wave objects
        self._interactions = []  # list of input vectors, to drive the ESN
        self._heights = []  # outputs, to be predicted by the ESN / plotted
        self._h_history = []  # history of height vectors
        self._n_drops = 0
        self._rain_rate = 0

        # for performance monitoring
        self._n_frames_total = 0
        self._t_0 = time.perf_counter()

        # for FPS display
        self._t_start = self._t_0
        self._n_frames = 0
        self._fps = 0

    def _init_layout(self, a_max):
        if self._n_x > LAYOUT['min_win_width']:
            width = self._n_x
            self._px_size = 1  # size of each pixel
        else:
            width = LAYOUT['min_win_width']
            self._px_size = int(width/self._n_x)  # size of each pixel
            width = self._n_x * self._px_size

        self._n_hist_disp = int(LAYOUT['hist_height']/self._px_size)
        hist_height = self._n_hist_disp * self._px_size
        self._midline = LAYOUT['line_height']
        self._win_size = width, hist_height + self._midline

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
            self._mouse_pos = (x, y)  # remember for rendering cursor
            if event == cv2.EVENT_LBUTTONDOWN:
                sign = 1  # if np.random.rand(1)>0.5 else -1
                self._new_drops.append({'x': drop_x, 'mass': drop_a*sign})
                logging.info("New wave: %i at x=%i, a=%i" % (len(self._waves), drop_x, drop_a))

        else:
            self._mouse_pos = None

    def _scale_up(self, img):
        """
        Scale up history image by self._px_size.
        """
        width, height = img.shape[1] * self._px_size, img.shape[0] * self._px_size
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    def render(self, cur_heights):
        img = self._blank_frame.copy()
        img[:self._midline, :, :] = self._render_heights(cur_heights)
        img[self._midline:, :, :] = self._scale_up(self._render_history())
        self._render_text(img)
        return img

    def _render_text(self, img):
        pos = [10, 20]

        def _put_string(s):
            cv2.putText(img, s, (pos[0], pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            pos[1] += 20

        strings = ["Time: %.1f s  (FPS:  %.2f)" % (self._n_frames_total * self._dt, self._fps),
                   "Rain rate: %.2f drops/s" % self._rain_rate,
                   "Drops fallen: %i" % self._n_drops,
                   "Current waves: %i" % len(self._waves),
                   "Vertical scale: %.1f" % self._y_scale]
        for s in strings:
            _put_string(s)

    def _render_heights(self, state_vec):
        """
        Scale vertically to fit the window and plot as a line (i.e. a vertical slice of the pond).
        """
        img = np.zeros((self._midline, self._n_x*self._px_size, 3), dtype=np.uint8)
        h = state_vec
        highest= np.max(np.abs(h))
        


        if highest < self._y_scale/2:
            self._n_underscaled_frames += 1
            if self._n_underscaled_frames > 30:
                if self._y_scale > self._min_y_scale:
                    self._y_scale = self._y_scale/2
                else:
                    self._y_scale = self._min_y_scale
                self._n_underscaled_frames = 0
        elif highest > self._y_scale:
            while highest > self._y_scale:
                self._y_scale *= 2

        def _h_to_y(h):
            y=  (1 - (h/self._y_scale)) * self._midline
            if not SIGNED_WAVES:
                return y
            else:
                return y/2 


            
        # if self._t_sim >5.0:
        #    import ipdb; ipdb.set_trace()
        y = _h_to_y(h)

        line_xy = np.vstack([np.arange(self._n_x), y]).T

        line_xy[:, 0] = line_xy[:, 0] * self._px_size + self._px_size/2

        cv2.polylines(img, [(line_xy*2**6).astype(np.int32)], isClosed=False, color=(
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
        # h = h ** (1/2.)  # gamma-adjust brightness up
        img = (h * 255).astype(np.uint8) if np.max(h) > 0 else h.astype(np.uint8)
        return np.concatenate([img[:, :, np.newaxis]]*3, axis=2)

    def _get_drops(self, dt, time_index=0):
        """
        Returns the drops that have fallen since the last frame and any raindrops.
        The number of raindrops is a poisson process with rate self._rain_rate.
        """
        drops = self._new_drops
        self._new_drops = []
        if self._rain_rate > 0:
            n_drops = np.random.poisson(self._rain_rate * dt)
            # print(self._rain_rate, n_drops, dt)
            raindrops = get_natural_raindrops(n_drops, 1.0, self._max_x, self._a_max)
            drops.extend(raindrops)

        for drop in drops:
            drop['t'] = time_index

        self._n_drops += len(drops)
        return drops

    def _add_h_hist(self, h):
        self._h_history.append(h)
        if len(self._h_history) > self._n_hist_disp:
            self._h_history.pop(0)

    def _disp_and_keyboard(self, img):
        cv2.imshow(self._win_name, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return False
        elif k == ord(' '):
            self._waves = []
            self._h_history = []
            logging.info("Resetting pond.")
        elif k == ord('r'):
            self._rain_rate = self._rain_rate * 2 if self._rain_rate > 0 else .01
            logging.info("Rain rate set to %.2f drops per second." % self._rain_rate)
        elif k == ord('f'):
            self._rain_rate = self._rain_rate / 2 if self._rain_rate > .01 else 0.
            logging.info("Rain rate set to %.2f drops per second." % self._rain_rate)

        self._n_frames += 1
        self._n_frames_total += 1
        if self._n_frames % 100 == 0:
            now = time.perf_counter()
            self._fps = self._n_frames / (now - self._t_start)
            logging.info("FPS:  %.3f,  current wave count: %i" % (self._fps, len(self._waves)))
            self._n_frames = 0
            self._t_start = now
        return True

    def simulate_interactive(self, realtime=True):
        """
        Simulate the pond's evolution over time (until no drops are left, or max_iter is reached).
        :param raindrops: A list of raindrops, each a dict with 'times','x','mass' keys.
        :param realtime: If True, the simulation will run in real time.
        :returns: A list of all raindrops that fell during the simulation.
        """
        cv2.namedWindow(self._win_name)
        cv2.resizeWindow(self._win_name, self._win_size[0], self._win_size[1])
        cv2.setMouseCallback(self._win_name, self._mouse)
        logging.info("Simulating live raindrops in x=[0, %.1f]." % (self._max_x,))

        delay = self._dt
        data = {'inputs': [], 'outputs': []}
        self._t_sim = 0
        drops = []
        while True:
            if realtime:
                time.sleep(delay)
            t_0 = time.perf_counter()
            self._t_sim += self._dt

            dropping_now = self._get_drops(self._dt, self._t_sim)
            self._waves, h, input = self._step_sim(self._waves, dropping_now, self._dt)

            img = self.render(h)
            if not self._disp_and_keyboard(img):
                break
            self._add_h_hist(h)

            render_time = time.perf_counter() - t_0
            delay = max(0, self._dt - render_time)

            data['inputs'].append(input)
            data['outputs'].append(h)
            drops.extend(dropping_now)

        cv2.destroyAllWindows()
        return data, drops


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dt = .10
    # pond = InteractivePond(speed_factor=1.)
    pond = InteractivePond(n_x=1000, x_max=100,decay_factor=1., dt=dt, speed_factor=.3, wave_scale=0.5)

    pond.simulate_interactive(realtime=False)
