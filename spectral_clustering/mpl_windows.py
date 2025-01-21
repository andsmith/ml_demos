import logging
from layout import Windows, WINDOW_LAYOUT
import matplotlib.pyplot as plt
import numpy as np
from mpl_plots import plot_clustering
from abc import ABCMeta, abstractmethod
from util import orthornormalize
from matplotlib.widgets import Slider, Button

class MPLWindow(metaclass=ABCMeta):
    def __init__(self, app, kind):
        self.app = app
        self._kind = kind
        self._fig, self._ax = self._init_plot_window()        
        self._axes_flat = self._ax.flatten() if isinstance(self._ax, np.ndarray) else [self._ax]

        plt.ion()
        plt.show()

    def _init_plot_window(self):
        fig, ax = plt.subplots(nrows=WINDOW_LAYOUT['windows'][self._kind]['nrows'],
                                         ncols=WINDOW_LAYOUT['windows'][self._kind]['ncols'],
                                         figsize=WINDOW_LAYOUT['windows'][self._kind]['figsize'])
        fig.subplots_adjust(bottom=WINDOW_LAYOUT['windows'][self._kind]['widgit_space'])
        
        return fig, ax

    @abstractmethod
    def refresh(self):
        # Update whatever is needed to refresh the matplotlib axes.
        pass

    def clear(self):
        # Clear the data and data in the window.
        for ax in self._axes_flat:
            ax.clear()


class RandProjWindow(MPLWindow):
    def __init__(self, app):
        super().__init__(app, kind=Windows.rand_proj)
        self._noise = 0.02
        self._noise_offsets = None  # 2d, scale by _noise and add to data after projecting
        self._axes = None  # 2xf, orthogonal vectors in feature space
        self._features = None
        self._d = 3

        self._init_widgets()

    def _update_noise(self, noise):
        self._noise = noise
        self.refresh()

    def _init_widgets(self):
        """
        Add a slider to adjust the noise level and a button to remake the axes.
        """
        axcolor = 'lightgoldenrodyellow'
        ax_noise = plt.axes([0.2, 0.1, 0.3, 0.03], facecolor=axcolor)
        self._noise_slider = Slider(ax_noise, 'Noise', 0.0, .1, valinit=self._noise)
        self._noise_slider.on_changed(self._update_noise)

        ax_remake = plt.axes([0.7, 0.1, 0.2, 0.04])
        self._remake_button = Button(ax_remake, 'Randomize', color=axcolor, hovercolor='0.975')
        self._remake_button.on_clicked(self._remake_axes)

    def _init_plot_window(self):
                
        fig = plt.figure(figsize=WINDOW_LAYOUT['windows'][Windows.rand_proj]['figsize'])
        ax = fig.add_subplot(projection='3d')
        fig.subplots_adjust(bottom=WINDOW_LAYOUT['windows'][Windows.rand_proj]['widget_space'])

        return fig, ax

    def clear(self):
        self._features = None
        self._axes = None
        self._noise_offsets = None
        #logging.info("Cleared RandProjWindow.")
        return super().clear()
    
    def set_features(self, features):
        #logging.info("Setting features in RandProjWindow.")
        self._features = features
        self._noise_offsets = np.random.randn(self._features.shape[0]*self._d).reshape(-1, self._d)
        
        self._remake_axes()
        self.refresh()

    def _remake_axes(self, event=None):
        """
        :param event: Button click event (not used, if called from button callback)
        """
        if self._features is None:
            return
        
        # make random axes, orthogonal in feature space
        self._axes = np.random.randn(self._d, self._features.shape[1])
        self._axes = orthornormalize(self._axes)
        # check orthogonality
        err = np.abs(np.sum(self._axes[0] * self._axes[1]))
        if err > 1e-6:
            raise ValueError("Axes are not orthogonal!?")
        self.refresh()

    def refresh(self):
        #logging.info("Refreshing RandProjWindow.")
        if self._features is None:
            return
        points = self._features @ self._axes.T
        noisy_points = points + self._noise_offsets * self._noise * .1
        colors = self.app.windows[Windows.ui].get_cluster_color_ids()
        # import ipdb; ipdb.set_trace()
        self._ax.clear()
        if self._d==2:
            plot_clustering(self._ax, noisy_points, colors['colors']/255., colors['ids'], image_size=(500,500), alpha=0.5)
            # since clusters will be on the border if there are many connected components, move everything in by a percentage
            marg_frac = 0.025
            x_lim, y_lim = self._ax.get_xlim(), self._ax.get_ylim()
            w, h = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
            self._ax.set_xlim(x_lim[0] - marg_frac*w, x_lim[1] + marg_frac*w)
            self._ax.set_ylim(y_lim[0] - marg_frac*h, y_lim[1] + marg_frac*h)
        elif self._d ==3:
            color_v = [colors['colors'][c_id]/255. for c_id in colors['ids']]
            self._ax.scatter(noisy_points[:, 0], noisy_points[:, 1], noisy_points[:, 2], c=color_v, alpha=0.5)
        
        self._ax.set_title("Random Projection")
        self._fig.canvas.draw_idle()

class FakeApp:
    class FakeUIWindow:
        def get_cluster_color_ids(self):
            return {'colors': np.random.randint(0, 255, (10, 3)), 'ids': np.random.randint(0, 10, 100)}
    def __init__(self):
        self.windows = {Windows.ui: self.FakeUIWindow()}

# list of matplotlib window objects (MPLWindow), each gets rendered in its own window
MPL_WINDOW_NAMES = {Windows.rand_proj: "Random Projection"}
MPL_WINDOW_TYPES = {Windows.rand_proj: RandProjWindow}


def test_randproj_window():
    app = FakeApp()
    window = RandProjWindow(app)
    window.set_features(np.random.randn(100, 10))
    window._bbox_size = (800, 600)
    window.refresh()
    plt.show()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    test_randproj_window()

