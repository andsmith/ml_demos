"""
What sub-types of individual digits are there in the MNIST dataset?
Can the digits with sub-types be automatically distinguished from digits all drawn the same way?

App is a matplotlib window with 3 subplots in 2 columns:

    +----------+-------+-------+
    |          |   0   |   1   |
    +  eigen-  +-------+-------+
    |          |   2   |   3   |
    + vectors  +-------+-------+
    |          |   4   |   5   |
    +          +-------+-------+
    |----------|   6   |   7   |
    + controls +-------+-------+
    |          |   8   |   9   |
    +----------+-------+-------+

(the 10 digit areas are all in the same matplotlib axis)

The three axes are:

 - Eigenvector subplot:  Shows the lowest N eigenvalues of all 10 similarity graphs & legend.
 - Controls subplot:  Has these tools:
      * Select clustering algorithm (Spectral/kmeans)
      * Select similarity graph type, from list:
          n-neighbors,
          n-neighbors-mutual,
          n-soft-neighbors-additive,
          n-soft-neighbors-multiplicative,
          epsilon,
          full
      * Slider for N, the number of clusters (aka "digit sub-types")
      * Slider for parameter, appropriate for similarity graph type. (activate/deactivate as sim graph type is changed)
          n-neighbors (and -mutual): k
          soft-n-neighbors(-additive/-multiplicative): alpha
          epsilon: epsilon
          full: sigma
      * Button to run clustering
 - Digit areas:  In each digit's cluster area:
     - for the K clusters, select P "prototype" digits from the training data.
     - Arrange them in that digit's area so as to be as large as possible.

"""
import time
from mnist_data import MNISTData
from clustering import SpectralAlgorithm, KMeansAlgorithm
from mnist_common import MNISTResult, GRAPH_TYPES, GRAPH_PARAM_NAMES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from threading import Thread


ALGORITHMS = {'k-means': KMeansAlgorithm,
              'spectral': SpectralAlgorithm}

# relative widget axies have origin at bottom left

x_left = 0.4  # between controls & left digit column
x_right = 0.7  # between digit columns
y_ctrl_div = 0.4  # y value between controls & eigenvalues (from bottom)
ctrl_w_h = x_left, 1.0 - y_ctrl_div
eigen_w_h = x_left, y_ctrl_div
digit_w_h = 1.0 - x_left, 1.0
_TITLE_SPACE = 0.05
w_margin = 0.01
LAYOUT = {'control_ax_loc': [w_margin, w_margin, x_left-w_margin*2, y_ctrl_div-w_margin*2],
          'eigen_ax_loc': [w_margin+.06, y_ctrl_div+w_margin, x_left-w_margin*2, 1-y_ctrl_div-_TITLE_SPACE-w_margin*2],
          'digit_ax_loc': [x_left-w_margin, w_margin, 1.0 - x_left - w_margin*2, 1-w_margin*2]}

# keys should match values in GRAPH_PARAM_NAMES
PARAM_RANGES = {'sigma': (0, 3000.),
                'alpha': (0, 250.),
                'k': (1, 100),
                'epsilon': (0, 3000.)}
MARGIN = 0.003


class ControlsPlot(object):
    """
      Layout:

          Algoritm:     Similarity Graph:
            k-means       n-neighbors
            spectral      n-neighbors-mutual
                          n-soft-neighbors-add
                          n-soft-neighbors-mult
          [Run]           epsilon
                          full

          Num Clust:  [slider------------------] 

          Param: [slider-----------------------]
    """
    # in unit square, will be scaled to control_ax_loc bbox
    _LIST_DIV_X = .3
    _SLIDER_INDENT = .35
    _SLIDER_RIGHT = .1
    CTRL_LAYOUT = {'alg_radio': [MARGIN, .75, _LIST_DIV_X-MARGIN*2, .25-MARGIN*2],
                   'sim_radio': [_LIST_DIV_X+MARGIN, .3+MARGIN, 1-_LIST_DIV_X-MARGIN*2, .7-MARGIN*2],
                   'N_slider': [MARGIN+_SLIDER_INDENT, .125 + MARGIN, 1-2*MARGIN-_SLIDER_INDENT-_SLIDER_RIGHT, .125-MARGIN*2],
                   'param_slider': [MARGIN+_SLIDER_INDENT, MARGIN, 1-2*MARGIN-_SLIDER_INDENT-_SLIDER_RIGHT, .125-MARGIN*2],
                   'run_button': [MARGIN, .45, .2, .1]}
    _SLIDER_FONT_SIZE = 8
    _RADIO_FONT_SIZE = 8

    _SIM_GRAPH_OPTIONS = {'n-neighbors': 'n-neighbors',
                          'n-neighbors_mutual': 'n-neighbors-mutual',
                          'soft_neighbors_additive': 'soft-neighbors-add',
                          'soft_neighbors_multiplicative': 'soft-neighbors-mult',
                          'epsilon': 'epsilon',
                          'full': 'full'}

    def _graph_name_from_option(self, option):
        match = [g_name for g_name in self._SIM_GRAPH_OPTIONS if self._SIM_GRAPH_OPTIONS[g_name] == option]
        if len(match) != 1:
            raise ValueError("Invalid option %s" % option)
        return match[0]

    def _get_sim_type(self):
        option = self._sim_radio.value_selected
        return self._graph_name_from_option(option)

    def __init__(self, fig):
        self._fig = fig
        self._init_controls()

    def _init_controls(self):

        ctrl_w, ctrl_h = LAYOUT['control_ax_loc'][2], LAYOUT['control_ax_loc'][3]
        ctrl_x0, ctrl_y0 = LAYOUT['control_ax_loc'][0], LAYOUT['control_ax_loc'][1]

        def _shift(bbox, i):
            return [bbox[0], bbox[1]+i/20, bbox[2], bbox[3]]

        def _rescale(bbox):
            # scale a bbox in the unit square to one in the control_ax_loc bbox
            x0, y0, w, h = bbox
            return [ctrl_x0 + x0*ctrl_w, ctrl_y0 + y0*ctrl_h, w*ctrl_w, h*ctrl_h]

        print(LAYOUT['control_ax_loc'])
        alg_names = [alg_name for alg_name in ALGORITHMS]
        sim_graph_names = [sim_name for sim_name in GRAPH_TYPES]

        self._algo_radio = RadioButtons(plt.axes(_rescale(self.CTRL_LAYOUT['alg_radio'])), alg_names)
        graph_options = [self._SIM_GRAPH_OPTIONS[sim_name] for sim_name in sim_graph_names]
        self._sim_radio = RadioButtons(plt.axes(_rescale(self.CTRL_LAYOUT['sim_radio'])), graph_options)
        self._N_slider = Slider(plt.axes(_rescale(self.CTRL_LAYOUT['N_slider'])), 'N clusters', 1, 20, valinit=2)
        # make four sliders, one for each parameter type
        self._slider_axes = {param_name: plt.axes(
            _rescale(self.CTRL_LAYOUT['param_slider'])) for param_name in PARAM_RANGES}        
        self._param_sliders = {'k': Slider(self._slider_axes['k'], 'k ',  # separate to control valstep
                                           valmax=PARAM_RANGES['k'][1], valmin=PARAM_RANGES['k'][0],
                                           valinit=PARAM_RANGES['k'][0], valstep=1),
                               'sigma': Slider(self._slider_axes['sigma'], 'sigma ',
                                               valmax=PARAM_RANGES['sigma'][1], valmin=PARAM_RANGES['sigma'][0],
                                               valinit=PARAM_RANGES['sigma'][0]),
                               'alpha': Slider(self._slider_axes['alpha'], 'alpha ',
                                               valmax=PARAM_RANGES['alpha'][1], valmin=PARAM_RANGES['alpha'][0],
                                               valinit=PARAM_RANGES['alpha'][0]),
                               'epsilon': Slider(self._slider_axes['epsilon'], 'epsilon ',
                                                 valmax=PARAM_RANGES['epsilon'][1], valmin=PARAM_RANGES['epsilon'][0],
                                                 valinit=PARAM_RANGES['epsilon'][0])}

        self._run_button = Button(plt.axes(_rescale(self.CTRL_LAYOUT['run_button'])), 'Run',)
        [l.set_fontsize(self._RADIO_FONT_SIZE) for l in self._algo_radio.labels]
        [l.set_fontsize(self._RADIO_FONT_SIZE) for l in self._sim_radio.labels]
        self._N_slider.label.set_fontsize(self._SLIDER_FONT_SIZE)

        self._run_button.on_clicked(self._run)
        self._algo_radio.on_clicked(self._algo_changed)
        self._sim_radio.on_clicked(self._sim_changed)
        self._N_slider.on_changed(self._N_changed)

        for param_name in self._param_sliders:

            def _change_par(new_val):
                print("Param %s changed to %f" % (param_name, new_val))
                self._param_changed(param_name, new_val)

            self._param_sliders[param_name].on_changed(_change_par)
            self._param_sliders[param_name].label.set_fontsize(self._SLIDER_FONT_SIZE)
            self._param_sliders[param_name].valtext.set_fontsize(self._SLIDER_FONT_SIZE)
            self._param_sliders[param_name].set_active(False)
            self._slider_axes[param_name].set_visible(False)
            
        if True:        
            for spine in self._algo_radio.ax.spines.values():
                spine.set_visible(False)
            for spine in self._sim_radio.ax.spines.values():
                spine.set_visible(False)

        self._set_slider_type()

    def _set_slider_type(self):
        # Turn on the right slider and the others off.
        sim_type = self._get_sim_type()
        for param_name in self._param_sliders:
            self._param_sliders[param_name].set_active(False)
            self._slider_axes[param_name].set_visible(False)
        param_name = GRAPH_PARAM_NAMES[sim_type]
        self._param_sliders[param_name].set_active(True)
        self._slider_axes[param_name].set_visible(True)
        plt.draw()

    def _algo_changed(self, label):
        print("Algorithm changed to %s" % label)

    def _sim_changed(self, label):
        print("Similarity graph changed to %s" % label)
        self._set_slider_type()

    def _N_changed(self, val):
        print("N changed to %i" % val)

    def _param_changed(self, param_name, val):
        print("Param %s changed to %f" % (param_name, val))

    def _run(self, event):
        print("Running")

class EigenvaluesPlot(object):
    # Plot all eigenvalues, zoom in to top 15 by default
    def __init__(self, app, ax):
        self._init_v_plot = 15
        self._app = app
        self._ax = ax
        self._values = {i: np.zeros(100) for i in range(10)}
        self._init_plot()

    def _init_plot(self):
        # turn off ticks
        self._plots = [self._ax.plot(self._values[d],'-o', label='%i' % d)[0] for d in range(10)]
        self._ax.get_xaxis().set_visible(False)
        # self._ax.get_yaxis().set_visible(False)
        self._ax.get_xaxis().set_ticks([])
        # self._ax.get_yaxis().set_ticks([])
        self._ax.set_ylabel('')
        self._ax.set_title('lowest eigenvalues')
        self._ax.grid(True)
        self._ax.legend()
        self._fix_lims()

    def update_values(self, new_values):
        """
        :param new_values: dict with digits as keys, arrays of eigenvalues as values
        """
        print("Updating")
        self._values = new_values
        x_vals = np.arange(len(self._values[0]))
        for d in range(10):
            self._plots[d].set_ydata(self._values[d])
            self._plots[d].set_xdata(x_vals)
        self._fix_lims()
        plt.draw()

    def _fix_lims(self):
        self._ax.set_xlim(0, self._init_v_plot)
        ymax = np.max([np.max(v[:self._init_v_plot]) for v in self._values.values()])
        self._ax.set_ylim(-0.02, max(1, ymax))


def test_eigenvalues_plot():
    fig, ax = plt.subplots()
    e = EigenvaluesPlot(None, ax)
    e.update_values({i: np.arange(15)**1/2. for i in range(100)})

    def update_proc():
        print("Starting updates.")
        time.sleep(.3)
        e.update_values({i: np.arange(15)**2. for i in range(100)})
        time.sleep(.3)
        e.update_values({i: np.arange(15)**3. for i in range(100)})
        time.sleep(.3)
        e.update_values({i: np.arange(15)**1/2. for i in range(100)})
        print("Updates done.")

    t = Thread(target=update_proc)
    t.start()

    plt.show()


def test_controls_plot():
    # just print out what interaction happened
    fig = plt.figure()
    eig_ax = fig.add_axes(LAYOUT['eigen_ax_loc'])
    e = EigenvaluesPlot(None, eig_ax)
    e.update_values({i: np.arange(150)**1/2. for i in range(10)})

    c = ControlsPlot(fig)
    plt.show()


class DigitClusteringApp(object):
    """
    Each experiment will compute the similarity graph,
    plot the spectrum, and let the user choose a K to cluster it.
    Results are displayed as plots of the clusters in 2d embeddings.
    """

    def __init__(self, pca_dim=30, n_train=5000):
        pass


if __name__ == '__main__':
    # test_eigenvalues_plot()
    test_controls_plot()
