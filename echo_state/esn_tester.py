import numpy as np
import matplotlib.pyplot as plt
from esn import EchoStateNetwork
from signals import make_square, make_sawtooth, make_triangle, make_sine, make_complex
import sys


class EchoStateTester(object):
    """d
    Create an ESN with the specified parameters, create an input signal (a square wave), and 
    five output signals (square, sawtooth, triangle, sine, and complex waves), and train the ESN
    to reproduce them.
    """

    def __init__(self, esn_params, n_train, n_test, period=16, train_washout=00):
        """
        Initialize & run all tests.
        """
        print("Testing ESN with params: ", esn_params)
        print("Training on %i samples, testing on %i samples" % (n_train, n_test))
        self._params = esn_params
        self._n_train = n_train
        self._train_washout = train_washout
        self._n_test = n_test
        self._period = period
        truth_out = .8

        self._train_phase_shifts = (0,)
        self._test_phase_shifts = (period//4, period//2)

        def _mk_train_test(signal_func):
            return {'train': [np.roll(signal_func(period, n_train), - phase_shift).reshape(-1, 1)*truth_out
                              for phase_shift in self._train_phase_shifts],

                    'tests': [np.roll(signal_func(period, n_test), - phase_shift).reshape(-1, 1)*truth_out
                              for phase_shift in self._test_phase_shifts]}

        self.data_in = _mk_train_test(make_square)

        self.data_out = {  # 'sawtooth': _mk_train_test(make_sawtooth),
            'triangle': _mk_train_test(make_triangle),
            'sine': _mk_train_test(make_sine),
            'complex': _mk_train_test(make_complex)
        }

        self.esns = {  # 'sawtooth': EchoStateNetwork(**esn_params),
            'triangle': EchoStateNetwork(**esn_params),
            'sine': EchoStateNetwork(**esn_params),
            'complex': EchoStateNetwork(**esn_params)
        }
        # single_esn = EchoStateNetwork(**esn_params)
        # self.esns = {k:single_esn for k in self.esns}

        # print trainind data size & shapes
        print("Training data sizes:")
        for signal in self.data_out:
            for i in range(len(self._train_phase_shifts)):
                print("signal: %s, phase shift: %i, shape: %s" %
                      (signal, self._train_phase_shifts[i], self.data_out[signal]['train'][i].shape))
        print("Testing data sizes:")
        for signal in self.data_out:
            for i in range(len(self._test_phase_shifts)):
                print("signal: %s, phase shift: %i, shape: %s" %
                      (signal, self._test_phase_shifts[i], self.data_out[signal]['tests'][i].shape))

        # Do training and predicting:

        self.pred_train = {}
        self.pred_test = {}
        for signal in self.data_out:
            for i in range(len(self._train_phase_shifts)):
                self.esns[signal].train_sequence(self.data_in['train'][i],
                                                 self.data_out[signal]['train'][i],
                                                 washout=self._train_washout)
            self.esns[signal].finish_training()
            self.pred_train[signal] = [self.esns[signal].predict(train_in) for train_in in self.data_in['train']]
            self.pred_test[signal] = [self.esns[signal].predict(test_in) for test_in in self.data_in['tests']]

    def plot_transduction(self, plot_n=100):
        """
        Plot the test signals, training output in blue, and predicted output in orange.
        """
        plot_n = min(plot_n, self._n_test)
        n_sigs = len(self.data_out)
        n_trains = min(len(self._train_phase_shifts), 2)
        n_tests = len(self._test_phase_shifts)

        fig, ax = plt.subplots(n_sigs+1, n_trains+n_tests, sharex=True)
        ax = ax.reshape(n_sigs+1, n_trains+n_tests)

        # plot input on top rows
        for i in range(n_trains):
            ax[0][i].plot(self.data_in['train'][i][:plot_n])
            ax[0][i].set_title('TRAIN, phase shift: %i' % (self._train_phase_shifts[i],))
        for i in range(n_tests):
            ax[0][n_trains+i].plot(self.data_in['tests'][i][:plot_n])
            ax[0][n_trains+i].set_title('TEST, phase shift: %i' % (self._test_phase_shifts[i],))

        def _plot_sig(row, col, true, pred):
            row = row+1
            error = self._get_error(pred[self._train_washout:], true[self._train_washout:])
            ax[row][col].plot(true, label="True %s" % signal if col == 0 else None)
            ax[row][col].plot(pred, '--', label='RMS:  %.3f' % (error,))
            ax[row][col].legend(loc='upper right')
            x = self._train_washout - .5
            ax[row][col].axvline(x, color='k', linestyle='--')
            ax[row][col].axvline(x, color='k', linestyle='--')
            if i < len(self.data_out.keys())-1:
                ax[row][col].set_xticks([])

        for i, signal in enumerate(self.data_out.keys()):

            # training data
            for train_ind in range(n_trains):
                _plot_sig(i, train_ind, self.data_out[signal]['train'][train_ind]
                          [:plot_n], self.pred_train[signal][train_ind][:plot_n])
            # test data
            for test_ind in range(n_tests):
                _plot_sig(i, 1+test_ind, self.data_out[signal]['tests'][test_ind]
                          [:plot_n], self.pred_test[signal][test_ind][:plot_n])

        ax[0][0].set_ylabel("Input")

        plt.suptitle("ESN(%i units, %.2f leak, %.2f radius)\nsig-period: %i, n_train: %i, train_washout: %i" % (self._params['n_reservoir'],
                                                                                                                self._params['leak_rate'],
                                                                                                                self._params['spectral_radius'],
                                                                                                                self._period,
                                                                                                                self._n_train, self._train_washout))
        plt.axis('tight')

        plt.show()

    def _get_error(self, pred, true):
        return np.sqrt(np.mean((true.reshape(-1)-pred.reshape(-1)) ** 2))


def _plot_sample_waves():
    n = 75
    waves = [{'w': make_square(16, n), 'title': 'Square'},
             {'w': make_triangle(16, n), 'title': 'Triangle'},
             {'w': make_sine(16, n), 'title': 'Sine'},
             {'w': make_complex(16, n), 'title': 'Complex'}]

    fig, ax = plt.subplots(len(waves), 1)
    for i, wave in enumerate(waves):
        print(wave, i)
        ax[i].plot(wave['w'])
        ax[i].set_title(wave['title'], fontsize=10)
        ax[i].axis('off')
        ax[i].set_ylim(-1.5, 1.5)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # np.random.seed(3)

    if len(sys.argv) < 2:
        n_res = 10
    else:
        n_res = int(sys.argv[1])

    print("Running tests using %i reservoir units." % n_res)

    esn_params = {'n_input': 1,
                  'n_reservoir': n_res,
                  'n_output': 1,
                  'input_scale': 1., 'feedback_scale': 0.0,
                  'leak_rate': 0.0, 'spectral_radius': 0.98,
                  'linear_out': False}
    #_plot_sample_waves()
    t = EchoStateTester(esn_params, n_train=50000, n_test=1000, train_washout=15)
    t.plot_transduction()
