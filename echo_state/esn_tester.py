import logging

import numpy as np
import matplotlib.pyplot as plt
from esn import EchoStateNetwork
from signals import make_square, make_sawtooth, make_triangle, make_sine, make_complex


class EchoStateTester(object):
    """d
    Create an ESN with the specified parameters, create an input signal (a square wave), and 
    five output signals (square, sawtooth, triangle, sine, and complex waves), and train the ESN
    to reproduce them.
    """

    def __init__(self, esn_params, n_train, n_test, period=15, train_washout=00):
        """
        Initialize & run all tests.
        """
        print("Testing ESN with params: ", esn_params)
        print("Training on %i samples, testing on %i samples" % (n_train, n_test))
        self._params = esn_params
        self._n_train = n_train
        self._train_washout = train_washout
        self._n_test = n_test

        truth_out = 0.8

        def _mk_train_test(signal_func):
            # For now, identical.
            return {'train': signal_func(period, n_train).reshape(-1, 1)*truth_out,
                    'test': signal_func(period, n_test).reshape(-1, 1)*truth_out}

        self.data_in = _mk_train_test(make_square)
        print("data_in: ", self.data_in['train'].shape)    
        self.data_out = {'sawtooth': _mk_train_test(make_sawtooth),
                         'triangle': _mk_train_test(make_triangle),
                         'sine': _mk_train_test(make_sine),
                         'complex': _mk_train_test(make_complex) }

        self.esns = {'sawtooth': EchoStateNetwork(**esn_params),
                     'triangle': EchoStateNetwork(**esn_params),
                     'sine': EchoStateNetwork(**esn_params),
                     'complex': EchoStateNetwork(**esn_params) }

        self.pred_train = {}
        self.pred_test = {}
        for signal in self.data_out:
            self.esns[signal].train(self.data_in['train'],
                                    self.data_out[signal]['train'],
                                    washout=self._train_washout)

            self.pred_train[signal] = self.esns[signal].predict(self.data_in['train'])
            self.pred_test[signal] = self.esns[signal].predict(self.data_in['test'])

    def plot_transduction(self, plot_n=200):
        """
        Plot the test signals, training output in blue, and predicted output in orange.
        """
        plot_n = min(plot_n, self._n_test)
        n_plots = len(self.data_out)
        fig, ax = plt.subplots(n_plots+1, 2, sharex=True)
        ax = ax.reshape(n_plots+1, 2)

        # plot input on top rows
        ax[0][0].plot(self.data_in['test'][:plot_n])
        ax[0][1].plot(self.data_in['train'][:plot_n])
        ax[0][0].set_ylabel("Input signal")
        ax[0][0].set_xticks([])
        ax[0][1].set_xticks([])

        def _plot_sig(row, col, true, pred):
            row = row+1
            error = self._get_error(pred[self._train_washout], true[self._train_washout])
            ax[row][col].plot(true, label="True %s" % signal)
            ax[row][col].plot(pred, '--', label='RMS:  %.3f' % (error,))
            ax[row][col].legend(loc='upper right')
            x = self._train_washout - .5
            ax[row][col].axvline(x, color='k', linestyle='--')
            ax[row][col].axvline(x, color='k', linestyle='--')
            if i < len(self.data_out.keys())-1:
                ax[row][col].set_xticks([])

        for i, signal in enumerate(self.data_out.keys()):

            # training data
            _plot_sig(i, 0, self.data_out[signal]['train'][:plot_n], self.pred_train[signal][:plot_n])

            # test data
            _plot_sig(i, 1, self.data_out[signal]['test'][:plot_n], self.pred_test[signal][:plot_n])

        ax[0][1].set_title("testing data")
        ax[0][0].set_title('training data')

        plt.suptitle("ESN(%i units, %.2f leak, %.2f radius)\nn_train: %i, train_washout: %i" % (self._params['n_reservoir'],
                                                                                                self._params['leak_rate'],
                                                                                                self._params['spectral_radius'],
                                                                                                self._n_train, self._train_washout))
        plt.axis('tight')

        plt.show()

    def _get_error(self, pred, true):
        return np.sqrt(np.mean((true.reshape(-1)-pred.reshape(-1)) ** 2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #np.random.seed(0)
    esn_params = {'n_input': 1, 'n_reservoir': 20, 'n_output': 1, 'input_scale': 1, 'feedback_scale': 0.0,
                  'leak_rate': 0.0, 'spectral_radius': 0.98, 'n_wash': 100, 'linear_out': False}

    t = EchoStateTester(esn_params, n_train=20000, n_test=1000, train_washout=15)
    t.plot_transduction()
