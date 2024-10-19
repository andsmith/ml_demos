import numpy as np
import matplotlib.pyplot as plt
from esn import EchoStateNetwork
from signals import  make_square, make_sawtooth, make_triangle, make_sine, make_complex


class EchoStateTester(object):
    """d
    Create an ESN with the specified parameters, create an input signal (a square wave), and 
    five output signals (square, sawtooth, triangle, sine, and complex waves), and train the ESN
    to reproduce them.
    """

    def __init__(self, esn_params, n_train_samples=3000, n_test_samples=500, period=20):
        """
        Initialize & run all tests.
        """
        self._n_train = n_train_samples

        self.data_in = make_square(period, n_train_samples + n_test_samples)

        self.data_out = {'square': make_square(period, n_train_samples + n_test_samples),
                         'sawtooth': make_sawtooth(period, n_train_samples + n_test_samples),
                         'triangle': make_triangle(period, n_train_samples + n_test_samples),
                         'sine': make_sine(period, n_train_samples + n_test_samples),
                         'complex': make_complex(period, n_train_samples + n_test_samples, 50)}

        self.esns = {'square': EchoStateNetwork(**esn_params),
                     'sawtooth': EchoStateNetwork(**esn_params),
                     'triangle': EchoStateNetwork(**esn_params),
                     'sine': EchoStateNetwork(**esn_params),
                     'complex': EchoStateNetwork(**esn_params)}
        
        for signal in self.data_out.keys():
            train_in = self.data_in[:self._n_train].reshape(-1,1)
            train_out = self.data_out[signal][:self._n_train].reshape(-1,1)
            self.esns[signal].train(train_in, train_out)
        test_data = self.data_in[self._n_train:].reshape(-1,1)
        self.predicted = {signal: self.esns[signal].predict(test_data) for signal in self.data_out.keys()}

        self.errors = {signal: self._get_error(self.predicted[signal], 
                                               self.data_out[signal][self._n_train:]) for signal in self.data_out.keys()}


    def plot_transduction(self, ):
        """
        Plot the test signals, training output in blue, and predicted output in orange.
        """
        plot_prune = 0

        for i,signal in enumerate(self.data_out.keys()):
            plt.subplot(len(self.data_out.keys()), 1, i+1) 
            plt.plot(self.data_out[signal][(self._n_train+plot_prune):].reshape(-1), label='True ' + signal)
            pred = self.predicted[signal][plot_prune:].reshape(-1)
            plt.plot(pred, label='Predicted ' + signal)
            plt.title("Transduction of %s, rms %.4f" % (signal, self.errors[signal]))
        plt.legend()

        plt.show()

    def _get_error(self, pred, true):
        return np.sqrt(np.mean((pred - true) ** 2))


if __name__ == "__main__":
    t = EchoStateTester({'n_input': 1, 'n_reservoir': 2, 'n_output': 1, 'input_scale':1, 'feedback_scale': .5,
                         'leak_rate': 0.0, 'spectral_radius': 0.95, 'n_wash': 0})
    t.plot_transduction()   