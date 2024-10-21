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

    def __init__(self, esn_params, n_train_samples=1000, n_test_samples=300, period=15,train_washout=0):
        """
        Initialize & run all tests.
        """
        self._params = esn_params
        self._n_train = n_train_samples
        self._train_washout = train_washout

        self.data_in = make_square(period, n_train_samples + n_test_samples)

        self.data_out = {#'square': make_square(period, n_train_samples + n_test_samples),
                         'sawtooth': make_sawtooth(period, n_train_samples + n_test_samples),
                         #'triangle': make_triangle(period, n_train_samples + n_test_samples),
                         'sine': make_sine(period, n_train_samples + n_test_samples),
                         'complex': make_complex(period, n_train_samples + n_test_samples, 50)
                         }

        self.esns = {#'square': EchoStateNetwork(**esn_params),
                     'sawtooth': EchoStateNetwork(**esn_params),
                     #'triangle': EchoStateNetwork(**esn_params),
                     'sine': EchoStateNetwork(**esn_params),
                        'complex': EchoStateNetwork(**esn_params)
                     }
        
        for signal in self.data_out.keys():
            #if signal=='sine':
                #import ipdb; ipdb.set_trace()
            train_in = self.data_in[:self._n_train].reshape(-1,1)
            train_out = self.data_out[signal][:self._n_train].reshape(-1,1)
            self.esns[signal].train(train_in, train_out, self._train_washout)

        
            
        test_data = self.data_in[self._n_train:].reshape(-1,1)
        self.predicted = {signal: self.esns[signal].predict(test_data) for signal in self.data_out.keys()}


    def plot_transduction(self, ):
        """
        Plot the test signals, training output in blue, and predicted output in orange.
        """
        plot_prune = 30  # don't plot before ESN settles?


        for i,signal in enumerate(self.data_out.keys()):
            plt.subplot(len(self.data_out.keys()), 1, i+1) 
            true = self.data_out[signal][(self._n_train+plot_prune):].reshape(-1)
            plt.plot(true,label="True %s" % signal)
            pred = self.predicted[signal][plot_prune:].reshape(-1)       
            error = self._get_error(pred, true)

            plt.plot(pred, label='RMS:  %.3f' %( error,))
            #plt.plot(true-pred, label='Error, trial: ' + signal)
            print(signal, np.sqrt(np.mean((true-pred)**2))) 


            plt.title("%s rms = %.4f" % (signal, error))
            if i< len(self.data_out.keys())-1:
                plt.xticks([])
            plt.legend(loc='upper right')
        plt.suptitle("ESN(%i units, %.2f leak, %.2f radius)" % (self._params['n_reservoir'],
                                                                 self._params['leak_rate'], 
                                                                 self._params['spectral_radius']))
        plt.axis('tight')
        
        plt.show()

    def _get_error(self, pred, true):
        return np.sqrt(np.mean((true.reshape(-1)-pred.reshape(-1)) ** 2))


if __name__ == "__main__":

    t = EchoStateTester({'n_input': 1, 'n_reservoir': 10    , 'n_output': 1, 'input_scale':1,
                         'leak_rate': 0.0, 'spectral_radius': 0.95, 'n_wash': 100},n_test_samples=1000)
    t.plot_transduction()   