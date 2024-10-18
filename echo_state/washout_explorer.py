from esn import EchoStateNetwork
import matplotlib.pyplot as plt
import numpy as np


class EquilibriumExplorer(object):
    """
    Explore the effects of changing ESN parameters on the time to equilibrium (washout)
    """

    def __init__(self, esn_params):
        self.esn = EchoStateNetwork(**esn_params)
        self.params = esn_params

    def _get_states(self, X):
        state = np.zeros(self.params['n_reservoir'])
        states = []
        for x in X:
            state = self.esn._update_reservoir(x, state)
            states.append(state)
        return np.vstack(states)

    def find_washout_time(self, init_state=None, tol=1e-6, n_hidden=3, n_radom=2, min_iter=10):
        """
        Run the predict() method with zero inputs until the ESN reaches equilibrium (internal states don't change).
        :param init_state: initial state of the ESN
        :param tol: tolerance for convergence
        :param n_hidden: number of hidden units' activations to return
        :param n_radom: number of random projections of the hidden states to return.
        :returns: dict(N: number of washout iterations to use,
                       states: Nxn_hidden array of internal activations,
                       proj: N x n_random array of random projections of the full state vectors
                        )
        """
        state = np.zeros(self.params['n_reservoir']) if init_state is None else init_state
        states = []
        iter=0
        print_interval = 4
        proj_vec = np.random.randn(self.params['n_reservoir'], n_radom)
        projs = []
        zero_input = np.zeros(self.params['n_input'])
        zero_output = np.zeros(self.params['n_output'])
        while True:

            new_state,new_output = self.esn._update_reservoir(zero_input, state, zero_output)
            states.append(new_state[:n_hidden])
            err = np.abs(new_state - state).max()
            if err < tol and iter> min_iter:
                break
            if iter == print_interval:
                print("Iteration %d, max error %.9f" % (iter, err))
                print_interval *= 2
            state = new_state
            projs.append(np.dot(new_state, proj_vec))
            iter +=1
        print("Stopped after %i iterations." % iter)
        return {'N': len(states),
                'states': np.array(states),
                'proj': np.array(projs)}

    def plot_washout(self, init_state = None, tol=1e-6, n_hidden=3, n_random=2):
        """
        Plot the internal states on top of each other in a single plot.
        """

        r = self.find_washout_time(init_state, tol, n_hidden, n_random)
        n_plots = n_hidden + n_random
        fig, ax = plt.subplots(n_plots, 1, sharex=True)
        for i in range(n_hidden):
            ax[i].plot(r['states'][:,i], label='Hidden unit %d' % i)
        for i in range(n_random):
            ax[n_hidden+i].plot(r['proj'][:,i], label='Random projection %d' % i)
        plt.suptitle('Washout time: %d iterations\n' % r['N'])
        plt.legend()



if __name__ == "__main__":
    e = EquilibriumExplorer({'n_input': 1, 'n_reservoir': 100, 'n_output': 1, 'input_scale':2.,
                             'feedback_scale': 0, 'leak_rate':0, 'spectral_radius': .9, 'n_wash': 100})
    e.plot_washout()
    plt.show()