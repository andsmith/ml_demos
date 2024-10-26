import numpy as np
import logging
from scipy.optimize import minimize
import logging


class EchoStateNetwork(object):
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, leak_rate=0.0, 
                  input_scale=1.0, feedback_scale=0., linear_out=False, max_eq_iter=1000,
                  state_noise=0.0, sparsity=0.0):

        # shape:
        self.n_in = n_input
        self.n_res = n_reservoir
        self.n_out = n_output
        self.linear_out = linear_out

        # properties:
        self.sr = spectral_radius
        self.lr = leak_rate
        self.input_scale = input_scale
        self.feedback_scale = feedback_scale
        self.max_eq_iter = max_eq_iter
        self.state_noise = state_noise
        self.sparsity = sparsity

        # Accumulate here each time train_sequence is called, then call finish_training to update weights.
        self._train_info = {}
        self._reset_training()
        self._init_weights()

        # Should always have eqlib state before training/predicting, find it now that we have reservoir weights:
        self.eq_state, self.n_eq_washout = self._get_equilib_state(n_iter=0)

    def _reset_training(self):
        # self._train_info['states'] = []
        #self._train_info['inputs'] = []
        # self._train_info['targets'] = []
        #print("Resetting training info.")   

        # Batch version:
        self._train_info['ATA_terms'] = []
        self._train_info['ATY_terms'] = []

    def _init_weights(self):

        self.W_in_res = np.random.normal(0, 1, (self.n_res, self.n_in + 1)) * self.input_scale
        self.W_res_out = np.random.normal(0, 1, (self.n_out, self.n_res + self.n_in + 1))
        self.W_out_res = np.random.normal(0, 1, (self.n_res, self.n_out)) if self.feedback_scale > 0 else None
        self.W_res_res = np.random.normal(0, 1, (self.n_res, self.n_res))
        # enforce sparsity:
        mask = np.random.rand(self.n_res, self.n_res) > self.sparsity
        self.W_res_res *= mask
        # enforce spectral radius:
        s_rad = np.max(np.abs(np.linalg.eigvals(self.W_res_res)))
        if s_rad > 0:
            self.W_res_res *= 1.0 / s_rad * self.sr
        else:
            logging.warning("Failed to enforce spectral radius, is matrix very small and sparse?")

    def _update_reservoir(self, x, state, last_output=None):
        """
        Process a state vector and input, given the weights.

        :param x: input vector
        :param state: current state
        :return: new state
        """
        # State feedback & input:
        # if x.shape[0]==1:

        excitations = np.dot(self.W_res_res, state) + np.dot(self.W_in_res, np.append(x, [1]))
        if self.feedback_scale > 0:
            excitations += np.dot(self.W_out_res, last_output) * self.feedback_scale
        activations = np.tanh(excitations)
        new_state = (self.lr) * state + (1.0-self.lr) * activations
        new_output = np.dot(self.W_res_out, np.concatenate((new_state, x, [1])))
        if not self.linear_out:
            new_output = np.tanh(new_output)

        return new_state, new_output

    def _get_equilib_state(self, n_iter=0, tol=1e-6):
        """
        Run for N iterations, or find the number N where the state stops changing given zeroed inputs/feedback.
        """
        state = np.zeros(self.n_res)
        z_input = np.zeros(self.n_in)
        z_output = np.zeros(self.n_out)
        prev_state = state + 1.0

        for iter in range(self.max_eq_iter):
            if n_iter > 0 and iter == n_iter:
                return state, iter

            state, _ = self._update_reservoir(z_input, state, z_output)

            if n_iter == 0 and np.max(np.abs(state - prev_state)) < tol:
                #logging.info("Found equilib state in %i iterations." % iter)
                return state, iter

            prev_state = state
        #logging.warning("Failed to find equilib state in %i iterations." % self.max_eq_iter)
        return state, self.max_eq_iter
    
    def get_training_state_range(self):
        return self._train_info['state_min_max']

    def train_sequence(self, X, Y, washout=0, batch_size=50000):
        """
        Train the ESN on the sequence of input/output pairs.
        :param X: n_samples x n_input array of input samples
        :param Y: n_samples x n_output array of output samples
        :param washout: number of initial x,y samples to show the ESN but disregard in final training.
        :param batch_size: if > 0, collect pinv terms in batches (Should be identical results.)
        """
        logging.info("Training ESN with %i samples, washout=%i, batch_size=%i" % (X.shape[0], washout, batch_size))
        state = self.eq_state.copy()

        n = X.shape[0]

        states = []

        batch_start_ind = 0
        for i, (x, y) in enumerate(zip(X, Y)):

            teacher = Y[i-1] if i > 0 else np.zeros(self.n_out)
            state, _ = self._update_reservoir(x, state, teacher)

            if i < washout:
                batch_start_ind = i + 1
                continue

            states.append(state)

            if (batch_size > 0 and len(states) == batch_size) or (i == n-1):
                inputs = X[batch_start_ind:batch_start_ind+len(states)]
                targets = Y[batch_start_ind:batch_start_ind+len(states)]

                logging.info("\tBatch %i, %i states" % (len(self._train_info['ATA_terms']), len(states)))
                self._update_batch(states,
                                   np.array(inputs),
                                   np.array(targets))
                states = []
                batch_start_ind = i + 1

        if len(states) > 0:
            self._update_batch(states, np.array(inputs), np.array(targets))

    def finish_training(self):
        ATA = np.sum(self._train_info['ATA_terms'], axis=0) - np.eye(self.n_res + self.n_in + 1) * .1  # regularization
        ATY = np.sum(self._train_info['ATY_terms'], axis=0)
        self.W_res_out = np.dot(np.linalg.inv(ATA), ATY).T
        self._reset_training()
    
    def _update_batch(self, states, inputs, targets):
        n_states = len(states)
        states = np.vstack(states) + np.random.normal(0, self.state_noise, (n_states, self.n_res))
        extended_states = np.concatenate((states, inputs, np.ones((n_states, 1))), axis=1)
        self._train_info['ATA_terms'].append(np.dot(extended_states.T, extended_states))
        self._train_info['ATY_terms'].append(np.dot(extended_states.T, targets))

    def predict(self, X):
        """
        Predict the output of the ESN on the input X.
        :param X: n_samples x n_input array of input samples
        :return: n_samples x n_output array of output samples
        """
        state = self.eq_state.copy()

        n = X.shape[0]
        outputs = []
        output = np.zeros(self.n_out)

        for i in range(n):
            x = X[i]
            state, output = self._update_reservoir(x, state, output)
            outputs.append(output)

        return np.vstack(outputs)


def test_esn():
    # Run different batch sizes, make sure results are the same.
    n_test = 5000
    batch_sizes = [0, 97, 502]
    # np.random.seed(33)
    esn = EchoStateNetwork(30, 100, 70)
    X = np.random.randn(n_test, 30)
    Y = np.random.randn(n_test, 70)
    w_mats, w_out_mats = [], []
    for batch_size in batch_sizes:
        logging.info("Testing batch_size=%i" % batch_size)
        esn.train_sequence(X, Y, washout=4, batch_size=batch_size)
        esn.finish_training()
        w_mats.append(esn.W_in_res)
        w_out_mats.append(esn.W_res_out)

    for i in range(1, len(w_mats)):
        assert np.allclose(w_mats[i], w_mats[0])
        assert np.allclose(w_out_mats[i], w_out_mats[0])
    logging.info("ESN batch size test passed.")


def __test():
    # unpickle training data:
    import pickle
    with open('drip_data.pkl', 'rb') as f:
        data = pickle.load(f)
        input, output = data
        n_input, n_output = input.shape[1], output.shape[1]

        esn = EchoStateNetwork(n_input, n_reservoir=500, n_output=n_output, spectral_radius=0.99,
                               leak_rate=0.0, linear_out=True, input_scale=1, feedback_scale=0.0)

        esn.train_sequence(input, output)
        esn.finish_training()

        train_out = esn.predict(input)
        import matplotlib.pyplot as plt
        print("Input range: %s, %s" % (np.min(input), np.max(input)))
        plt.imshow(np.hstack((output, train_out))[1000:1100, :].T)
        plt.colorbar()
        plt.show()
        import ipdb
        ipdb.set_trace()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_esn()
    # __test()
