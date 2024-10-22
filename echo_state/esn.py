import numpy as np
import logging
from scipy.optimize import minimize


class EchoStateNetwork(object):
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, leak_rate=0.0,  input_scale=1.0, n_wash=100):
        self.n_in = n_input
        self.n_res = n_reservoir
        self.n_out = n_output
        self.sr = spectral_radius
        self.lr = leak_rate
        self.input_scale = input_scale
        self.n_wash = n_wash

        self._init_weights()

    def _init_weights(self):
        # only W_res_res is fixed, w_in_res and w_out are learned during training

        self.W_in_res = np.random.normal(0, 1, (self.n_res, self.n_in + 1))
        self.W_out = np.random.normal(0, 1, (self.n_out, self.n_res + self.n_in + 1))
        self.W_res_res = np.random.normal(0, 1, (self.n_res, self.n_res))
        self.W_in_res *= self.input_scale
        self.W_res_res *= 1.0 / np.max(np.abs(np.linalg.eigvals(self.W_res_res))) * self.sr

    def _get_equilib_state(self):
        state = np.zeros(self.n_res)
        z_input = np.zeros(self.n_in)
        for _ in range(self.n_wash):
            state = self._update_reservoir(z_input, state)
        return state

    def _update_reservoir(self, x, state):
        """
        Update the reservoir state with input x and current state.
        :param x: input vector
        :param state: current state
        :return: new state
        """
        # State feedback & input:
        # if x.shape[0]==1:
        #      import ipdb; ipdb.set_trace()
        excitations = np.dot(self.W_res_res, state) + np.dot(self.W_in_res, np.append(x, [1]))
        activations = np.tanh(excitations)
        new_state = (self.lr) * state + (1.0-self.lr) * activations

        return new_state

    def train(self, X, Y, washout=100, batch_size=0):
        """
        Train the ESN on the input-output pairs X,Y.
        :param X: n_samples x n_input array of input samples
        :param Y: n_samples x n_output array of output samples
        :param washout: number of initial samples to show the ESN but disregard in final training.
        :param batch_size: if > 0, train the ESN in mini-batches of this size.
        """
        state = self._get_equilib_state()
        print("Training ESN with %i samples, washout=%i, batch_size=%i" % (X.shape[0], washout, batch_size))
        states = []

        # Store these terms to avoid storing a huge number of state vectors.
        # i.e. instead of accumulating A and calculating inv(A'A)A'Y at the
        # end, accumulate A'A and A'Y which are D x D and D x n_out respectively (for each batch).
        ATA_cache = []
        ATY_cache = []

        n = X.shape[0]

        def _get_pinv_parts(states, x_vals, y_vals):
            """
            Get the parts of the pseudo-inverse calculation for a batch of states.  (ATA) and (ATY)
            """
            return ATA, ATY
        batch_n = 0
        batch_start_ind = 0
        n_batches = 1 if batch_size == 0 else n // batch_size + 1

        for i in range(n):
            x = X[i]
            state = self._update_reservoir(x, state)

            if i < washout:
                batch_start_ind = i + 1
                continue

            states.append(state)

            if (batch_size > 0 and len(states) == batch_size) or (i == n-1):
                # if end of batch, or end of data, calculate the pseudo-inverse
                n_states = len(states)
                logging.info("\tBatch %i / %i, %i states" % (1+batch_n, n_batches, n_states))

                states = np.vstack(states)
                extended_states = np.concatenate(
                    (states, X[batch_start_ind:batch_start_ind+n_states], np.ones((n_states, 1))), axis=1)
                ATA_cache.append(np.dot(extended_states.T, extended_states))
                ATY_cache.append(np.dot(extended_states.T, Y[batch_start_ind:batch_start_ind+n_states]))
                states = []
                batch_start_ind = i+1
                batch_n += 1

        # calculate the pseudo-inverse of the A matrix
        ATA = np.sum(ATA_cache, axis=0)
        ATY = np.sum(ATY_cache, axis=0)
        self.W_out = np.dot(np.linalg.pinv(ATA), ATY).T

    def predict(self, X):
        """
        Predict the output of the ESN on the input X.
        :param X: n_samples x n_input array of input samples
        :return: n_samples x n_output array of output samples
        """
        state = self._get_equilib_state()
        n = X.shape[0]
        outputs = []
        for i in range(n):
            x = X[i]

            state = self._update_reservoir(x, state)
            output = np.dot(self.W_out, np.concatenate((state, x, [1])))
            outputs.append(output)
        return np.vstack(outputs)


def test_esn():
    # Run different batch sizes, make sure results are the same.
    n_test = 5000
    batch_sizes = [0, 97, 502]
    np.random.seed(3)
    esn = EchoStateNetwork(30, 100, 70)
    X = np.random.randn(n_test, 30)
    Y = np.random.randn(n_test, 70)
    w_mats, w_out_mats = [], []
    for batch_size in batch_sizes:
        logging.info("Testing batch_size=%i" % batch_size)
        esn.train(X, Y, washout=4, batch_size=batch_size)
        w_mats.append(esn.W_in_res)
        w_out_mats.append(esn.W_out)

    for i in range(1, len(w_mats)):
        assert np.allclose(w_mats[i], w_mats[0])
        assert np.allclose(w_out_mats[i], w_out_mats[0])
    logging.info("ESN batch size test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_esn()
