import numpy as np
import logging
from scipy.optimize import minimize
import logging

class EchoStateNetwork(object):
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, leak_rate=0.0,  input_scale=1.0, n_wash=0, feedback_scale=0., linear_out=False):
        self.linear_out = linear_out
        self.n_in = n_input
        self.n_res = n_reservoir
        self.n_out = n_output
        self.sr = spectral_radius
        self.lr = leak_rate
        self.input_scale = input_scale
        self.n_wash = n_wash
        self.feedback_scale = feedback_scale

        self._init_weights()

    def _init_weights(self):

        self.W_in_res = np.random.normal(0, 1, (self.n_res, self.n_in + 1)) * self.input_scale
        self.W_res_out = np.random.normal(0, 1, (self.n_out, self.n_res + self.n_in + 1))
        self.W_out_res = np.random.normal(0, 1, (self.n_res, self.n_out)) if self.feedback_scale > 0 else None
        self.W_res_res = np.random.normal(0, 1, (self.n_res, self.n_res))
        # enforce spectral radius:
        self.W_res_res *= 1.0 / np.max(np.abs(np.linalg.eigvals(self.W_res_res))) * self.sr

    def _get_equilib_state(self):
        state = np.zeros(self.n_res)
        z_input = np.zeros(self.n_in)
        for _ in range(self.n_wash):  
            state, _ = self._update_reservoir(z_input, state,np.zeros(self.n_out))
        return state

    def _update_reservoir(self, x, state, last_output=None):
        """
        Update the reservoir state with input x and current state.
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

    def train(self, X, Y, washout=0, batch_size=10000):
        """
        Train the ESN on the input-output pairs X,Y.
        :param X: n_samples x n_input array of input samples
        :param Y: n_samples x n_output array of output samples
        :param washout: number of initial samples to show the ESN but disregard in final training.
        :param batch_size: if > 0, collect pinv terms in batches (Should be identical results.)
        """
        state = self._get_equilib_state()
        logging.info("Training ESN with %i samples, washout=%i, batch_size=%i" % (X.shape[0], washout, batch_size))

        # Store these terms to avoid storing a huge number of state vectors.
        # i.e. instead of accumulating A and calculating inv(A'A)A'Y at the
        # end, accumulate A'A and A'Y which are D x D and D x n_out respectively (for each batch).
        ATA_cache = []
        ATY_cache = []     
        states = []


        n = X.shape[0]

        batch_n = 0
        batch_start_ind = 0
        n_batches = 1 if batch_size == 0 else n // batch_size + 1

        for i in range(n):

            x = X[i]
            teacher = Y[i-1] if i > 0 else np.zeros(self.n_out)
            state, output = self._update_reservoir(x, state, teacher)

            if i < washout:
                batch_start_ind = i + 1
                continue

            states.append(state)

            # Batch part 1:  Accumulate the terms:
            if (batch_size > 0 and len(states) == batch_size) or (i == n-1):
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
                
            
        # Batch part 2:
        ATA = np.sum(ATA_cache, axis=0) - np.eye(self.n_res + self.n_in + 1) * .1  # regularization
        ATY = np.sum(ATY_cache, axis=0)
    
        """
        # non-batch version  (remove Batch parts 1 and 2 to use.)
        states = np.vstack(states)
        extended_states = np.concatenate(
            (states, X[washout:], np.ones((states.shape[0], 1))), axis=1)
        ATA=(np.dot(extended_states.T, extended_states))
        ATY=(np.dot(extended_states.T, Y[washout:]))
        """

        self.W_res_out = np.dot(np.linalg.pinv(ATA), ATY).T


    def predict(self, X):
        """
        Predict the output of the ESN on the input X.
        :param X: n_samples x n_input array of input samples
        :return: n_samples x n_output array of output samples
        """
        state = self._get_equilib_state()
        
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
    np.random.seed(3)
    esn = EchoStateNetwork(30, 100, 70)
    X = np.random.randn(n_test, 30)
    Y = np.random.randn(n_test, 70)
    w_mats, w_out_mats = [], []
    for batch_size in batch_sizes:
        logging.info("Testing batch_size=%i" % batch_size)
        esn.train(X, Y, washout=4, batch_size=batch_size)
        w_mats.append(esn.W_in_res)
        w_out_mats.append(esn.W_res_out)

    for i in range(1, len(w_mats)):
        assert np.allclose(w_mats[i], w_mats[0])
        assert np.allclose(w_out_mats[i], w_out_mats[0])
    logging.info("ESN batch size test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_esn()
