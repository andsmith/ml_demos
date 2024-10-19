import numpy as np


class EchoStateNetwork(object):
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, leak_rate=0.0, feedback_scale=0.0, input_scale=1.0, n_wash=100):
        self.n_in = n_input
        self.n_res = n_reservoir
        self.n_out = n_output
        self.fs = feedback_scale
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
        self.W_out_res = np.random.normal(0, 1, (self.n_res, self.n_out)) if self.fs != 0 else None

        self.W_in_res *= self.input_scale
        self.W_res_res *= 1.0 / np.max(np.abs(np.linalg.eigvals(self.W_res_res))) * self.sr

    def _get_washed_state(self):
        state=np.zeros(self.n_res)
        z_input=np.zeros(self.n_in)
        output=np.zeros(self.n_out)  # used if feedback is enabled
        for _ in range(self.n_wash):
            state, output=self._update_reservoir(z_input, state, output)  # output * 0 ?
        return state

    def _update_reservoir(self, x, state, prev_out=None):
        """
        Update the reservoir state with input x and current state.
        :param x: input vector
        :param state: current state
        :param prev_out: previous output (used if feedback is enabled)
        :return: new state
        """
        # State feedback & input:
        excitations=np.dot(self.W_res_res, state) + np.dot(self.W_in_res, np.append(x, [1]))
        # Output feedback:
        if self.fs != 0:
            excitations += np.dot(self.W_out_res, prev_out)
        activations=np.tanh(excitations)
        new_state=(self.lr) * state + (1.0-self.lr) * activations
        new_output=np.dot(self.W_out, np.concatenate((new_state, x, [1])))
        return new_state, new_output

    def train(self, X, Y):
        """
        Train the ESN on the input-output pairs X,Y.
        :param X: n_samples x n_input array of input samples
        :param Y: n_samples x n_output array of output samples
        """
        #import ipdb; ipdb.set_trace()
        washout = self.n_wash
        state=self._get_washed_state()
        states=[]
        outputs = []
        for ind, (x, y) in enumerate(zip(X, Y)):
            prev_y = np.zeros(self.n_out) if ind == 0 else Y[ind - 1]
            state, output=self._update_reservoir(x, state, y)
            states.append(state)
            outputs.append(output)
        states=np.vstack(states[washout:])
        import matplotlib.pyplot as plt
        outputs= np.vstack(outputs[washout:])
        #plt.plot(outputs[:,0])
        ##plt.plot(Y[washout:,0])
        #   plt.show()
        extended_states = np.concatenate((states, X[washout:,:], np.ones((X.shape[0]-washout, 1))), axis=1)
        print(X[washout:,:].shape, states.shape, extended_states.shape, Y[washout:,:].shape)
        self.W_out =np.dot(np.linalg.pinv(extended_states), Y[washout:,:]).T

        y_hat = self.predict(X)

        plt.plot(Y)
        plt.plot(y_hat)
        plt.show()

    def predict(self, X):
        """
        Predict the output of the ESN on the input X.
        :param X: n_samples x n_input array of input samples
        :return: n_samples x n_output array of output samples
        """
        state=self._get_washed_state()

        outputs=[np.zeros(self.n_out)]
        for x in X:
            state, output=self._update_reservoir(x, state, outputs[-1])
            outputs.append(output)
        return np.vstack(outputs[1:])


if __name__ == "__main__":
    esn=EchoStateNetwork(30, 100, 70)
    X=np.random.randn(100, 30)
    Y=np.random.randn(100, 70)
    esn.train(X, Y)
    Y_hat=esn.predict(X)
