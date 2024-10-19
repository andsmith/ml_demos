import numpy as np


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

    def _get_washed_state(self):
        state=np.zeros(self.n_res)
        z_input=np.zeros(self.n_in)
        for _ in range(self.n_wash):
            state =self._update_reservoir(z_input, state)  
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
        #if x.shape[0]==1:
        #      import ipdb; ipdb.set_trace()
        excitations=np.dot(self.W_res_res, state) + np.dot(self.W_in_res, np.append(x, [1]))   
        activations=np.tanh(excitations)
        new_state=(self.lr) * state + (1.0-self.lr) * activations

        return new_state

    def train(self, X, Y,washout = 100 ):
        """
        Train the ESN on the input-output pairs X,Y.
        :param X: n_samples x n_input array of input samples
        :param Y: n_samples x n_output array of output samples
        """
        #import ipdb; ipdb.set_trace()
        
        state=self._get_washed_state()
        states=[]
        n = X.shape[0]

        for i in range(n):
            x = X[i]

            state=self._update_reservoir(x, state)
            states.append(state)
            
        states=np.vstack(states[washout:])
        extended_states = np.concatenate((states, X[washout:,:], np.ones((X.shape[0]-washout, 1))), axis=1)
        self.W_out =np.dot(np.linalg.pinv(extended_states), Y[washout:,:]).T
        # import matplotlib.pyplot as plt
        # plt.plot(outputs[:,0])
        # plt.plot(Y[washout:,0])
        #   plt.show()
        
        # y_hat = self.predict(X)
        # print(np.mean(y_hat)-np.mean(Y))    
        # plt.plot(Y[washout:])
        # plt.plot(y_hat)
        # plt.show()

    def predict(self, X):
        """
        Predict the output of the ESN on the input X.
        :param X: n_samples x n_input array of input samples
        :return: n_samples x n_output array of output samples
        """
        state=self._get_washed_state()
        n = X.shape[0]
        outputs=[]
        for i in range(n):
            x = X[i]

            state=self._update_reservoir(x, state)
            output=np.dot(self.W_out, np.concatenate((state, x, [1])))
            outputs.append(output)
        return np.vstack(outputs)


if __name__ == "__main__":
    np.random.seed(3)
    esn=EchoStateNetwork(30, 100, 70)
    X=np.random.randn(100, 30)
    Y=np.random.randn(100, 70)
    esn.train(X, Y,washout=4)
    Y_hat=esn.predict(X)
