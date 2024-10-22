
"""
As an alternative to the ESN, we can use a simple feedforward neural network 
to predict the reservoir_state -> output_state mapping.
"""
import numpy as np
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class NNet(object):
    def __init__(self, n_in, n_hidden, n_out, lin_out=False):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lin_out = lin_out
        self._init_weights()

    def _init_weights(self):
        self.W_in_hid = np.random.normal(0, 1, (self.n_hidden, self.n_in + 1))
        self.W_hid_out = np.random.normal(0, 1, (self.n_out, self.n_hidden + 1))

    @staticmethod
    def _eval_net(w_hidden, w_out, x, lin_out=False):
        n = x.shape[0]
        x = np.concatenate((x, np.ones((n, 1))), axis=1)
        hid = np.tanh(np.dot(w_hidden,x.T))
        hid = np.concatenate((hid, np.ones((1, n))), axis=0)
        out = np.dot(w_out, hid)
        if not lin_out:
            out = np.tanh(out)
        return out

    def eval(self, X):
        return self._eval_net(self.W_in_hid, self.W_hid_out, X, self.lin_out)

    def train(self, X, Y):

        n_w_in = (self.n_in + 1) * self.n_hidden
        w_in_shape = (self.n_hidden, self.n_in + 1)
        w_out_shape = (self.n_out, self.n_hidden + 1)

        def _sq_err(weights):
            w_hid = weights[:n_w_in].reshape(w_in_shape)
            w_out = weights[n_w_in:].reshape(w_out_shape)
            y_hat = self._eval_net(w_hid, w_out, X, self.lin_out)
            err = (y_hat - Y)**2.
            sse = np.sum(err)
            #print(y_hat[:,:5].T, Y[:5])
            return sse

        weights = np.concatenate((self.W_in_hid.flatten(), self.W_hid_out.flatten()))

        res = minimize(_sq_err, weights, method='L-BFGS-B')
        if not res.success:
            logging.error("Optimization failed: %s" % res.message)
        else:
            self.W_in_hid = res.x[:n_w_in].reshape(w_in_shape)
            self.W_hid_out = res.x[n_w_in:].reshape(w_out_shape)

        return res
    
from sklearn import datasets


def _get_iris_tests():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    def _make_input(class_1, class_2):
        r= np.concatenate((X[Y == class_1], X[Y == class_2]))
        print("input: ",r.shape)
        return r

    def _make_targ(class_1, class_2):
        output = np.zeros(np.sum((Y == class_1) | (Y == class_2))) - 1
        output[Y[(Y == class_1) | (Y == class_2)] == class_1] = 1
        print("output:", output.shape)
        return output
    
    # Add tests here, x is N x D, y is N x 1
    return {'iris':
             [(_make_input(0,1), _make_targ(0,1)[:,np.newaxis]),
              (_make_input(1,2), _make_targ(1,2)[:,np.newaxis]),
              (_make_input(0,2), _make_targ(0,2)[:,np.newaxis])],
             }



def test_nnet(n_hidden=10):
  
    def _get_accuracy(X, Y):
        nnet = NNet(X.shape[1], n_hidden, 1)
        out_pre = nnet.eval(X)
        nnet.train(X, Y)
        out_post = nnet.eval(X)
        plt.plot(out_pre.flatten(), 'r',linewidth=2, label='pre-trained output')
        plt.plot(out_post.flatten(), 'b-', label='post-trained output')
        plt.plot(Y.flatten(), 'g:', label='target')
        plt.legend()
        plt.show()
        Y_hat = nnet.eval(X)
        return np.mean(np.round(Y_hat) == Y)

    tests = _get_iris_tests()

    for name in tests:
        for i, (X, Y) in enumerate(tests[name]):
            #import ipdb;ipdb.set_trace()
            print("Test #%i on %s dataset, x.shape=%s, y.shape=%s, output labels:  %s" % (i, name, X.shape, Y.shape, set(Y.flatten().tolist())))
            print("Accuracy: %.3f" % _get_accuracy(X, Y))


if __name__ == "__main__":
    test_nnet()
    logging.info("NNet tests passed.")
