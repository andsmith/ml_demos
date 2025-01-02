import logging

import numpy as np
import matplotlib.pyplot as plt
from spiral import make_bump
from classify import plot_classifier
from scipy.optimize import minimize


class DecisionStump(object):
    def __init__(self):
        self._dim = None
        self._thresh = None
        self._sign = None

    @staticmethod
    def _eval_thresh( X, dim, thresh, sign):
        """
        Evaluate the stump on the given data.
        :param X: data to evaluate (N x 2)
        :param dim: dimension to split on (0 or 1)
        :param thresh: threshold to split on
        :param sign: sign of the split (1 or -1)
        :returns: predicted labels (N x 1) in {-1, 1}
        """
        exceeding = (X[:,dim] > thresh).astype(np.int32)*2-1
        return sign * exceeding



    def fit(self, X, y, sample_weight=None):
        """
        Fit the model using the training data.
        :param X: training data (N x d)
        :param y: labels (N x 1), -1 or 1
        :param sample_weight: weights for each sample (N x 1) 
        """
        X = np.array(X)
        N, d = X.shape
        if sample_weight is None:
            sample_weight = np.ones(N).astype(np.float64)  / N
        best_error = np.inf

        for dim in range(d):
            low, high = X[:,dim].min(), X[:,dim].max()
            # Check midpoint between each pair of points if there are < 100 unique values, otherwise
            # check 100 points between the min and max
            if len(np.unique(X[:,dim])) < 100:
                values = np.unique(X[:,dim])
                # pad with min and max + epsilon
                values = np.hstack([low-.01, values, high + .01])
                threshes =  (values[1:] + values[:-1]) / 2
            else:
                threshes = np.linspace(low, high, 100)

            for thresh in threshes:
                for sign in [-1, 1]:
                    
                    y_hat = self._eval_thresh(X, dim, thresh, sign)
                    error = np.sum(sample_weight[y != y_hat])
                    if error < best_error:
                        best_error = error
                        self._dim = dim
                        self._thresh = thresh
                        self._sign = sign
                        #print("New best stump: dim %i, thresh %.3f, sign %i, error %.3f" % (dim, thresh, sign, error))
        return self
    
    def predict_proba(self, X):
        return np.hstack([1 - self.predict(X), self.predict(X)])
    
    def predict(self, X):
        """
        Predict the labels of the data.
        :param X: data to predict (N x d)
        :return: predicted labels (N x 1)
        """
        exceeding = (X[:,self._dim] > self._thresh) * 2 - 1  # -1 or 1 if exceeding threshold
        matching = self._sign * exceeding
        return matching
    
    def score(self, X, y):
        """
        Return the accuracy of the model on the data.
        :param X: data to predict (N x d)
        :param y: true labels (N x 1)
        :return: accuracy
        """
        return np.mean(self.predict(X) == y)
    
    def plot(self, axis, *args, **kwargs):
        """
        Plot the decision boundary of the model on the given axis.
        :param axis: axis to plot on
        """
        #print("Plotting stump on dim %i, thresh %.3f" % (self._dim, self._thresh))
        if self._dim is None:
            return
        if self._dim == 0:
            lims = axis.get_ylim()
            p0 = np.array([self._thresh, lims[0]])
            p1 = np.array([self._thresh, lims[1]])
        else:
            lims = axis.get_xlim()
            p0 = np.array([lims[0], self._thresh])
            p1 = np.array([lims[1], self._thresh])
        axis.plot([p0[0], p1[0]], [p0[1], p1[1]],':' ,color='black', *args, **kwargs)


class Perceptron(object):
    def __init__(self):
        self._weights = None
        self._bias = None


    def predict_proba(self, X):
        p_1 =  1. / (1 + np.exp(-np.dot(X, self._weights) - self._bias)).reshape(-1,1)
        return np.hstack([1 - p_1, p_1])

    def fit(self, X, y, sample_weight = None):
        """
        Fit the model using the training data.
        :param X: training data (N x d)
        :param y: labels (N x 1), 0 or 1
        :param sample_weight: weights for each sample (N x 1) 
        """
        X = np.array(X)
        N, d = X.shape
        if sample_weight is None:
            sample_weight = np.ones(N).astype(np.float64) 

        # minimize the weighted error
        def loss(wb):
            w = wb[:-1]
            b = wb[-1]
            excitation = np.dot(X, w) + b
            activation = 1. / (1 + np.exp(-excitation))
            error = (activation - y) ** 2
            return np.sum(sample_weight * error)
        
        res = minimize(loss, np.zeros(d + 1))   
        self._weights = res.x[:-1]
        self._bias = res.x[-1]
        return self
    
    def predict(self, X):
        """
        Predict the labels of the data.
        :param X: data to predict (N x d)
        :return: predicted labels (N x 1)
        """
        return self.predict_proba(X)[:,1] > .5
    
    def score(self, X, y):
        """
        Return the accuracy of the model on the data.
        :param X: data to predict (N x d)
        :param y: true labels (N x 1)
        :return: accuracy
        """
        return np.mean(self.predict(X) == y)
    
    def plot(self, axis, *args, **kwargs):
        """
        Plot the decision boundary of the model on the given axis.
        :param axis: axis to plot on
        """
        if self._weights is None:
            return
        coeffs = self._weights
        intercept = self._bias
        if coeffs[1] == 0:
            p0 = np.array([-intercept / coeffs[0], 0])
            p1 = np.array([(-intercept - coeffs[1]) / coeffs[0], 1])
        else:
            p0 = np.array([0, -intercept / coeffs[1]])
            p1 = np.array([1, (-intercept - coeffs[0]) / coeffs[1]]) 
                          
        axis.plot([p0[0], p1[0]], [p0[1], p1[1]], color='black',*args, **kwargs)  
    
        
def test_and_plot_stump():
    fig, ax = plt.subplots(2, 2)
    X, y = make_bump(20, 0.3, 0.5, 0.0, 0.0, random=False, separable=True)

    # plot data
    ax[0,0].plot(X[y==-1, 0], X[y==-1, 1], '.', color='red',  markersize=2)
    ax[0,0].plot(X[y==1, 0], X[y==1, 1], '.', color='blue', markersize=2)
    ax[0,0].set_title('Bump dataset (0=red, 1=blue)')
    ax[0,0].set_aspect('equal')
    ax[0,0].yaxis.set_visible(False)
    ax[0,0].xaxis.set_visible(False)



    clf = DecisionStump()
    clf.fit(X, y)

    plot_classifier(ax[0,1], X, y, clf, None, markersize=2)
    ax[0,1].set_title('Decision Stump\nunweighted')
    plt.tight_layout()
    


    # fit perceptron to weighted data (light bump)
    clf2 = DecisionStump()
    # set weights of "bump" to 1% original
    w = np.ones(X.shape[0]) / X.shape[0]
    w[(y==-1 ) & (X[:,1] >= 0.5)] *= 0.01
    clf2.fit(X, y, sample_weight=w)
    plot_classifier(ax[1,0], X, y, clf2,None, markersize=2)

    ax[1,0].set_title('D-Stump, \n(bump wt. x .01)')
    
    clf3 = DecisionStump()
    # set weights of "bump" to 100x original
    w = np.ones(X.shape[0]) / X.shape[0]
    w[(y==-1 ) & (X[:,1] >= 0.5)] *= 100
    clf3.fit(X, y, sample_weight=w)
    
    plot_classifier(ax[1,1], X, y, clf3,None, markersize=2)
    ax[1,1].set_title('D-Stump\n(bump wt. x 100)')
    plt.tight_layout()
    plt.show()

def test_and_plot_perceptron():
    fig, ax = plt.subplots(2, 2)

    # data
    X, y = make_bump(20, 0.3, 0.5, 0.0, 0.0, random=False, separable=True)
    y = np.int32(y)
    ax[0,0].plot(X[y==0, 0], X[y==0, 1], '.', color='red',  markersize=2)
    ax[0,0].plot(X[y==1, 0], X[y==1, 1], '.', color='blue', markersize=2)
    ax[0,0].set_title('Bump dataset (0=red, 1=blue)')

    ax[0,0].set_aspect('equal')
    ax[0,0].yaxis.set_visible(False)
    ax[0,0].xaxis.set_visible(False)
    # fit perceptron to unweighted data

    clf1 = Perceptron()
    clf1.fit(X, y)
    print(clf1._weights, clf1._bias)
    plot_classifier(ax[0,1], X, y, clf1,None, markersize=2)
    ax[0,1].set_title('Perceptron, unweighted data')

    # fit perceptron to weighted data (light bump)
    clf2 = Perceptron()
    # set weights of "bump" to 1% original
    w = np.ones(X.shape[0]) / X.shape[0]
    w[(y==-1 ) & (X[:,1] >= 0.5)] *= 0.01
    clf2.fit(X, y, sample_weight=w)
    plot_classifier(ax[1,0], X, y, clf2,None, markersize=2)

    ax[1,0].set_title('Perceptron, \n(bump wt. x .01)')
    
    # fit perceptron to weighted data (heavy bump)
    clf3 = Perceptron()
    # set weights of "bump" to 10% original
    w = np.ones(X.shape[0]) / X.shape[0]
    w[(y==-1 ) & (X[:,1] >= 0.5)] *= 100
    clf3.fit(X, y, sample_weight=w)
    
    plot_classifier(ax[1,1], X, y, clf3,None, markersize=2)
    ax[1,1].set_title('Perceptron\n(bump wt. x 100)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_and_plot_stump()

