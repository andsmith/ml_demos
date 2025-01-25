"""
Learn a Fisher LDA model as a binary classifier.
"""
import numpy as np
import numpy.linalg as la
from mpl_plots import plot_binary_clustering
import matplotlib.pyplot as plt


class FisherLDA(object):
    def __init__(self, ):
        self._fit = False

    def is_fit(self):
        return self._fit

    def fit(self, x, y):
        """
        Fit the Fisher LDA model.
        :returns: predictions (0,1) of the training data
        """
        self._fit = True
        
        # Separate the data points by class
        x1 = x[y == 1]
        x0 = x[y == 0]

        # Compute the means of each class
        m1 = np.mean(x1, axis=0)
        m0 = np.mean(x0, axis=0)

        # Compute the within-class scatter matrix
        s1 = np.cov(x1, rowvar=False)
        s0 = np.cov(x0, rowvar=False)
        sw = s1 + s0

        # Compute the Fisher LDA projection vector
        self.w = la.inv(sw) @ (m1 - m0)

        # compute the threshold
        self.threshold = (m1 @ self.w + m0 @ self.w) / 2

        # Project the data points onto the Fisher LDA line
        return self.assign(x)

    def assign(self, x):
        """
        Project data points onto the Fisher LDA line.
        :param x: data points (N x D)
        :return: projected data points (N)
        """
        return x @ self.w > self.threshold


def test_fisher_lda():
    n = 1000
    random = np.random.RandomState(42)
    x = np.vstack([random.randn(n, 2),
                   random.randn(n, 2) + np.array([3, 3])])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int32)
    model = FisherLDA()
    train_preds = model.fit(x, y)
    fig, ax = plt.subplots()
    plot_binary_clustering(ax, x, train_preds, y)
    plt.show()


if __name__ == '__main__':
    test_fisher_lda()
