from make_data import make_spiral_data, make_bump
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import logging
from multiprocessing import Pool, cpu_count
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy.optimize import minimize
from abc import ABC, abstractmethod


from plotting import plot_classifier

class ClassifierTester(object):
    def __init__(self, dataset, classifiers):
        """
        :param dataset: tuple of X, y (Nx2, N)
        :param classifiers: [[classif_1, classif_2], [classif_3, classif_4], ...] 
          (each classif_i is a function that takes X, y and returns a model, a function
          that takes X and returns the predicted p(y=1|x))
        :param n: tuple of (rows, cols) for the plot
        """
        self._x, self._y = dataset
        self._classif = classifiers
        self._n = len(classifiers), len(classifiers[0])
        self._run_test()

    def _run_test(self):
        logging.info("Running %i tests:" % (self._n[0] * self._n[1]))
        for row in self._classif:
            for classif in row:
                logging.info("Fitting %s" % classif)
                classif.fit(self._x, self._y)
                score = classif._model.score(self._x, self._y)
                logging.info("\tScore: %f" % score)

    def plot(self, X, y, boundary):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(*self._n)
        if self._n[0] == 1:
            if self._n[1] == 1:
                axs = np.array([[axs]])
            else:
                axs = axs[np.newaxis, :]

        for i, row in enumerate(self._classif):
            for j, classifier in enumerate(row):
                accuracy = plot_classifier(axs[i, j], self._x, self._y, classifier, boundary)

                axs[i, j].set_title(f'{classifier}\nAccuracy: {accuracy:.2f}')
        # axs[0, 0].legend()
        plt.tight_layout()


class ClassifierWrapper(ABC):
    @abstractmethod
    def __str__(self):
        pass

    def fit(self, X, y, w=None):
        pass

    def eval(self, X):
        pass


class LRClass(ClassifierWrapper):
    def __init__(self):
        self._model = None
        logging.info("LogisticRegression() created.")

    def __str__(self):
        return 'Logistic Regression'

    def fit(self, X, y, w=None):
        self._model = LogisticRegression().fit(X, y, sample_weight=w)
        return self

    def eval(self, X):
        probs = self._model.predict_proba(X)
        labels = np.argmax(probs, axis=1) == 1
        return labels


class Perceptron(ClassifierWrapper):
    def __init__(self):
        self._model = None
        logging.info("Perceptron() created.")

    def __str__(self):
        return 'Perceptron (NNet with 1-hidden unit)'

    @staticmethod
    def _get_activation(net_weights, X):
        activation = np.dot(X, net_weights[:-1]) + net_weights[-1]
        return np.tanh(activation)

    def fit(self, X, y, w=None, sample_weight=None):
        w = sample_weight if (w is None and sample_weight is not None) else w
        w = np.ones(len(y)) if w is None else w

        def errn_fn(net_weights):
            output = self._get_activation(net_weights, X)
            error = np.sum(output[~y] * w[~y]) + np.sum(1 - output[y]*w[y])
            return error

        self._model = minimize(errn_fn, np.random.rand(3))
        return self

    def predict_proba(self, X):
        return self._get_activation(self._model.x, X)

    def eval(self, X):
        return self.predict_proba(X) > 0.0

    def predict(self, X):
        return self.eval(X).astype(int)

    def plot(self, ax):
        # draw decision boundary
        x_coeff, y_coeff, bias = self._model.x
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        x = np.linspace(*x_lim, 100)
        y = (-bias - x_coeff * x) / y_coeff
        ax.plot(x, y, 'r-')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


def test_and_plot_perceptron():
    fig, ax = plt.subplots(2, 2)
    print(ax.shape)
    # data
    X, y = make_bump(20, 0.3, 0.5, 0.0, 0.0, random=False, separable=True)
    ax[0,0].plot(X[y, 0], X[y, 1], '.', color='red',  markersize=3)
    ax[0,0].plot(X[~y, 0], X[~y, 1], '.', color='blue', markersize=3)
    ax[0,0].set_title('Bump dataset')

    # unweighted result
    p = Perceptron()
    p.fit(X, y)
    print(p._model.x)
    plot_classifier(ax[0,1], X, y, p, None)
    p.plot(ax[0,1])
    ax[0,1].set_title('Perceptron w/unweighted data')
    
    # weighted, make the "bump" points have very little weight
    w = np.ones(len(y)) / len(y)
    w02 = w[0]
    #print(p._model.x)

    '''
        w02 = w[0]

    w[(~y) & (X[:, 1] > 0.5)] /= -100
    p2 = Perceptron()
    p2.fit(X, y, w=w)
    plot_classifier(ax[1,0], X, y, p2, None)
    # plot circle around points with low weight
    ax[1,0].plot(X[(w < w02), 0], X[(w < w02), 1], 'o', color='black', markersize=10, fillstyle='none')
    ax[1,0].set_title('Perceptron w/weighted data (light bump)')
    # weighted so bump has high weight
    plt.subplot(2, 2, 4)
    w = np.ones(len(y)) / len(y)
    w_orig = w.copy()
    w[(~y) & (X[:, 1] > 0.5)] *= 10
    p = Perceptron()
    p.fit(X, y, w=w)
    plot_classifier(plt.gca(), X, y, p, None)
    # plot black circle around points with high weight
    plt.plot(X[(w > w0), 0], X[(w > w0), 1], 'o', color='black', markersize=10, fillstyle='none')
    plt.title('Perceptron w/weighted data (heavy bump)')
    '''
    plt.show()


class DTClass(ClassifierWrapper):
    def __init__(self, **kwargs):
        self._model = None
        self._kwargs = kwargs
        logging.info("DecisionTreeClassifier(%s) created." % ", ".join([f'{k}: {v}' for k, v in kwargs.items()]))

    def __str__(self):
        return 'Decision Tree('+", ".join([f'{k}: {v}' for k, v in self._kwargs.items()]) + ')'

    def fit(self, X, y, w=None):
        self._model = DecisionTreeClassifier(**self._kwargs).fit(X, y, sample_weight=w)
        # import ipdb; ipdb.set_trace()
        return self

    def eval(self, X):
        return self._model.predict(X)


class NBClass(ClassifierWrapper):
    def __init__(self):
        logging.info("GaussianNB() created.")
        self._model = None

    def __str__(self):
        return 'Naive Bayes'

    def fit(self, X, y, w=None):
        self._model = GaussianNB().fit(X, y, sample_weight=w)
        return self

    def eval(self, X):
        return self._model.predict(X)


@ignore_warnings(category=ConvergenceWarning)
def fit_mlp(x, y, w, n_hidden, net_args):
    net = MLPClassifier(hidden_layer_sizes=(n_hidden,), **net_args)

    converged = net.n_iter_no_change < net.max_iter
    net.fit(x, y)
    score = net.score(x, y)
    convergence_str = " (not converged)" if not converged else ""
    print("\t\t\tscore: %f %s" % (score, convergence_str))
    return net, score


class MLPClass(ClassifierWrapper):
    def __init__(self, n_hidden=10, n_reps=10, net_args=None):

        self._model = None
        self._n_reps = n_reps
        self._n_hidden = n_hidden
        self._net_args = {'max_iter': 1000, 'solver': 'lbfgs', 'tol': 1e-10}
        if net_args is not None:
            self._net_args.update(net_args)

        self._n_cpus = cpu_count() - 4
        logging.info("MLPClassifier(n_hidden=%i, n_reps=%i) created." % (n_hidden, n_reps))

    def __str__(self):
        return 'n_hidden=%i' % self._n_hidden

    def fit(self, X, y, w=None):
        logging.info("Fitting NNet(%s), %i times, with %i cores:" % (self, self._n_reps, self._n_cpus))
        work = [(X, y, w, self._n_hidden, self._net_args) for _ in range(self._n_reps)]
        print(self._n_hidden, len(work))

        if self._n_cpus == 1:
            results = [fit_mlp(*work_unit) for work_unit in work]
        else:
            with Pool(self._n_cpus) as pool:
                results = pool.starmap(fit_mlp, work)

        scores = [score for _, score in results]
        best = np.argmax(scores)
        logging.info("\tBest score: %f" % scores[best])
        self._model = results[best][0]

    def eval(self, X):
        return self._model.predict(X)


if __name__ == '__main__':
    # test_and_plot_perceptron()
    logging.basicConfig(level=logging.INFO)
    test_and_plot_perceptron()
    X, y, boundary = make_spiral_data(50, 1.25, 1)
    tests = [[LRClass(), DTClass(max_depth=4), DTClass(max_depth=8)]]  # ,
  #           [MLPClass(n_hidden=5, n_reps=50), MLPClass(n_hidden=10, n_reps=40), MLPClass(n_hidden=20, n_reps=25),],
  #           [MLPClass(n_hidden=50), MLPClass(n_hidden=100), MLPClass(n_hidden=150)]]
    tester = ClassifierTester((X, y), tests)
    tester.plot(X, y, boundary)
    plt.show()
