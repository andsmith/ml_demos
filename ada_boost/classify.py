from spiral import make_spiral_data
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
        for row in self._classif:
            for classif in row:
                logging.info("Fitting %s" % classif)
                classif.fit(self._x, self._y)
                score = classif._model.score(self._x, self._y)
                logging.info("\tScore: %f" % score)

    def plot_result(self, ax, model, boundary, res=500):
        # plot samples (different symbols?)
        ax.plot(self._x[self._y, 0], self._x[self._y, 1], '.', color='gray',  markersize=1)
        ax.plot(self._x[~self._y, 0], self._x[~self._y, 1], '.', color='gray', markersize=1)

        # show true boundary
        ax.plot(boundary[1:-1, 0], boundary[1:-1, 1], 'w:')

        # mark incorrect symbols, calc accuracy
        y_hat = model.eval(self._x)
        incorrect = np.where(y_hat != self._y)[0]
        ax.plot(self._x[incorrect, 0], self._x[incorrect, 1], 'rx', label='incorrect', markersize=4)
        accuracy = 1 - len(incorrect) / len(self._y)
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()

        # show decision boundary
        xx, yy = np.meshgrid(np.linspace(*x_lim, res), np.linspace(*y_lim, res))
        x, y = xx.flatten(), yy.flatten()
        X = np.vstack((x, y)).T
        p = model.eval(X)
        p = p.reshape(res, res)

        # display = DecisionBoundaryDisplay.from_estimator(model._model, X, ax=ax)
        # display.plot(ax=ax, xticks=[], yticks=[])
        ax.contour(xx, yy, p, alpha=0.9, colors='w', levels=[0.5])
        return accuracy

    def plot(self, X, y, boundary):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(*self._n)
        if self._n[0] == 1:
            axs = axs[np.newaxis, :]

        for i, row in enumerate(self._classif):
            for j, classifier in enumerate(row):
                accuracy = self.plot_result(axs[i, j], classifier, boundary)

                axs[i, j].set_title(f'{classifier}\nAccuracy: {accuracy:.2f}')
        # axs[0, 0].legend()
        plt.tight_layout()


class LRClass(object):
    def __init__(self):
        self._model = None

    def __str__(self):
        return 'Logistic Regression'

    def fit(self, X, y):
        self._model = LogisticRegression().fit(X, y)

    def eval(self, X):
        return self._model.predict(X)


class DTClass(object):
    def __init__(self, **kwargs):
        self._model = None
        self._kwargs = kwargs

    def __str__(self):
        return 'Decision Tree('+", ".join([f'{k}: {v}' for k, v in self._kwargs.items()]) + ')'

    def fit(self, X, y):
        self._model = DecisionTreeClassifier(**self._kwargs).fit(X, y)

    def eval(self, X):
        return self._model.predict(X)


class NBClass(object):
    def __init__(self):
        self._model = None

    def __str__(self):
        return 'Naive Bayes'

    def fit(self, X, y):
        self._model = GaussianNB().fit(X, y)

    def eval(self, X):
        return self._model.predict(X)

@ignore_warnings(category=ConvergenceWarning)
def fit_mlp(x, y, n_hidden, net_args):
    net = MLPClassifier(hidden_layer_sizes=(n_hidden,), **net_args)
    
    converged = net.n_iter_no_change < net.max_iter
    net.fit(x, y)
    score = net.score(x, y)
    convergence_str = " (not converged)" if not converged else ""
    print("\t\t\tscore: %f %s" % (score,convergence_str))
    return net, score


class MLPClass(object):
    def __init__(self, n_hidden=10, n_reps=10, net_args=None):

        self._model = None
        self._n_reps = n_reps
        self._n_hidden = n_hidden
        self._net_args = {'max_iter': 1000, 'solver': 'lbfgs'}
        if net_args is not None:
            self._net_args.update(net_args)

        self._n_cpus =  cpu_count() -2

    def __str__(self):
        return 'n_hidden=%i' % self._n_hidden

    def fit(self, X, y):
        logging.info("Fitting NNet(%s), %i times, with %i cores:" % (self, self._n_reps, self._n_cpus)) 
        work = [(X, y, self._n_hidden, self._net_args) for _ in range(self._n_reps)]

        if self._n_cpus == 1:
            results = [fit_mlp(*w) for w in work]
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
    logging.basicConfig(level=logging.INFO)
    X, y, boundary = make_spiral_data(6000, 1.25, 1)
    tests = [[LRClass(), MLPClass(n_reps=100), MLPClass(n_hidden=100)],
             [DTClass(max_depth=4), DTClass(max_depth=6), DTClass(max_depth=10)],
             [MLPClass(n_hidden=15, n_reps=100), MLPClass(n_hidden=25, n_reps=25), MLPClass(n_reps=50,n_hidden=20)]]
    tester = ClassifierTester((X, y), tests)
    tester.plot(X, y, boundary)
    plt.show()
