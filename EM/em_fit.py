"""
Base class for mixture components and mixture models
"""
from util import normalize_log_probs
import numpy as np
from abc import ABC, abstractmethod
from util import plot_dist, plot_classification, sum_log_probs, Histogram, random_colors
import matplotlib.pyplot as plt


class ProbDist(ABC):

    @abstractmethod
    def pdf(self, x):
        """
        Return the probability density at x.
        :param x: N element array of x values
        :return: N element array of probabilities
        """
        pass

    @abstractmethod
    def log_pdf(self, x):
        """
        Return the log probability density at x.
        :param x: N element array of x values
        :return: N element array of log probabilities
        """
        pass

    @abstractmethod
    def sample(self, n):
        """
        Sample from the model (i.e. run the generative process).
        :param n: the number of samples to draw
        :return: n samples
        """
        pass

    @staticmethod
    @abstractmethod
    def from_data(cls, data, example_weights=None):
        """
        Estimate the parameters of the distribution from data.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_random(cls, spread):
        """
        Generate a random instance of the distribution (for generating demo distributions) by
        sampling N(0, spread) for the mean and a random value for the standard deviation.
        """
        pass


class MixtureModel(object):
    def __init__(self, components, priors, colors=None):
        self.nc = len(priors)
        self.priors = priors
        self.components = components
        self.colors = colors if colors is not None else random_colors(self.nc)

    def pdf(self, x):
        """
        Return the probability density at x.
        :param x: N element array of x values
        :return: N element array of probabilities
        """
        return np.sum([self.priors[i] * self.components[i].pdf(x)
                       for i in range(len(self.priors))], axis=0)

    def sample(self, n):
        """
        Sample from the model (i.e. run the generative process).
        :param n: the number of samples to draw
        :return: n samples
        """
        y = np.random.choice(self.nc, n, p=self.priors)
        values = np.zeros(n)
        for c in range(self.nc):
            values[y == c] = self.components[c].sample(np.sum(y == c))
        return values

    def classify(self, x):
        """
        Return p(y|x) for each component y and data point x.
        :param x: N element array of x values
        :return: N x M matrix of probabilities, where M is the number of components.
        """
        log_probs = np.array([component.log_pdf(x) + np.log(self.priors[i])
                             for i, component in enumerate(self.components)]).T
        return normalize_log_probs(log_probs)

    @staticmethod
    def from_data(component_type, x, n, max_iter=100, animate_interval=0):
        colors = random_colors(n)
        priors, components = fit_em(component_type, x, n, max_iter, colors, animate_interval)
        return MixtureModel(components, priors, colors=colors)


def fit_em(component_type, x, n, max_iter=10, colors=None, animate_interval=0):
    """
    Fit a 1-d mixture model to the data using the EM algorithm:

        Alternate between the E-step:

            Estimate the "responsibilities" of each component for each x.  Using Bayes' rule:

                p(y_i|x) = p(y_i) * p(x|y_i) / p(x), where

            p(y=i) is the prior probability of component i,
            p(x|y_i) is the density model of component i,
            p(x) is unnecessary to calculate, since p(y=0|x) + p(y=1|x) = 1, etc. for more components.


        and the M-step:

            Maximize the likelihood of p(x) given the estimated responsibilities, updating the priors and components:
                p(y_i) is the weighted number of points assigned to each component
                p(x|y) is the mean and standard deviation of all points, weighted by responsibility.

    if animate_interval > 0, for each iteration that is a multiple of it:
        * plot each M-step (show the weights, the previous component's distribution and the updated version)
        * plot the E-step (show the responsibilities as different colors)

    :param component_type: the class of the components to fit (subclass of ProbDist)
    :param x: the data, an M element array of real values
    :param n: the number of components to fit
    :param colors:  list of n rgb colors, one for each component (or None for no plotting)
    :param max_iter: the maximum number of iterations to run
    :param animate_interval: the number of iterations between plots (0 for no animation)
    :return: priors (an n-element array w/ the mixing ratios),
                components (list of ProbDist objects)
    """
    if colors is not None and animate_interval > 0:
        fig, ax = plt.subplots(2, 1, sharex=True)
        x_range = np.max(x) - np.min(x)
        y_range = x_range/15
        y = np.random.rand(x.shape[0]) * y_range
        hist = Histogram(x, 50)

    # start with uniform priors and random components
    priors = np.ones(n) / n  # p(y)
    components = [component_type(np.random.rand(), 1.0)
                  for _ in range(n)]  # p(x|y)

    def _get_log_probs():
        # return log(p(x|y) * p(y)) for each x and y
        return np.array([component.log_pdf(x) + np.log(priors[i])
                         for i, component in enumerate(components)]).T

    last_ll = -np.inf
    converged = False
    log_probs = None
    for iter in range(max_iter):

        # E-step (run in log-space to avoid numerical underflow)
        log_probs = log_probs if log_probs is not None else _get_log_probs()
        responsibilities = normalize_log_probs(log_probs)  # p(z|x)

        # M-step, priors
        total_counts = np.sum(responsibilities, axis=0)
        priors = total_counts / np.sum(total_counts)  # p(z)

        # M-step, components:
        for i in range(n):
            # p(x|z)
            components[i] = component_type.from_data(x, weights=responsibilities[:, i])

        # Calculate the log likelihood & save for next iteration's E-step
        log_probs = _get_log_probs()
        log_likelihood = np.sum(sum_log_probs(log_probs))
        diff = log_likelihood - last_ll
        rel_diff = diff / np.abs(log_likelihood)

        report_str = "Iteration %d: - log likelihood = %.7f (+ %.2f %%)" % (iter, -log_likelihood, rel_diff * 100)
        if animate_interval == 0 or animate_interval > 0 and iter % 10 == 0:
            print(report_str)

        if animate_interval > 0 and iter % animate_interval == 0:
            # E-step, show new distributions of each component and resulting classification
            # clear plots
            ax[0].clear()
            hist.plot(ax[0])
            full_model = MixtureModel(components, priors, colors=colors)
            plot_dist(ax[0], full_model, linestyle='--', n_pts=1000, label='Fit')
            for i, component in enumerate(components):
                plot_dist(ax[0], component, n_pts=1000, weight=priors[i], color=colors[i], label='C-%d' % i)
            ax[1].clear()
            responsibilities = np.argmax(log_probs, axis=1)
            plot_classification(ax[1], np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)]),
                                responsibilities, colors)
            # plt.show()
            ax[0].legend(loc='upper right')
            ax[0].set_title("Iteration %d: -LL = %.2f (+ %.2f %%)" % (iter, -log_likelihood, rel_diff * 100)
                            )
            ax[1].set_title("current classification")
            plt.pause(0.1)
            # import ipdb; ipdb.set_trace()

        # Check for convergence
        if rel_diff < 1e-10:
            converged = True
            break

        last_ll = log_likelihood

    if not converged:
        print("Did not converge after %d iterations" % max_iter)
    else:
        print("Converged after %d iterations" % iter)
    return priors, components
