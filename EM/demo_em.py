"""
Demonstrate EM For fitting mixtures of 1-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
from util import Histogram, plot_classification, plot_dist, normalize_log_probs, sum_log_probs


class GaussianComponent(object):
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __str__(self):
        return "N(%.2f, %.2f)" % (self.mean, self.sd)

    @staticmethod
    def from_data(data, weights):
        """
        Estimate the mean and standard deviation of a Gaussian from data.
        """
        mean = np.average(data, weights=weights)
        sd = np.sqrt(np.average((data - mean) ** 2, weights=weights))
        return GaussianComponent(mean, sd)

    def pdf(self, x):
        return 1.0 / (self.sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - self.mean) / self.sd) ** 2)

    def log_pdf(self, x):
        return -0.5 * np.log(2 * np.pi) - np.log(self.sd) - 0.5 * ((x - self.mean) / self.sd) ** 2

    def sample(self, n):
        return np.random.normal(self.mean, self.sd, n)


class GaussianMixtureModel(object):
    def __init__(self, priors, components, colors=None):
        self.priors = priors
        self.components = components
        self._n = len(priors)
        self.colors = colors

    def __str__(self):
        return "(" + ", ".join(["%.2f, %s" % (self.priors[i], self.components[i])
                                for i in range(self._n)]) + ")"

    @staticmethod
    def from_random(n, spread=1.0):
        """
        Generate a random Gaussian mixture model.
        """
        priors = np.ones(n)/n
        priors /= priors.sum()
        means = np.random.randn(n) * spread
        sds = np.random.rand(n) 
        sds = np.maximum(sds, 0.1)  # not too small
        components = [GaussianComponent(means[i], sds[i])
                      for i in range(n)]
        return GaussianMixtureModel(priors, components)

    @staticmethod
    def _rand_colors(n):
        return np.random.rand(n, 3)

    @staticmethod
    def from_data(data, n, max_iter=10, animate_interval=0):
        colors = GaussianMixtureModel._rand_colors(n) 
        priors, components = GaussianMixtureModel.fit_em(data, n, max_iter, colors, animate_interval)
        return GaussianMixtureModel(priors, components, colors=colors)

    def classify(self, x):
        """
        Return p(y|x) for each component y and data point x.
        :param x: N element array of x values
        :return: N x M matrix of probabilities, where M is the number of components.
        """
        log_probs = np.array([component.log_pdf(x) + np.log(self.priors[i])
                             for i, component in enumerate(self.components)]).T
        return normalize_log_probs(log_probs)

    def pdf(self, x):
        """
        Return the probability density at x.
        :param x: N element array of x values
        :return: N element array of probabilities
        """
        return np.sum([self.priors[i] * self.components[i].pdf(x)
                       for i in range(self._n)], axis=0)

    def sample(self, n):
        """
        Sample from the model (i.e. run the generative process).
        :param n: the number of samples to draw
        :return: n samples
        """
        y = np.random.choice(self._n, n, p=self.priors)
        return np.array([self.components[y[i]].sample(1)[0] for i in range(n)])

    @staticmethod
    def fit_em(x, n, max_iter=10, colors=None, animate_interval=0):
        """
        Fit the model to the data using the EM algorithm:

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

        if self._plot, for each iteration:
            * plot each M-step (show the weights, the previous component's distribution and the updated version)
            * plot the E-step (show the responsibilities as different colors)

        :param x: the data, an M element array of real values
        :param n: the number of components to fit
        :param colors:  list of n rgb colors, one for each component (or None for no plotting)
        :param max_iter: the maximum number of iterations to run
        :param animate_interval: the number of iterations between plots (0 for no animation)
        :return: priors (an n-element array w/ the mixing ratios),
                 components (list of GaussianComponent objects)
        """
        if colors is not None and animate_interval > 0:
            fig, ax = plt.subplots(2, 1, sharex=True)
            x_range = np.max(x) - np.min(x)
            y_range = x_range/15
            y = np.random.rand(x.shape[0]) * y_range

        priors = np.ones(n) / n  # p(y), updated by the M-step
        components = [GaussianComponent(np.random.rand(), 1.0)
                      for _ in range(n)]  # p(x|y), updated by the M-step
        hist = Histogram(x, 50)

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
            responsibilities = normalize_log_probs(log_probs)

            # M-step, priors
            total_counts = np.sum(responsibilities, axis=0)
            priors = total_counts / np.sum(total_counts)

            # M-step, components:
            for i in range(n):
                components[i] = GaussianComponent.from_data(x, weights=responsibilities[:, i])

            # Calculate the log likelihood & save for next iteration's E-step
            log_probs = _get_log_probs()
            log_likelihood = np.sum(sum_log_probs(log_probs))
            diff = log_likelihood - last_ll
            rel_diff = diff / np.abs(log_likelihood)

            report_str = "Iteration %d: - log likelihood = %.7f (+ %.2f %%)" % (iter, -log_likelihood, rel_diff * 100)
            print(report_str)

            if animate_interval > 0 and iter % animate_interval == 0:
                # E-step, show new distributions of each component and resulting classification
                # clear plots
                ax[0].clear()
                hist.plot(ax[0])
                full_model = GaussianMixtureModel(priors, components)
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
            if rel_diff < 1e-7:
                converged = True
                break

            last_ll = log_likelihood

        if not converged:
            print("Did not converge after %d iterations" % max_iter)
        else:
            print("Converged after %d iterations" % iter)
        return priors, components


def parse_args():
    """
    syntax> python demo_em.py n_comps_real n_comps_fit [max_iter=100] [spread=1.0] [n_points=1000] 
    """
    parser = ArgumentParser(description="Demonstrate EM for fitting mixtures of 1-D Gaussians")
    parser.add_argument('-r', "--n_real", type=int, default=5, help="Number of components generating data.")
    parser.add_argument('-f', "--n_fit", type=int, default=5, help="Number of components in the fit model")
    parser.add_argument('-i', "--iter", type=int, default=300, help="Maximum number of iterations to run")
    parser.add_argument('-s', "--spread", type=float, default=1.5, help="Spread of the true model")
    parser.add_argument('-p', "--n_points", type=int, default=2000, help="Number of data points to generate")
    parser.add_argument('-a', "--animate_frame", type=int, default=1, help="Animate every n-th frame (or 0 for no animation)")
    args = parser.parse_args()
    return args.n_real, args.n_fit, args.iter, args.spread, args.n_points, args.animate_frame


def _test_plots():
    """
    Test plotting functions.
    """
    fig, ax = plt.subplots(2, 1)

    # test single component
    comp = GaussianComponent(0, 1)
    hist = Histogram(np.random.randn(100000), 50)
    hist.plot(ax[0])
    plot_dist(ax[0], comp, label="%s" % comp)
    ax[0].legend()

    # test mixture model
    model = GaussianMixtureModel.from_random(3)
    hist = Histogram(model.sample(100000), 50)
    hist.plot(ax[1])
    plot_dist(ax[1], model, label="GMM(3)")
    ax[1].legend()

    plt.show()


def demo(n_comps_real, n_comps_fit, max_iter, spread, n_pts=1000, ainmate_interval=1):
    """
    Demonstrate EM for fitting mixtures of 1-D Gaussians.
    """
    # Generate a random Gaussian mixture model
    model = GaussianMixtureModel.from_random(n_comps_real, spread)
    model.colors = GaussianMixtureModel._rand_colors(n_comps_real)
    data = model.sample(n_pts)

    # Fit a Gaussian mixture model to the data
    plt.ion()

    # show the data
    n_pts = data.size
    hist = Histogram(data, 50)
    fig, ax = plt.subplots(2, sharex=True)
    hist.plot(ax[0])
    labels = np.zeros(n_pts, dtype=int)
    y = np.random.rand(n_pts)
    plot_classification(ax[1], np.hstack((data.reshape(-1, 1), y.reshape(-1, 1))), labels, colors=[[1, 1, 1]])
    ax[0].set_ylabel("p(x)")
    fig.suptitle("data + normalized histogram\n(click to start)")
    fig.waitforbuttonpress()

    # animate the fit
    model_fit = GaussianMixtureModel.from_data(data, n_comps_fit, animate_interval=ainmate_interval, max_iter=max_iter)

    # show true and fit models
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].clear()
    # plot mixture distrubtions:
    hist.plot(ax[0], label=None)
    def_blue = np.array((31, 119, 180))/255
    def_orange = np.array((255, 127, 14))/255
    plot_dist(ax[0], model, n_pts=n_pts, label='True', color=def_blue)
    plot_dist(ax[0], model_fit, n_pts=n_pts, label='Fit', linestyle='--', color=def_orange)
    ax[0].legend()
    ax[0].set_xlabel(None)
    ax[0].set_title("True & EM-fit mixture distributions - p(x)")
    ax[1].clear()
    # plot true and fit components
    for i, component in enumerate(model.components):
        plot_dist(ax[1], component, n_pts=n_pts, weight=model.priors[i], color=def_blue)

    for i, component in enumerate(model_fit.components):
        plot_dist(ax[1], component, n_pts=n_pts, weight=model.priors[i], linestyle='--', color=def_orange)
    ax[1].set_title("True & EM-fit components' distributions - p(x|y)p(y)")
    plt.show()
    plt.pause(0)


if __name__ == "__main__":

    plt.style.use('dark_background')
    # test_sum_log_probs()
    n_comps_real, n_comps_fit, max_iter, spread, n_pts, animate_interval = parse_args()
    demo(n_comps_real, n_comps_fit, max_iter, spread, n_pts, ainmate_interval=animate_interval)
