"""
Demonstrate EM For fitting mixtures of 1-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plt
import sys


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


class Histogram(object):
    def __init__(self, x, n_bins):
        margin = 0.05
        x_min, x_max = np.min(x), np.max(x)
        x_span = x_max - x_min
        self._x_min = x_min - margin * x_span
        self._x_max = x_max + margin * x_span
        self._n = n_bins
        self._bin_width = (self._x_max - self._x_min) / self._n
        self._counts, self._bins = np.histogram(x, bins=self._n, range=(self._x_min, self._x_max))
        self._density = self._counts / (len(x) * self._bin_width)

    def plot(self, ax, *args, **kwargs):
        """
        Plot the histogram. (black and white bar outlines)
        :param ax: the axis to plot on
        :param args: additional arguments to pass to ax.plot
        """
        label = "Histogram of x"
        bin_centers = self._bins[:-1] + self._bin_width / 2
        ax.bar(bin_centers, self._density, width=self._bin_width, label=label,
               fill=False, edgecolor='white', *args, **kwargs)
        ax.set_xlim(self._x_min, self._x_max)


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
        priors = np.random.rand(n)
        priors /= priors.sum()
        components = [GaussianComponent(np.random.rand() * spread, np.random.rand())
                      for _ in range(n)]
        return GaussianMixtureModel(priors, components)
    
    @staticmethod
    def _rand_colors(n):
        return np.random.rand(n, 3)

    @staticmethod
    def from_data(data, n, max_iter=10, plot=False):
        colors=GaussianMixtureModel._rand_colors(n) if plot else None
        priors, components = GaussianMixtureModel.fit_em(data, n, max_iter, colors)
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
    def fit_em(x, n, max_iter=10, colors=None):
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
        :param colors:  list of n rgb colors, one for each component
        :param max_iter: the maximum number of iterations to run
        :return: priors (an n-element array w/ the mixing ratios),
                 components (list of GaussianComponent objects)
        """

        priors = np.ones(n) / n  # p(y), updated by the M-step
        components = [GaussianComponent(np.random.rand(), 1.0)
                      for _ in range(n)]  # p(x|y), updated by the M-step

        def _get_log_probs():
            # return log(p(x|y) * p(y)) for each x and y
            return np.array([component.log_pdf(x) + np.log(priors[i])
                             for i, component in enumerate(components)]).T

        last_ll = -np.inf
        converged = False
        log_probs = None
        for iter in range(max_iter):

            # E-step (run in log-space to avoid numerical underflow)
            log_probs = _get_log_probs() if log_probs is None else log_probs
            responsibilities = normalize_log_probs(log_probs)

            # M-step, priors
            total_counts = np.sum(responsibilities, axis=0)
            priors = total_counts / np.sum(total_counts)

            # M-step, components:
            for i in range(n):
                # old_component = components[i]
                components[i] = GaussianComponent.from_data(x, weights=responsibilities[:, i])
                # if self._plot:
                #       self.plot_m_step(old_component, self._components[i], i)

            # Calculate the log likelihood & save for next iteration
            log_probs = _get_log_probs()
            log_likelihood = np.sum(sum_log_probs(log_probs))
            print("Iteration %d: log likelihood = %.2f" % (iter, log_likelihood))
            # if self._plot:
            #    self.plot_e_step(log_likelihood)

            # Check for convergence
            if log_likelihood - last_ll < 1e-6:
                # alt, check no label changes (i.e. responsibilities are stable)
                converged = True
                break

            last_ll = log_likelihood

        if not converged:
            print("Did not converge after %d iterations" % max_iter)
        else:
            print("Converged after %d iterations" % i)
        return priors, components


def plot_classification(ax, points, classifier, colors, aspect=15.0, *args, **kwargs):
    """
    Plot the classification of points by a classifier.

    points are 1-d so spread them out in a rectangle w/given aspect ratio

    :param ax: the axis to plot on
    :param points: N element array of x values
    :param classifier: a function that takes a point and returns a class label
    :param colors: a list of colors to use for each class
    :param aspect: the aspect ratio of the plot
    :param args & kwargs: additional arguments to pass to ax.plot
    """

    probs = classifier(points)
    labels = np.argmax(probs, axis=1)
    x_range = max(points) - min(points)
    y_range = x_range / aspect
    y = np.random.rand(len(points)) * y_range
    for label in set(labels):
        color = colors[label]
        x_vals = points[labels == label]
        y_vals = y[labels == label]
        print(color, points.shape)
        plt.plot(x_vals, y_vals, '.',
                 color=color,
                 label="Class %d" % label,
                 *args, **kwargs)

    ax.set_yticks([])
    ax.set_xlabel("x")


def plot_dist(ax, dist,  weight=1.0, n_pts=1000, *args, **kwargs):
    """
    Plot a distribution over the reals.  
    (Use current axis limits)

    :param ax: the axis to plot on
    :param dist: something with a .pdf() method
    :param n_pts: the number of sample points to plot
    :param args & kwargs: additional arguments to pass to ax.plot
    """
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, n_pts)
    y = dist.pdf(x) * weight
    ax.plot(x, y, *args, **kwargs)
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("p(x)")
    ax.set_xlabel("x")



def parse_args():
    """
    Parse command line arguments
    syntax> python demo_em.py n_comps_real n_comps_fit [spread]
    spread = expected mean of components' standard deviations
    """
    spread = 1.0
    if len(sys.argv) < 3:
        print("Syntax: python demo_em.py n_comps_real n_comps_fit [spread=1.0]")
        sys.exit(1)
    n_comps_real = int(sys.argv[1])
    n_comps_fit = int(sys.argv[2])
    if len(sys.argv) > 3:
        spread = float(sys.argv[3])
    return n_comps_real, n_comps_fit, spread


def sum_log_probs(log_probs):
    """
    Return the log(sum(exp(log_probs))) for each row of log_probs.
    Since adding extremely small numbers will be unstable, we use this fact:

      log(a + b) = log(c * (a + b) / c) 
                 = log(c * a + c * b) - log(c)

    so as long as we pick a constant such that the argument to log() is reasonable,
    the result will be stable.

    Use c =  1 / max(a, b) so that the argument is in (1, 2].

    :param log_probs: M x N matrix of log probabilities, log_probs[i,j] = log(p(x_i|y_j) * p(y_j))
    :return: M element array of log(sum(exp(log_probs[i,:])))
    """
    log_c = 1.0 - np.max(log_probs, axis=1)
    return np.log(np.sum(np.exp(log_probs + log_c[:, np.newaxis]), axis=1)) - log_c


def test_sum_log_probs():
    """
    Test sum_log_probs()
    """

    def _run_test(probs):
        t1 = sum_log_probs(np.log(probs))
        c1 = np.log(np.sum(probs, axis=1))
        for i in range(probs.shape[0]):
            assert np.allclose(t1[i], c1[i]), "Expected %s, got %s" % (c1, t1)

    probs1 = np.array([[0.1, 0.2, 0.3],
                      [0.2, 0.3, 0.4]])
    probs2 = np.random.rand(100, 10)
    _run_test(probs1)
    _run_test(probs2)
    print("sum_log_probs() passed")


def normalize_log_probs(log_probs):
    """
    Normalize the log probabilities.
    """
    return np.exp(log_probs - sum_log_probs(log_probs)[:, np.newaxis])


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


def demo(n_comps_real, n_comps_fit, spread, plot=True, n_pts=1000):
    """
    Demonstrate EM for fitting mixtures of 1-D Gaussians.
    """
    # Generate a random Gaussian mixture model
    model = GaussianMixtureModel.from_random(n_comps_real, spread)
    print("True model: %s" % model)
    data = model.sample(n_pts)

    # Fit a Gaussian mixture model to the data
    model_fit = GaussianMixtureModel.from_data(data, n_comps_fit, plot=plot, max_iter = 1000)
    if plot:
        fig, ax = plt.subplots(2,sharex=True)
        hist = Histogram(data, 50)
        hist.plot(ax[0])
        plot_dist(ax[0], model, n_pts=n_pts, label='True')
        plot_dist(ax[0], model_fit, n_pts=n_pts, label='Fit')
        plot_classification(ax[1], data, model.classify,colors=model_fit.colors)
        plt.legend()
        plt.show()

if __name__ == "__main__":

    plt.style.use('dark_background')
    # test_sum_log_probs()
    n_comps_real, n_comps_fit, spread = parse_args()
    demo(n_comps_real, n_comps_fit, spread)
