import numpy as np


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
        bin_centers = self._bins[:-1] + self._bin_width / 2
        ax.bar(bin_centers, self._density, width=self._bin_width,
               fill=False, edgecolor='white', *args, **kwargs)
        ax.set_xlim(self._x_min, self._x_max)


def plot_classification(ax, points, labels, colors, *args, **kwargs):
    """
    Plot the classification of points by a classifier.

    :param ax: the axis to plot on
    :param points: Nx2 element array of points
    :param classifier: a function that takes a point and returns a class label
    :param colors: a list of colors to use for each class
    :param aspect: the aspect ratio of the plot
    :param args & kwargs: additional arguments to pass to ax.plot
    """
    default_marker_size = 10
    markersize = max(default_marker_size - 2*np.log10(points.shape[0]), 1)
    for label in set(labels):
        color = colors[label]
        pts = points[labels == label, :]
        ax.plot(pts[:, 0], pts[:, 1], '.',
                color=color, markersize=markersize,
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
    # import ipdb; ipdb.set_trace()
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, n_pts)
    y = dist.pdf(x) * weight
    ax.plot(x, y, *args, **kwargs)
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("p(x)")
    ax.set_xlabel("x")


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
