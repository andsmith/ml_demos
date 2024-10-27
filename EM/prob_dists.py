"""
Demonstrate EM For fitting mixtures of 1-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plt
from util import Histogram, plot_classification, plot_dist, normalize_log_probs, sum_log_probs
from em_fit import ProbDist


class GaussianDist(ProbDist):
    MIN_SD = 0.0001

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd if sd > self.MIN_SD else self.MIN_SD

    def __str__(self):
        return "N(%.2f, %.2f)" % (self.mean, self.sd)

    def pdf(self, x):
        return 1.0 / (self.sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - self.mean) / self.sd) ** 2)

    def log_pdf(self, x):
        return -0.5 * np.log(2 * np.pi) - np.log(self.sd) - 0.5 * ((x - self.mean) / self.sd) ** 2

    def sample(self, n):
        return np.random.normal(self.mean, self.sd, n)

    @classmethod
    def from_random(cls, spread):
        mean = np.random.randn() * spread
        sd = np.maximum(np.abs(np.random.rand(1)), .05)
        return GaussianDist(mean, sd)

    @staticmethod
    def from_data(data, weights):
        """
        Estimate the mean and standard deviation of a Gaussian from data.
        """
        mean = np.average(data, weights=weights)
        sd = np.sqrt(np.average((data - mean) ** 2, weights=weights))
        return GaussianDist(mean, sd)


class LaplaceDist(ProbDist):
    MIN_SD = 0.0001

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd if sd > self.MIN_SD else self.MIN_SD

    def __str__(self):
        return "L(%.2f, %.2f)" % (self.mean, self.sd)

    def pdf(self, x):
        return 1.0 / (2 * self.sd) * np.exp(-np.abs(x - self.mean) / self.sd)

    def log_pdf(self, x):
        return -np.log(2 * self.sd) - np.abs(x - self.mean) / self.sd

    def sample(self, n):
        return np.random.laplace(self.mean, self.sd, n)

    @classmethod
    def from_random(cls, spread):
        mean = np.random.randn() * spread
        sd = np.maximum(np.abs(np.random.rand(1)), .05)
        return LaplaceDist(mean, sd)

    @staticmethod
    def from_data(data, weights):
        """
        Estimate the mean and standard deviation of a Laplace distribution from data.
        """
        mean = np.average(data, weights=weights)
        sd = np.average(np.abs(data - mean), weights=weights)
        return LaplaceDist(mean, sd)
    
DISTRIBUTIONS = [GaussianDist, LaplaceDist]

def test_plot_dist():
    """
    Test plot_dist by plotting a Gaussian and a Laplace distribution.
    """
    n_pts=5000
    x_lim=7
    fig, ax = plt.subplots(1,2)
    x = np.linspace(-x_lim, x_lim, n_pts)
    ax[0].set_xlim(-x_lim, x_lim)
    for i, dist in enumerate(DISTRIBUTIONS):
        plot_dist(ax[0], dist(-0, 2), label=str(dist.__name__))
    ax[0].legend()
    pts = [dist(-0, 2).sample(n_pts) for dist in DISTRIBUTIONS]
    for points in pts:
        plt.plot(points, np.random.rand(n_pts), '.',markersize=4, alpha=0.3)
    ax[1].set_xlim(-x_lim, x_lim)
    plt.show()


if __name__ == "__main__":
    test_plot_dist()