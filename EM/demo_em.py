"""
Demonstrate EM For fitting mixtures of 1-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
from util import Histogram, plot_classification, plot_dist
from prob_dists import GaussianDist
from em_fit import MixtureModel




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
    parser.add_argument('-u', "--uniform",action='store_true', help="Initialize data w/o clusters.")
    args = parser.parse_args()
    return args.n_real, args.n_fit, args.iter, args.spread, args.n_points, args.animate_frame, args.uniform



def demo(n_comps_real, n_comps_fit, max_iter, spread, n_pts=1000, ainmate_interval=1, uniform_init = False):
    """
    Demonstrate EM for fitting mixtures of 1-D Gaussians.
    """
    # Generate a random Gaussian mixture model
    priors = np.random.rand(n_comps_real)
    priors /= np.sum(priors)
    if not uniform_init:
        model = MixtureModel([GaussianDist.from_random(spread) for _ in range(n_comps_real)],
                            priors)
        data = model.sample(n_pts)
    else:
        data = np.random.rand(n_pts) 

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
    model_fit = MixtureModel.from_data(GaussianDist,data, n_comps_fit, animate_interval=ainmate_interval, max_iter=max_iter)

    # show true and fit models
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].clear()
    # plot mixture distrubtions:
    hist.plot(ax[0], label=None)
    def_blue = np.array((31, 119, 180))/255
    def_orange = np.array((255, 127, 14))/255
    if not uniform_init:
        plot_dist(ax[0], model, n_pts=n_pts, label='True', color=def_blue)
    else:
        ax[0].plot([-1,-1e-8, 0, 1, 1+1e-8, 2], [0,0,1, 1,0,0], color=def_blue, label='True')

    plot_dist(ax[0], model_fit, n_pts=n_pts, label='Fit', linestyle='--', color=def_orange)
    ax[0].legend()
    ax[0].set_xlabel(None)
    ax[0].set_title("True & EM-fit mixture distributions - p(x)")
    ax[1].clear()
    # plot true and fit components
    if not uniform_init:
        for i, component in enumerate(model.components):
            plot_dist(ax[1], component, n_pts=n_pts, weight=model.priors[i], color=def_blue)
        ax[1].set_title("True & EM-fit components' distributions - p(x|y)p(y)")

    else:
        ax[1].set_title("EM-fit components' distributions - p(x|y)p(y)")

    for i, component in enumerate(model_fit.components):
        plot_dist(ax[1], component, n_pts=n_pts, weight=model_fit.priors[i], linestyle='--', color=def_orange)
    plt.show()
    plt.pause(0)


if __name__ == "__main__":

    plt.style.use('dark_background')
    # test_sum_log_probs()
    n_comps_real, n_comps_fit, max_iter, spread, n_pts, animate_interval, unif = parse_args()
    demo(n_comps_real, n_comps_fit, max_iter, spread,
         n_pts, ainmate_interval=animate_interval, uniform_init=unif)
