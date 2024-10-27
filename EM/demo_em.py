"""
Demonstrate EM For fitting mixtures of 1-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
from util import Histogram, plot_classification, plot_dist
from prob_dists import GaussianDist, LaplaceDist
from em_fit import MixtureModel



def parse_args():
    parser = ArgumentParser(description="Demonstrate EM for fitting mixtures of 1-D data using Gaussian and Laplacian components.")
    parser.add_argument('-g', "--n_gaussian", type=int, default=5, help="Number of Gaussian components generating data.")
    parser.add_argument('-l', "--n_laplacian", type=int, default=0, help="Number of Laplacian components generating data.")
    parser.add_argument('-gf', "--n_gauss_fit", type=int, default=5, help="Number of Gaussian components in the fit model")
    parser.add_argument('-lf', "--n_laplace_fit", type=int, default=0, help="Number of Laplacian components in the fit model")
    parser.add_argument('-i', "--iter", type=int, default=300, help="Maximum number of iterations to run")
    parser.add_argument('-s', "--spread", type=float, default=1.5, help="Spread of the true model")
    parser.add_argument('-p', "--n_points", type=int, default=2000, help="Number of data points to generate")
    parser.add_argument('-a', "--animate_frame", type=int, default=1, help="Animate every n-th frame (or 0 for no animation)")
    parser.add_argument('-u', "--uniform",action='store_true', help="Initialize data w/o clusters (ignore -g, -l, -s).")
    args = parser.parse_args()
    return args



def demo(cmd_args):
    """
    Demonstrate EM for fitting mixtures of 1-D Gaussians.
    :param n_comps_real: tuple(n_gaussian, n_laplacian), the number of components in the true model 
    :param n_comps_fit: tuple(n_gaussian, n_laplacian), the number of components in the fit model
    :param max_iter: the maximum number of iterations to run
    """
    ng_real, nl_real = (cmd_args.n_gaussian, cmd_args.n_laplacian)
    ng_fit, nl_fit = (cmd_args.n_gauss_fit, cmd_args.n_laplace_fit)
    max_iter = cmd_args.iter
    spread = cmd_args.spread
    n_pts = cmd_args.n_points
    ainmate_interval = cmd_args.animate_frame
    uniform_init = cmd_args.uniform
    
    n_real = ng_real + nl_real
    n_fit= ng_fit + nl_fit
    

    priors = np.random.rand(n_real)
    priors /= np.sum(priors)
    
    component_types = [GaussianDist] * ng_real + [LaplaceDist] * nl_real
    if not uniform_init:
        print("Initializing data from %i Gaussians and %i Laplacians." % (ng_real, nl_real))
        model = MixtureModel([component_type.from_random(spread) for component_type in component_types], priors)
        data = model.sample(n_pts)
    else:
        print("Initializing uniform random data.")
        data = np.random.rand(n_pts) 

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
    model_fit = MixtureModel.from_data(component_types,data, animate_interval=ainmate_interval, max_iter=max_iter)

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

    # print priors for true and fit models (for both kinds of components)
    # for each component type, print the priors in descending order
    for component_type in [GaussianDist, LaplaceDist]:
        priors_true = [model.priors[i] for i, component in enumerate(model.components) if isinstance(component, component_type)]
        priors_fit = [model_fit.priors[i] for i, component in enumerate(model_fit.components) if isinstance(component, component_type)]
        print("True %s priors: %s" % (component_type.__name__, ", ".join(["%.3f"%(pr,) for pr in sorted(priors_true, reverse=True)])))
        print("Fit %s priors: %s" % (component_type.__name__, ",".join(["%.3f"%(pr,) for pr in sorted(priors_fit,reverse=True)])))
        


    plt.show()
    plt.pause(0)


if __name__ == "__main__":

    plt.style.use('dark_background')
    # test_sum_log_probs()
    demo(parse_args())
