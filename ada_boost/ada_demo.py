"""
Show each iteration of adaboost.
"""
import numpy as np
import matplotlib.pyplot as plt
from perceptron import DecisionStump
from make_data import make_bump, make_spiral_data, make_minimal_data, make_checker_data
import logging
from plotting import plot_dataset, plot_classifier, get_plot_size
import argparse


def weights_to_sizes(weights, n_pts=0):
    s= 100 * weights / np.max(weights)
    if n_pts < 50:
        s=s*4
    return s


def _draw_lr(ax, lr_model, *args, **kwargs):
    coeffs = lr_model._weights
    intercept = lr_model._bias
    if coeffs[1] == 0:
        p0 = np.array([-intercept / coeffs[0], 0])
        p1 = np.array([(-intercept - coeffs[1]) / coeffs[0], 1])
    else:
        p0 = np.array([0, -intercept / coeffs[1]])
        p1 = np.array([1, (-intercept - coeffs[0]) / coeffs[1]])

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], *args, **kwargs)


class AdaDemo(object):
    """
    At each iteration, add these plots to the current figure row:
        * In the left column shows the classifier so far, the data, and the misclassified points
        * In the right column shows the weighted data and the new classifier added to the ensemble.    
    """

    def __init__(self, n_side_pts=40,max_iter=100, n_rows=3, skip_interval=1, kind='bump', pausing=True):
        """
        :param n_side_pts: number of points on each side of the square
        :param n_rows: number of rows in the plot
        :param skip_interval: only plot every skip_interval iterations
        :param kind: 'bump', 'spiral', 'minimal', 'checker' (which dataset to use)
        """
        self._max_iter=max_iter
        self._pausing = pausing
        if kind == 'bump':
            self._points, labels = make_bump(n_side_pts, 0.15, 0.2, 0.0, noise_frac=0.03, separable=True)
        elif kind == 'spiral':
            self._points, labels, _ = make_spiral_data(n_side_pts, turns=2.0, ecc=1.0, margin=0.04, random=True)
        elif kind == 'minimal':
            self._points, labels = make_minimal_data()
        elif kind == 'checker':
            self._points, labels = make_checker_data(n_side_pts, clip_cols=1)
        else:
            raise ValueError("Unknown AdaDemo kind %s" % kind)

        self._point_marker_size, _,_ = get_plot_size(self._points.shape[0])

        # make labels -1, 1
        u_labs = np.unique(labels)
        if len(u_labs) != 2:
            raise ValueError("Need exactly two classes")

        self._labels = np.zeros(labels.shape)
        self._labels[labels == u_labs[0]] = -1
        self._labels[labels == u_labs[1]] = 1

        # Plot iteration if (iter +1) % skip_interval == 0
        self._skip_interval = skip_interval

        # self._points, self._labels = make_bump(n_side_pts, h=.25,w=.2,x_left=.0, noise_frac=.003)
        # self._points, self._labels,_ = make_spiral_data(n_side_pts, turns=2.0, ecc=1.0, margin=0.04, random=False)
        self._n_pts = self._points.shape[0]
        self._n_rows = n_rows
        plt.ion()
        self._fig, self._axs = plt.subplots(n_rows, 4)
        # turn off all subplots for now
        for ax in self._axs.ravel():
            ax.axis('off')
        self._axs = np.atleast_2d(self._axs)
        self._next_row = 0

        self._iter = 0
        self._weights = np.ones(self._n_pts) / self._n_pts
        self._error_rates = []  # error rate of hypotheses i under distribution weights[i]
        self._models = []
        self._alphas = []
        self._loss = []  # loss of ensemble at each iteration
        # self._pre_test()
        self._run()

    def eval_ensenble(self, points=None, sign=True):
        points = points if points is not None else self._points
        preds = np.zeros(points.shape[0])
        for model, alpha in zip(self._models, self._alphas):
            single_predictions = model.predict(points)
            preds += alpha * single_predictions

        return np.sign(preds) if sign else preds

    def _plot_ensemble(self, row, iter):
        """
        Plot the ensemble of models so far, the data, and the misclassified points.
        """
        if iter % self._skip_interval != 0:
            return
        ax = self._axs[row, 3]
        ax.axis('on')
        ax.clear()
        y_hat = self.eval_ensenble()
        plot_classifier(ax, self._points, self._labels, model=y_hat, boundary=None)
        self._plot_boundary(ax)
        accuracy = np.mean(y_hat == self._labels)
        if row == 0:
            ax.set_title('Ensemble accuracy')

        ax.set_ylabel('%.3f' % (accuracy,))
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.yaxis.set_visible(True)

    def _plot_weights(self, weights, row, iter):

        if iter % self._skip_interval != 0:
            return
        # draw weights as circles around each sample
        ax = self._axs[row, 0]
        ax.axis('on')
        ax.clear()
        plot_dataset(ax, self._points, self._labels, markersize=self._point_marker_size)

        sizes = weights_to_sizes(weights, self._n_pts)
        ax.scatter(self._points[:, 0], self._points[:, 1], s=sizes, c='black', alpha=.5)
        if row == 0:
            ax.set_title('Weights')

        ax.set_ylabel('Iteration %i' % iter)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.yaxis.set_visible(True)

        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.xaxis.set_visible(False)
        # plot the weight distribution
        ax = self._axs[row, 1]
        ax.axis('on')
        ax.clear()
        ax.plot(sorted(weights))
        ax.tick_params(axis='both', which='both', labelsize=6)        
        if row == 0:
            ax.set_title('Weight dist.'.format(iter))
        # set tick label font size to 5
        

    def _plot_weak(self, new_model, w_loss, row, iter):

        if iter % self._skip_interval != 0:
            return
        ax = self._axs[row, 2]
        ax.axis('on')
        ax.clear()
        
        plot_classifier(ax, self._points, self._labels, model=new_model, boundary=None)

        new_model.plot(ax)
        if row == 0:
            ax.set_title('Weighted loss')
        ax.set_ylabel('%.3f' % (w_loss,))
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.yaxis.set_visible(True)

    def _plot_boundary(self, ax):
        """
        Plot the decision boundary of the current ensemble.
        """
        n_pts = 150
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.linspace(x_min, x_max, n_pts)
        y = np.linspace(y_min, y_max, n_pts)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T
        preds = self.eval_ensenble(points, sign=True)
        preds = preds.reshape(n_pts, n_pts)
        # ax.contourf(xx, yy, preds, cmap=plt.cm.RdBu, alpha=0.8)
        # show predictions as an image, with the color indicating the prediction
        image = ax.imshow(preds, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='RdBu', alpha=.5)

        # plt.colorbar(mappable=image, cax=None, ax=ax)
        image.set_zorder(-1)
        ax.set_aspect('equal')


    def _run(self):

        # plot dataset in it's own window before running
        # plt.figure()
        # self._plot_dataset(plt.gca())
        # plt.show()
        # plt.waitforbuttonpress()

        while True:
            # show weights first
            self._plot_weights(self._weights, self._next_row, self._iter)

            # logging.info("Iteration %i training w/weights:  %s" % (self._iter, self._weights))
            new_model = DecisionStump()  #
            new_model.fit(self._points, self._labels, sample_weight=self._weights)

            new_predictions = new_model.predict(self._points)
            misclassified = new_predictions != self._labels
            weighted_loss = np.sum(self._weights[misclassified])

            # stop if weighted loss is too high (> .5)
            if weighted_loss > .5:
                logging.info("Stopping, weighted loss too high: %.3f" % weighted_loss)
                break

            # Show new model, its weighted loss.

            self._plot_weak(new_model, weighted_loss, self._next_row, self._iter)
            # calculate alpha
            alpha = np.log((1 - weighted_loss) / weighted_loss)

            # update weights
            weights2 = np.exp(np.log(self._weights) + alpha * misclassified * (self._weights > 0))
            self._weights *= np.exp(alpha * misclassified)
            self._weights /= np.sum(self._weights)

            # update ensemble & evaluate
            self._models.append(new_model)
            self._alphas.append(alpha)
            self._error_rates.append(weighted_loss)
            new_acc = self._get_ensemble_acc()
            self._loss.append(1.0-new_acc)
            logging.info('Iteration {}: w_loss = {}, loss = {}, alpha = {}\n'.format(
                self._iter, weighted_loss, self._loss[-1], alpha))

            # show ensemble so far
            self._plot_ensemble(self._next_row, self._iter)
            if new_acc == 1.0:
                logging.info("Perfect accuracy achieved, stopping.")
                break

            if self._iter % self._skip_interval == 0:
                plt.show()
                if self._pausing:
                    plt.waitforbuttonpress()
                else:
                    plt.pause(.1)  # give the user a chance to see the plot
                self._next_row += 1
                if self._next_row >= self._n_rows:
                    # start a new figure?
                    # self._fig, self._axs = plt.subplots(self._n_rows, 3)
                    # self._axs = np.atleast_2d(self._axs)
                    self._next_row = 0
            self._iter += 1
            if self._iter >= self._max_iter:
                logging.info("Max iterations reached, stopping.")
                break

        # plot final decision boundary & hit/misses
        fig, ax = plt.subplots()
        y_hat = self.eval_ensenble()
        plot_classifier(ax, self._points, self._labels, model=y_hat, boundary=None)
        self._plot_boundary(ax)
        accuracy = np.mean(y_hat == self._labels)
        ax.set_title('Ensemble accuracy\n%% %.3f' % (100 * accuracy,))
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        #plt.tight_layout()
        plt.show()
        plt.waitforbuttonpress()

    def _get_ensemble_acc(self):
        predictions = self.eval_ensenble()
        accuracy = np.mean(predictions == self._labels)
        return accuracy


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', '-k', default='bump', help='bump, spiral, minimal, checker')
    parser.add_argument('--max_iter', '-x', default=100, type=int, help='Maximum number of weak learners')
    parser.add_argument('--n_side_pts', '-n', default=25, type=int,
                        help='Use a square grid of points w/ this many points on each side')
    parser.add_argument('--n_rows', '-r', default=3, type=int, help='Number of rows in the plot')
    parser.add_argument('--fast', '-f', action='store_true', help="Don't wait for user input between iterations")
    parser.add_argument('--skip_interval', '-s', default=1, type=int, help='Only plot every skip_interval iterations')

    args = parser.parse_args()

    AdaDemo(n_side_pts=args.n_side_pts,
            max_iter=args.max_iter,
            n_rows=args.n_rows,
            skip_interval=args.skip_interval,
            kind=args.kind, pausing=not args.fast)
    plt.show()
