"""
Demo of RANSAC algorithm, fitting a line to a set of points with outliers.
"""
import numpy as np
import matplotlib.pyplot as plt

from util_line import make_line_data, plot_line, point_line_distances, fit_line
from ransac import RansacModel, RansacDataFeatures, solve_ransac


class RansacLineData(RansacDataFeatures):
    """
    A dataset of 2d points for RANSAC, with a line model.
    """

    def __init__(self, data):
        """
        :param data: a Nx2 array of points
        """
        super().__init__(data, 2)

    def _extract_features(self):
        """
        Features for fitting a line to points are the points themselves.
        """
        self.n_features = self._dataset.shape[0]

    def get_min_sample_inds(self):
        # TODO: ensure minimum pairwise distance between sample points
        return np.random.choice(self.n_features, self.min_features, replace=False)

    def get_features(self, indices=None):
        """
        Get the features extracted from the data (used by RansacModel._fit).
        :param indices: array of indices of features to return, must be set if mask is None
        :return: list of features (i.e. self._features[...])
        """
        if indices is not None:
            return self._dataset[indices]
        else:
            return self._dataset

    def plot(self, ax=None, inlier_mask=None):
        """
        Plot the data points, coloring inliers and outliers differently.
        :param inlier_mask: boolean array indicating which points are inliers\
        """
        ax = ax if ax is not None else plt.gca()
        point_size = 10

        # Plot all points
        if inlier_mask is None:
            ax.scatter(self._dataset[:, 0], self._dataset[:, 1], c='k', label='all points', s=point_size)
        else:
            outliers = np.logical_not(inlier_mask)
            ax.scatter(self._dataset[inlier_mask, 0], self._dataset[inlier_mask, 1], c='b', label='inliers', s=point_size)
            ax.scatter(self._dataset[outliers, 0], self._dataset[outliers, 1], c='r', label='outliers', s=point_size)

        return ax
    
    def get_n_features(self):
        """
        Get the number of features extracted from the data.
        """
        return self.n_features


class RansacLine(RansacModel):
    """
    A 2d line estimator for RANSAC robust to outliers.
    """

    def __init__(self, data, inlier_threshold, training_inds, iter=None):
        """
        Set model params a, b, c of the line a*x + b*y + c = 0.
        """
        super().__init__(data, inlier_threshold, training_inds, iter)
        self._fig, self._ax = None, None

    def __str__(self):
        return "Line: %.3f*x + %.3f*y + %.3f = 0" % tuple(self._model_params)

    def _fit(self):
        """
        Fit a line to the set of points indicated by self.training_inds.
            - if N=2, the line will be through the points exactly.
            - otherwise, the line will be the least squares fit.
        """
        self._model_params = fit_line(self.data.get_features(indices=self.training_inds))

    def _get_inliers(self):
        distances = point_line_distances(self.data.get_features(), *self._model_params)
        return distances < self.thresh

    @staticmethod
    def _animation_setup():
        """
        Set up the animation for plotting the model.
        """
        RansacModel._FIG, RansacModel._AXES = plt.subplots(1, 2)

    def plot_iteration(self, data, best_so_far, is_final=False, max_iter=None):
        """
        2 plots, left is current iteration (self), with all points plotted, minimum sample, inliers/outliers distinguished 
        & fit line.  Right is the same for the best iteration's model so far.
        :param data: the RansacData object
        :param best_so_far: the best model so far (RansacModel)
        :param is_final: whether this is the final plot
        """

        if RansacModel._AXES is None:
            RansacLine._animation_setup()
        axes = RansacModel._AXES
        points = data.get_features()

        if is_final:
            title = 'Final RANSAC model after %i iterations\niter %i had %i inliers (%.2f %%)' % \
                (self.iter,
                 best_so_far.iter+1,
                 np.sum(best_so_far.inlier_mask),
                 100*np.mean(best_so_far.inlier_mask))

            fig, ax = plt.subplots(1)

            # Final plot has data w/ inliers & sample from best model
            data.plot(ax, inlier_mask=best_so_far.inlier_mask)

            # the sample used to find the inliers
            sample_pts = points[best_so_far.training_inds]
            ax.scatter(*sample_pts.T, s=50,
                       label='sample', facecolors='none', edgecolors='b', linewidth=1)

            # the model fitting the sample
            plot_line(*best_so_far.get_params(), ax=ax,
                      plt_args=['b-'], plt_kwargs={'label': 'best model, iter %i' % (best_so_far.iter+1)})

            # the model fitting the consensus set (inliers)
            plot_line(*self._model_params, ax=ax,
                      plt_args=['g-'], plt_kwargs={'label': 'final model'})

            # a least-squares fit to all points, as a baseline
            plot_line(*fit_line(points), ax=ax,
                      plt_args=['k--'], plt_kwargs={'label': 'LS fit, all data'})

            ax.legend()
            ax.set_title(title)
        else:
            title = 'iteration %i found %i inliers' % \
                    (self.iter+1, np.sum(self.inlier_mask))

            title_best = 'best iteration (%i): %i inliers' % (best_so_far.iter+1, np.sum(best_so_far.inlier_mask))

            # Clear the axes
            for a in axes:
                a.clear()

            def _plot(ax, plot_model, line_label, plot_str='b-'):
                """
                Plot the points, (Best) sample features, model (line), inliers/outliers.
                """
                data.plot(ax, inlier_mask=plot_model.inlier_mask)

                ax.scatter(*np.array(points[plot_model.training_inds]).T, s=50,
                           label='model features', facecolors='none', edgecolors='b', linewidth=1)
                plot_line(*plot_model.get_params(), ax=ax,
                          plt_args=[plot_str], plt_kwargs={'label': line_label})

            _plot(axes[0], self, '')
            _plot(axes[1], best_so_far, 'fit line')
            axes[0].set_title(title)
            axes[1].set_title(title_best)
            axes[1].legend()
            plt.suptitle("RANSAC Line Fitting:  iter %i of %i" % (self.iter+1, max_iter,))
        plt.draw()
