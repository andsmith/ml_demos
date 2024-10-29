"""
Iterate through conrner detection parameters to find the best combination for synthetic test images,
Both images should have roughly the same number of corners (50-2/300?), but the transformed image many have
fewer if some locations were transformed out of the image bounds.

for every set of parameters to cv2.cornerHarris, param:
  for trial in [1, n_trials]
    * Create an image pair, detect corners in the first, corners in the second.
    * Let score_1 be the score for the corner detector on image 1 (see score_1 for details)
    * Transform the corners in the first image to coordinates in the second image (removing those out of bounds), call these the "true" corners.
    * Let score_2 be the score for image 2, calculated from comparing the detected corners to the "true" corners (see score_2 for details)
    * score = score_1 * score_2

    (all scores in [0, 1] where higher is better)
    
The tuning algorithm shows grids of these scores and returns the parameters with the highest score.

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from image_util import TestImage
from greedy_assign import greedy_assign
from multiprocessing import Pool, cpu_count
import logging
from scipy import interpolate


def plot_trial(ax, img1, corners, true_corners=None, title=None, score=None):
    """
    Plot the image, detected corners, and transformed corners (if given) on the given axis.
    """
    img1.plot(ax, which='rgb')
    if corners is not None:
        ax.plot(corners[:, 0], corners[:, 1], 'ro')
    if true_corners is not None:
        ax.plot(true_corners[:, 0], true_corners[:, 1], 'b+', markersize=10)
    ax.set_title(title)
    if score is not None:
        ax.set_xlabel('score:  %.5f' % score)
    # Turn of x and y ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


class CornerDetectionTrial(object):
    """
    Represent test for 1 set of detection params (1 or more random trials).
    """
    _SCORE_1_DEF = [(0, 0),
                    (19, .1),
                    (20, .1),  # Score of corner detector on image 1 is defined by
                    (40, 1),  # the function interpolated around these values,
                    (100, 1),  # (corner count, score)
                    (150, 0.5),
                    (200, .1),
                    (400, 0),
                    (401, 0)]
    _SCORE_FN = interpolate.interp1d(*zip(*_SCORE_1_DEF), fill_value='extrapolate', kind='linear')

    def __init__(self, t_i, img_size, params, n_reps, noise_frac, margin=10, kind='double', plot=False):
        """
        (set up, but don't run trial.)

        Will test params for these kinds of trials:
            * 'single':  Scoring the detector on the first (untransformed) image.
            * 'double':  Scoring the detector on both images.

        :param t_i: index of this trial
        :param img_size: tuple of image dimensions (width, height)
        :param params: dict of corner detection parameters (blockSize, ksize, k)
        :param n_reps: number of trials to run for this set of params
        :param noise_frac: fraction of image pixels to randomly change in creating image 2
        :param margin: (param to TestImage.find_corners) margin around image edges to ignore corners
        :param kind: 'single' or 'double' (see above)
        :param plot: whether to plot the results of this trial (for debugging)
        """
        if kind not in ['single', 'double']:
            raise ValueError('kind must be "single" or "double"')
        self.kind = kind
        self.i = t_i
        self.img_size = img_size
        self.noise_frac = noise_frac
        self.params = params
        self.n_reps = n_reps
        self.margin = margin
        self.plot = plot

        self._n_corners_1 = []
        self._n_corners_2 = []
        self._n_correct_corners = []

        self.mean_corners_1 = None
        self.mean_corners_2 = None
        self.mean_correct_corners = None

        self._scores = []
        self.score = None  # average of _scores

    def __str__(self):
        return 'CornerDetectionTrial:  kind=%s, params=%s, score=%.5f' % (self.kind, self.params, self)

    def _trial(self, t_index):
        """
        Run a single trial of corner detection w/current params on a pair of synthetic test images.
        If self._plot is True and this is trial 0:
            * for kind 'single' trials, show the image, detected corners, number of corners, and the score.
            * for kind 'double' trials, show the images, detected corners, transformed (image 1 -> 2) corners, number of corners in both images,
              number of "correct" corners in the second image, the 2 score terms, and the final score.
        :returns: score of this trial, number of corners detected in image 1, number of corners detected in image 2 (or None if kind='single')
        """
        # Create the first image and detect corners
        image1 = TestImage(size=self.img_size,margin_px = self.margin)
        corners1 = np.array(image1.find_corners(harris_kwargs=self.params))
        score1 = self._score_1(corners1)
        if self.kind == 'single':
            if self.plot and t_index == 0:
                fig, ax = plt.subplots(1, 1)
                title = 'Image 1, %i corners' % corners1.shape[0]
                plot_trial(ax, image1, corners1, title=title, score=score1)
                plt.show()
            return score1, len(corners1), None, None

        # Create the second image, detect corners, and transform the corners from the first image to the second.
        image2, transf = image1.transform(self.noise_frac,rand_transf_params='demo')
        corners2 = np.array(image2.find_corners(harris_kwargs=self.params))
        true_corners2 = image2.filter_pts_in_bbox(transf.apply(corners1))
        score2, n_correct = self._score_2(corners1, corners2, true_corners2)
        score = score1 * score2
        if self.plot and t_index == 0:
            fig, ax = plt.subplots(1, 2)
            title = 'Image 1, %i corners' % corners1.shape[0]
            plot_trial(ax[0], image1, corners1, title=title, score=score1)
            title = 'Image 2, %i corners, (%i correct)' % (corners2.shape[0], n_correct)
            plot_trial(ax[1], image2, corners2, true_corners2, title=title, score=score2)
            plt.suptitle('combined score:  %.5f' % score)
            plt.show()

        return score, len(corners1), len(corners2), n_correct

    def _score_1(self, corners1):
        """
        Calculate a score for the corner detector on image 1:
            Assume the corners are sensible (this can be visually verified), all we care is that a "reasonable" number of corners are detected.
            The shape of our score function will be piecewise linear.
        """
        n_corners = len(corners1)
        return self._SCORE_FN(n_corners)

    def _score_2(self, corners1, corners2, true_corners2, max_dist_px=6.0):
        """
        Calculate a score for image 2, calculated from comparing the detected corners to the "true" corners:
            * The "true" corners are the corners in image 1 transformed to the coordinates of image 2.
            * Detected corners in image 2 are assigned to true corners in a greedy fashion until distances exceeds max_dist_px.
            * The score is the accuracy of the detected corners in image 2:
                score = n_correct / n_detected

        :param corners1: corners detected in image 1 (Nx2 array)
        :param corners2: corners detected in image 2 (Nx2 array)
        :param transform_func: transformation function from coordinate system of image 1 to image 2
        :returns: score for image 2, n_correct detections 
        """
        if len(corners1) == 0 or len(corners2) == 0 or len(true_corners2) == 0:
            return 0,0

        # Calculate the distance between each pair of corners
        dists = np.linalg.norm(true_corners2[:, None] - corners2[None], axis=-1).T
        assigns, _, _ = greedy_assign(dists, max_dist_px)
        return len(assigns) / len(corners2), len(assigns)

    def eval(self):
        """
        Run corner detection on synthetic test images with the given parameters, n_reps times.
        """
        for i in range(self.n_reps):
            score, nc1, nc2, ncc = self._trial(i)
            self._n_corners_1.append(nc1)
            self._n_corners_2.append(nc2)
            self._n_correct_corners.append(ncc)
            self._scores.append(score)

        self.score = np.mean(self._scores)
        self.mean_corners_1 = np.mean(self._n_corners_1)
        self.mean_corners_2 = np.mean(self._n_corners_2) if self.kind == 'double' else None
        self.mean_correct_corners = np.mean(self._n_correct_corners) if self.kind == 'double' else None

        report = "trial %s complete (%i repetitions):  " % (self.params, self.n_reps) + \
            "\tmean score:  %.5f" % (self.score,) +\
            "\tmean corners in 1:  %i" % (self.mean_corners_1, )

        if self.mean_corners_2 is not None:
            report += "\tmean corners in 2:  %i" % (self.mean_corners_2, )
            report += "\tmean correct corners:  %i" % (self.mean_correct_corners, )
        print(report)   # no logging in subprocesses?
        return self


def _eval_trial(trial):
    # Map this function to a list of CornerDetectionTrial objects for parallel processing.
    return trial.eval()

def tune(img_size, blockSize_vals, kSize_vals, k_vals, noise_frac, n_trials_per_param=50, n_cores=0, kind='double', plot=False):
    """
    Iterate through corner detection parameters to find the best combination for synthetic test images.
    (params to cv2.cornerHarris)
    :param img_size: tuple of image dimensions (width, height)
    :param blockSize_vals: list of block sizes for corner detection
    :param kSize_vals: list of Sobel kernel sizes for corner detection
    :param k_vals: list of Harris corner detection parameter k
    :param n_trials_per_param: number of trials per parameter combination
    :param n_cores: number of cores to use for parallel processing (0 for all cores)
    """
    # Generate parameter combination list
    param_combos = list(product(blockSize_vals, kSize_vals, k_vals))

    def _get_kw_params(params):
        return dict(blockSize=params[0], ksize=params[1], k=params[2])
    param_combos = [_get_kw_params(params) for params in param_combos]
    trials = [CornerDetectionTrial(i, img_size, params, n_trials_per_param, noise_frac, kind=kind)
              for i, params in enumerate(param_combos)]

    # Run corner detection on each image with each parameter combination
    n_cores = cpu_count()+1 if n_cores == 0 else n_cores
    logging.info("Block sizes(%i):  %s" % (len(blockSize_vals), blockSize_vals))
    logging.info("kSizes(%i):  %s" % (len(kSize_vals), kSize_vals))
    logging.info("k values(%i):  %s" % (len(k_vals), k_vals))
    logging.info("About to run %i trials on %i cores..." % (len(trials), n_cores))
    if n_cores == 1:
        for trial in trials:
            trial.eval()
    else:
        with Pool(n_cores) as p:
            results = p.map(_eval_trial, trials)
            trials = results
    scores = np.array([trial.score for trial in trials],dtype=np.float64)

    # Find the parameter combination with the highest average number of corners detected
    n_bad= np.sum(np.isnan(scores))

    if n_bad > 0:
        print("WARNING:  %i failed trials." % n_bad)
        scores[~np.isnan(scores)] = 0  # from failed trials

    best_params = param_combos[np.argmax(scores)]
    best_score = np.max(scores)

    if plot:
        # Create a subplot for every blockSize value, plot a grid of scores for each kSize, k value pair tested w/ that blockSize.
        # Show the plot w/a color bar.
        if len(blockSize_vals) > 1:
            n_plots = len(blockSize_vals)
            n_rows = int(np.floor(np.sqrt(n_plots)))
            n_cols = int(np.ceil(n_plots / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols)
            axes = axes.flatten()
        else:
            n_plots, n_cols, n_rows = 1, 1, 1
            fig, axes = plt.subplots(1, 1)
            axes = [axes]

        # collect each score matrix for the different block sizes
        score_blocks = [np.zeros((len(kSize_vals), len(k_vals))) for _ in range(len(blockSize_vals))]
        for b_i, blockSize in enumerate(blockSize_vals):
            for i, trial in enumerate(trials):
                if trial.params['blockSize'] == blockSize:
                    kSize_i = kSize_vals.index(trial.params['ksize'])
                    k_i = k_vals.index(trial.params['k'])
                    score_blocks[b_i][kSize_i, k_i] = trial.score

        for i in range(n_plots):

            axes[i].set_title('blockSize = %d' % blockSize_vals[i])
            cax = axes[i].matshow(score_blocks[i], cmap='viridis')
            fig.colorbar(cax, ax=axes[i])

            # use at most 4 k_val ticks
            n_ticks = 3
            if len(k_vals) > (n_ticks-1):
                k_val_step = int(len(k_vals) / (n_ticks-1))
                axes[i].set_xticks(range(0, len(k_vals), k_val_step), ["%.3f" % k for k in k_vals[::k_val_step]])
            else:
                axes[i].set_xticks(range(len(k_vals)), ["%.3f" % k for k in k_vals])
            axes[i].set_yticks(range(len(kSize_vals)), kSize_vals)
            axes[i].tick_params(axis='x', top=False, labeltop=False, bottom=True, labelbottom=True)

            # label horizontal axis of bottom row only
            if i >= n_plots - n_cols:
                axes[i].set_xlabel('k')
            # label vertical axis of left column only
            if i % n_cols == 0:
                axes[i].set_ylabel('kSize')

        # hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Harris Corner Detection Parameter Tuning:  type %s (noise=%.3f)" % (kind, noise_frac))
        plt.show()

    return best_params, best_score


def optimize(noise_frac, kind, plot=True):

    img_size = 500, 500

    if True:
        # full range of parameters to test:
        blockSize_vals = [ 2,4,6,8]
        kSize_vals = [3, 7, 11, 13, 15, 17, 19, 21, 23, 25]
        k_vals = np.arange(0.03, 0.1, .005).tolist()
        n_reps = 50
    else:
        # smaller range of parameters to test, for debugging
        blockSize_vals = [2, 3, 4, 5, 6, 7]
        kSize_vals = [3, 5, 7]
        k_vals = [.04, .05, .06]
        n_reps = 10

    best_params, score = tune(img_size, blockSize_vals, kSize_vals, k_vals,
                              noise_frac, n_trials_per_param=n_reps, n_cores=16, kind=kind, plot=plot)
    logging.info('\n\nBest parameters (score=%.5f):\n%s\n\n' % (score,  best_params))
    return best_params, score


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    noise_frac = 0.1

    optimize(noise_frac=noise_frac, kind='double', plot=True)
