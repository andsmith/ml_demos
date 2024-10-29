"""
Create and evaluate a single trial (parameter set).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from image_util import TestImage, in_bbox
import logging

from tune_corner_detection import CornerDetectionTrial


def test_corner_detection_trial(noise_frac, n_trials=10, params=None, kind='single', plot=False):
    """
    Test the CornerDetectionTrial class.
    """
    img_size = 500,500
    _params = dict(blockSize=2,
                   ksize=3,
                   k=0.04)
    
    if params is not None:
        _params.update(params)

    trial = CornerDetectionTrial(0, img_size, _params, n_trials, noise_frac, kind=kind, plot=plot)
    trial.eval()

    logging.info('CornerDetectionTrial (%s) found:' % kind)
    logging.info('\tn_trials:  %s' % (trial.n_reps,))
    logging.info('\tscore:  %s' % (trial.score,))
    logging.info('\tmean_corners_1:  %s' % (trial.mean_corners_1,))
    if kind == 'double':
        logging.info("\timage 2 noise fraction:  %.2f" % trial.noise_frac)
        logging.info('\tmean_corners_2:  %s' % (trial.mean_corners_2,))
        assert trial.score > 0.01, \
            'CornerDetectionTrial score_2 score is very low.  Check the plot.'

    assert 0 <= trial.score <= 1, 'CornerDetectionTrial score out of expected range:  %.2f' % trial.score
    logging.info('CornerDetectionTrial (%s) test passed.' % kind)


def _plot_corner_detector(noise_frac=0.0):
    """
    Make an image pair, detect corners in both, plot corners in both, translate corners from image 1 to image 2 and plot those over image 2.
    """
    size = 400, 400
    margin=15
    
    params = dict(blockSize=2,
                  ksize=9,
                  k=0.06)

    img1 = TestImage(size)
    img2, transf = img1.transform(noise_frac=noise_frac,rand_transf_params='test_detection_tuner')
    corners1_xy = img1.find_corners(harris_kwargs=params,margin=margin)
    corners2_xy = img2.find_corners(harris_kwargs=params,margin=margin)
    true_corners2_xy = img2.filter_pts_in_bbox(transf.apply(corners1_xy))


    fig, ax = plt.subplots(1, 2)
    img1.plot(ax[0], which = 'rgb')
    ax[0].plot(corners1_xy[:, 0], corners1_xy[:, 1], 'ro')
    ax[0].legend(['detected corners'])
    img2.plot(ax[1], which = 'rgb')
    ax[1].plot(corners2_xy[:, 0], corners2_xy[:, 1], 'ro')
    ax[1].plot(true_corners2_xy[:, 0], true_corners2_xy[:, 1], 'b+', markersize=10)
    ax[1].legend(['detected corners', 'transformed corners\nfrom image 1'])
    ax[0].set_title('Image 1')
    ax[1].set_title('Image 2')

    plt.show()

def _plot_detector_function():
    """
    Plot the corner detection score function.
    """
    n = np.arange(0, 400)
    scores = CornerDetectionTrial._SCORE_FN(n)
    fig, ax = plt.subplots()
    ax.plot(n, scores)
    ax.set_xlabel('n_corners')
    ax.set_ylabel('score')
    ax.set_title('Corner Detection Score Function\n')
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    size = 500, 500
    noise_frac = 0.10

    params = dict(blockSize=8,
                  ksize=23,
                  k=0.045)
    #_plot_corner_detector(noise_frac)
    #_plot_detector_function()
    test_corner_detection_trial(noise_frac=noise_frac, n_trials=3, params=params, kind='single',plot=True)
    test_corner_detection_trial(noise_frac=noise_frac, n_trials=3, params=params, kind='double',plot=True)
    

    logging.info('All tests passed.')
