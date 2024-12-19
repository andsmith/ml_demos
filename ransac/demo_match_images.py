"""
Demo for aligning two images using detected corners and RANSAC.
"""
from ransac import solve_ransac
from match_images import RansacImageData, RansacAffine, DOT_SIZE, NEON_GREEN
from image_util import TestImage
import matplotlib.pyplot as plt
import logging


def demo(**ransac_args):
    """
    Demonstrate RANSAC, finding a line among a set of 2d points.

    :param n_line_pts: number of points in the line
    :param n_outliers: number of outlier points
    :param ransac_args: additional parameters for solve_ransac()
    """
    # run algorithm

    matcher_threshold = 0.4
    size = 700, 700
    noise_level = 0.1
    args = dict(blockSize=8,
                ksize=23,
                k=0.045)

    # make data
    img1 = TestImage(size)
    img2, orig_transf = img1.transform(noise_level)
    data = (img1, img2)
    ransac_data = RansacImageData(data, harris_kwargs=args, matcher_threshold=matcher_threshold)

    # Plot initial data
    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.subplots(2, 1)
    ransac_data.plot_features(ax[1])
    fig.text(.077, .60, "Corner Detector", fontsize=10, color='black', rotation=90)
    ax[0].axis('off')
    # Ax[0].set_title("All detected corners & candidate matches\n(close to run RANSAC)")
    ax = fig.subplots(2, 2)
    img1.plot(ax[0][0], which='rgb')
    img2.plot(ax[0][1], which='rgb')
    ax[0][0].set_title("Image 1", fontsize=10)
    ax[0][1].set_title("Image 2", fontsize=10)
    ax[0][0].axis('off')
    ax[0][1].axis('off')
    ax[1][0].axis('off')
    ax[1][1].axis('off')

    # Add labels since axes are off

    fig.text(.077, .20, "Corner Matcher", fontsize=10, color='black', rotation=90)

    # plot corners
    ax[0][0].plot(ransac_data.corners_1[:, 0], ransac_data.corners_1[:, 1], '.', color=NEON_GREEN, markersize=DOT_SIZE)
    ax[0][1].plot(ransac_data.corners_2[:, 0], ransac_data.corners_2[:, 1], '.', color=NEON_GREEN, markersize=DOT_SIZE)
    plt.suptitle("Image alignment using RANSAC\n(click to start)")
    plt.waitforbuttonpress()

    # Run RANSAC
    result = solve_ransac(ransac_data, RansacAffine, **ransac_args)

    # print results
    print('\n\nRANSAC found the best solution on iteration %i / %i:' %
          (result['best'].iter, result['final'].iter))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.ion()
    ransac_args = dict(max_error=5.0,  # pixel distance defining inliers
                       max_iter=1000,
                       animate_pause_sec=0.6,  # 0 to pause between iterations, None to disable plotting
                       animate_interval=10,
                       )
    demo(**ransac_args)
