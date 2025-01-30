import numpy as np
import cv2
import pprint
import matplotlib.pyplot as plt
from image_util import TestImage
from tune_corner_detection import plot_trial

MARGIN = 10  # reject corners within this distance of the image border


def tune_similarity_metric(data1, data2, transf, plot=False):
    """
    Create a dataset, image pair. find corners, extract histograms, and create pairwise similarity matrix.
    Show a grid showing up to the first 10 of those corner windows on the top row, and below each, the
           highest & lowest N matching corners from the other image in descending order.
    :param data1: a tuple (TestImage, corners) for the first image (and detected corners)
    :param data2: a tuple (TestImage, corners) for the second image (and detected corners

    """
    test_img1, det_corners1 = data1
    test_img2, det_corners2 = data2

    true_corners1_in_img2 = test_img1.filter_pts_in_bbox(transf.apply(corners1))

    window_size = 21  # size of the window around each corner to extract a descriptor (pixels)

    # Detect corners and extract descriptors for each corner
    hist1 = np.array([test_img1.get_patch_descriptor(corner_center, window_size)
                      for corner_center in det_corners1])
    hist2 = np.array([test_img2.get_patch_descriptor(corner_center, window_size)
                      for corner_center in det_corners2])

    # Compute the similarity between each pair of descriptors
    similarity = np.array([[TestImage.compare_descriptors(hist1[i], hist2[j])
                            for j in range(len(hist2))] for i in range(len(hist1))])

    # show the first few corners of image1 and (up to) their top 8 matches in image2, in descending order of similarity
    # Next to each image show the histogram.
    palette = test_img1.palette.astype(np.float32) / 255.

    # in this window, plot the two images, detected corners & "true" corners (in image2):
    fig, ax = plt.subplots(1, 2)
    plot_trial(ax[0], test_img1, corners1, None, "image 1,\ndetected corners (dots)")
    plot_trial(ax[1], test_img2, corners2, true_corners1_in_img2,
               "image 2,\ndetected corners (dots),\nimage 1 corners, transferred (+)")


    # in this, show the matches:

    def _plot_hist(ax, hist):
        """
        Bar graph using the color palette.
        :param hist:  The array of histogram counts (normalized).
        """
        ax.bar(range(len(hist)), hist, color=palette, width=4)
        ax.axis('off')

    def _plot_patch(ax, patch, score=None):
        """
        Show a patch with a score (if given)
        """
        ax.imshow((patch))
        ax.axis('off')
        if score is not None:
            ax.text(0, window_size, '%.2f' % score, color='k', fontsize=10, va='top')

    if not plot:
        return

    # Number of examples to show (set to 1 to see the best and worst, etc)
    n_corner_examples = 6  # columns
    n_match_examples = 4
    n_worst_examples = 2

    n_cols = np.min([n_corner_examples, len(corners1)]) * 2  # for image and histogram
    n_rows = 1 + n_worst_examples + n_match_examples  # for best and worst matches
    fig, ax = plt.subplots(n_rows, n_cols)

    corners_to_plot = np.random.choice(len(corners1), n_corner_examples, replace=False)
    # print('Showing corners:', corners_to_plot)

    for i, corner_i in enumerate(corners_to_plot):

        # show the corner at the top
        window = test_img1.get_patch(corners1[corner_i], window_size, which='rgb')
        _plot_patch(ax[0, i*2], window)
        _plot_hist(ax[0, i*2+1], hist1[corner_i])

        # show the best
        for j, idx in enumerate(np.argsort(similarity[corner_i])[:n_match_examples]):
            window = test_img2.get_patch( corners2[idx], window_size, which='rgb')
            _plot_patch(ax[j+1, i*2], window, similarity[corner_i][idx])
            _plot_hist(ax[j+1, i*2+1], hist2[idx])

        # show the worst
        for j, idx in enumerate(np.argsort(similarity[corner_i])[-n_worst_examples:]):
            window = test_img2.get_patch(corners2[idx], window_size, which='rgb')
            _plot_patch(ax[j+1+n_match_examples, i*2], window, similarity[corner_i][idx])
            _plot_hist(ax[j+1+n_match_examples, i*2+1], hist2[idx])

    # Annotate plots window with separation lines under first row and between best/worst rows.
    # get the y-coordinates between the first and second row in figure coordinates.
    _, y1 = ax[1, 0].transAxes.transform([0, 1.10])
    _, y2 = ax[n_match_examples, 0].transAxes.transform([0, -.27])
    y1 = fig.transFigure.inverted().transform([0, y1])[1]
    y2 = fig.transFigure.inverted().transform([0, y2])[1]

    # x limits are 5% and 95% of the figure width
    x1 = 0.12
    x2 = 0.90
    # annotate

    ax[1, 0].annotate('', xy=(x1, y1), xytext=(
        x2, y1), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))
    y_offset = -0.002
    ax[n_match_examples, 0].annotate('', xy=(x1, y2+y_offset), xytext=(
        x2, y2+y_offset), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))

    # Add titles to the three row sections (off to the left)
    ax[0, 0].text(-0.1, 0.5, 'image1\ncorners', fontsize=12, color='r',
                  ha='right', va='center', transform=ax[0, 0].transAxes)
    ax[1, 0].text(-0.1, 0.5, 'best %i\nmatches\nin image2' % (n_match_examples,), fontsize=10,
                  color='r', ha='right', va='center', transform=ax[1, 0].transAxes)
    ax[n_match_examples+1, 0].text(-0.1, 0.35, 'worst %i\nmatches\nin image2' % (n_worst_examples,), fontsize=10,
                                   color='r', ha='right', va='center', transform=ax[n_match_examples+1, 0].transAxes)

    # remove most space between subplots
    plt.subplots_adjust(wspace=.1, hspace=.4)
    plt.suptitle('corner matcher: distances scores\n(candidate matches for %i corners detected in image1)'% (n_corner_examples))

    # show and return
    

if __name__ == "__main__":
    #plt.ion()  # do all windows on startup

    img_size = 400, 400
    noise_frac = 0.1

    params = dict(blockSize=8,
                  ksize=25,
                  k=0.05)

    q_img1 = TestImage(img_size, margin_px=MARGIN)
    q_img2, transf = q_img1.transform(noise_frac=noise_frac, rand_transf_params='demo')

    corners1 = q_img1.find_corners(harris_kwargs=params)
    corners2 = q_img2.find_corners(harris_kwargs=params)

    tune_similarity_metric((q_img1, corners1),
                           (q_img2, corners2),
                           transf=transf, plot=True)

    #plt.pause(0)
    plt.show()
