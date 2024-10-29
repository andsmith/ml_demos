"""
Class for representing synthetic images for testing homography estimation.
Uses Harris corner detection to find corners in the image, and extracts local feature descriptors (histograms)
for each corner.  The descriptors can be compared using a variety of metrics.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from make_test_imgs import draw_img, transform_img
from util_affine import in_bbox


class TestImage(object):
    """
    Class for synthetic image homography testing.

    The image coordinate system is (0,0) in the center, and the x/y scale is same.

    Generate a test image with a known transformation between it and a second image.
    Perform (Harris) corner detection tuned for this kind of image.
    Extract local feature descriptors (represented internally as histograms, somewhat invariant to this restricted set of images)
    Compare two descriptors to return a score.
    """

    def __init__(self, size, n_rects=10, n_circle_colors=30, n_rect_colors=3, margin_px=10):
        """ 
        :param size: the size of the image (width, height)
        :param n_rects: the number of rectangles to draw on top of the circles
        :param n_circle_colors: the number of colors to use for the circles
        :param n_rect_colors: the number of colors to use for the rectangles
        :param margin_px: Corners won't be detected this close to the border, and
            points transformed to a position on this margin will be considered out
            of bounds.
        """
        half_span_x = size[0] / 2. - .5  # middle pixel at 0,0 or if size is even, at +/- 0.5
        half_span_y = size[1] / 2. - .5
        self._origin_offset = np.array([half_span_x, half_span_y])
        self._bbox = {'x': [-half_span_x, half_span_x],
                      'y': [-half_span_y, half_span_y]}
        self._bbox_w_margin = {'x': [-half_span_x + margin_px, half_span_x - margin_px],
                               'y': [-half_span_y + margin_px, half_span_y - margin_px]}

        self.size = size
        self._margin_px = margin_px
        self.img, self.palette = draw_img(size, n_rects, n_circle_colors, n_rect_colors)
        self._init()

    def get_bbox(self):
        return self._bbox   

    def px_to_xy(self, px):
        """
        Convert pixel coordinates to xy coordinates.
        :param px: an Nx2 array of pixel coordinates
        :returns: an Nx2 array of xy coordinates
        """
        xy = (px - self._origin_offset)
        # print("px->xy: ", px, xy)
        return xy

    def xy_to_px(self, xy):
        """
        Convert xy coordinates to pixel coordinates.
        :param xy: an Nx2 array of xy coordinates
        :returns: an Nx2 array of pixel coordinates
        """
        px = xy + self._origin_offset
        # print("xy->px: ", xy, px)
        return px

    def _init(self):
        self.n_colors = self.palette.shape[0]
        self.rgb_img = self.palette[self.img].astype(self.palette.dtype)
        self.gray = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2GRAY)

    def transform(self, noise_frac, rand_transf_params='demo'):
        """
        Create a new TestImage by randomly transforming the current image.
        (bbox is not changed, so the image may be cropped if it moves out of the box)
        :param image: a 2d array of pixel values, each in [0, palette.shape[0])
        :param noise_frac: the fraction of pixels to randomly change
        :param rand_transf_params: what kind of random transformation to apply (see util_afine:from_random())
        """
        img2, transf = transform_img(self.img, noise_frac,
                                     max_color_ind=self.n_colors,
                                     rand_transf_params=rand_transf_params)
        r = TestImage(size=self.size)
        r.img = img2
        r.palette = self.palette
        r._init()
        return r, transf

    @staticmethod
    def compare_descriptors(hist1, hist2):
        # Symmetric Kullback-Leibler divergence:
        # return 0.5 * (np.sum(hist1 * np.log(hist1 / hist2)) + np.sum(hist2 * np.log(hist2 / hist1)))

        # Bhattacharyya distance:
        # return -np.log(np.sum(np.sqrt(hist1 * hist2)))

        # Chi-squared distance:
        return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2))

        # dot product of max-normalized histograms
        # hist1 /= np.max(hist1)
        # hist2 /= np.max(hist2)
        return -np.dot(hist1, hist2)

    def get_patch(self, xy_img, patch_size, which='index'):
        """
        Get a patch of the image around a point.
        :param xy_img tuple(float,float), xy-coordinate of the center of the patch (in image coords)
        :param patch_size: the size of the patch (square, will cover x,y +/- patch_size//2)
        returns: an array of the patch, either in rgb tuples or index values
        """
        x, y = self.xy_to_px(xy_img)
        if which not in ['index', 'rgb']:
            raise ValueError("which must be one of 'index' or 'rgb'")
        patch_size = patch_size // 2
        x_min, xmax = int(max([0, x-patch_size])), int(min([self.img.shape[1], x+patch_size]))
        y_min, ymax = int(max([0, y-patch_size])), int(min([self.img.shape[0], y+patch_size]))
        src = self.img if which == 'index' else self.rgb_img
        return src[y_min:ymax, x_min:xmax]

    def get_patch_descriptor(self, xy_img, patch_size, smoothing=.5):
        """
        Get the histogram of the patch around a point.
        :param xy_img tuple(float,float), xy-coordinate of the center of the patch (in image coords)
        :param patch_size: the size of the patch (square, will cover x,y +/- patch_size//2)
        :param smoothing: a small value to add to each bin to avoid dividing by zero in the histogram comparison
        """
        window = self.get_patch(xy_img, patch_size)
        hist = np.array([np.sum(window == i) for i in range(self.n_colors)], dtype=np.float32).reshape(-1)
        hist += smoothing
        hist /= np.sum(hist)
        return hist

    def find_corners(self, harris_kwargs=None):
        """
        Find corners in an image using the Harris corner detector.
        :param harris_kwargs: additional keyword arguments to pass to cv2.cornerHarris
            - blockSize: the size of the window to consider for each corner
            - ksize: the size of the Sobel kernel to use for the derivative (must be odd)
            - k: a free parameter in the Harris detector equation (increasing K decreases the number of corners)

        :returns: an Nx2 array of corner coordinates (in image coords, not pixels)
        """
        default_harris_kwargs = dict(blockSize=2,
                                     ksize=3,
                                     k=0.04)
        if harris_kwargs is not None:
            default_harris_kwargs.update(harris_kwargs)

        dst = cv2.cornerHarris(self.gray.astype(np.float32), **default_harris_kwargs)

        # Dilate to mark the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        # dst[dst > 0.01*dst.max()] = [0, 0, 255]

        # Get sub-pixel accuracy on the corners
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # remove spurious centroid in the center of the image
        centroids = centroids[1:]

        # Define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        try:
            corners = cv2.cornerSubPix(self.gray.astype(np.float32), np.float32(
                centroids), (5, 5), (-1, -1), criteria)

        except cv2.error:
            # print min/max x/y of centroids
            print("Error in cornerSubPix: ", centroids.min(axis=0), centroids.max(axis=0))
            print("returning centroids without subpixel refinement.")
            print("\tCentroid range: ", centroids.min(axis=0), centroids.max(axis=0))
            print("\tNumber of centroids found: ", len(centroids))
            corners = centroids

        # Filter out corners near the edge
        corner_coords = self.px_to_xy(corners)
        corners = self.filter_pts_in_bbox(corner_coords)
        return corners

    def filter_pts_in_bbox(self, pts_xy):
        valid = in_bbox(self._bbox_w_margin, pts_xy)
        return pts_xy[valid]

    def get_plot_extent(self):
        return [self._bbox['x'][0], self._bbox['x'][1], self._bbox['y'][0], self._bbox['y'][1]]

    def plot(self, ax=None, which='both'):
        """
        Plot the image (RGB & grayscale) side by side if no axis given.
        :param ax: the axis to plot in (if None, create a new figure)
        :param which: 'rgb' or 'gray' or 'both' 
        """
        if ax is not None and which == 'both':
            raise ValueError("If an axis is given, which must be 'rgb' or 'gray'")
        if which not in ['rgb', 'gray', 'both']:
            raise ValueError("which must be one of 'rgb', 'gray', or 'both'")
        if ax is None:
            if which == 'both':
                print("Creating subplots")
                fig, ax = plt.subplots(1, 2)
            else:
                print("Creating single plot")
                fig, ax = plt.subplots()
                ax = [ax, ax]
        extent = self.get_plot_extent()
        if which in ['rgb', 'both']:
            axis = ax if which == 'rgb' else ax[0]
            axis.imshow(self.rgb_img[::-1, :, :], extent=extent, )
            axis.set_title('RGB Image')
            axis.axis('off')
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[2:])

        if which in ['gray', 'both']:
            axis = ax if which == 'gray' else ax[1]
            axis.imshow(self.gray[::-1, :], cmap='gray', extent=extent)
            axis.set_title('Grayscale Image')
            axis.axis('off')
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[2:])

        return ax


def _test_image(plot=False):
    q_img1 = TestImage((400, 400))
    if plot:
        q_img1.plot(which='both')
        plt.show()
        plt.pause(0)
    return q_img1


def _test_similarity_metric(data1, data2, transf, plot=False):
    """
    Create a dataset, image pair. find corners, extract histograms, and create pairwise similarity matrix.
    Show a grid showing up to the first 10 of those corner windows on the top row, and below each, the
           highest & lowest N matching corners from the other image in descending order.
    :param data1: a tuple (QuantizedColorImage, corners) for the first image (and detected corners)
    :param data2: a tuple (QuantizedColorImage, corners) for the second image (and detected corners

    """
    n_color_bins = 5

    # Number of examples to show (set to 1 to see the best and worst, etc)
    qimg1, corners1 = data1
    qimg2, corners2 = data2
    img1, img2 = qimg1.rgb_img, qimg2.rgb_img

    window_size = 21  # size of the window around each corner to extract a descriptor (pixels)

    # Detect corners and extract descriptors for each corner
    hist1 = np.array([qimg1.get_patch_descriptor(corner_center, window_size)
                      for corner_center in corners1])
    hist2 = np.array([qimg2.get_patch_descriptor(corner_center, window_size)
                      for corner_center in corners2])

    # Compute the similarity between each pair of descriptors
    similarity = np.array([[TestImage.compare_descriptors(hist1[i], hist2[j])
                            for j in range(len(hist2))] for i in range(len(hist1))])

    # show the first few corners of image1 and (up to) their top 8 matches in image2, in descending order of similarity
    # Next to each image show the histogram.
    palette = qimg1.palette.astype(np.float32) / 255.

    def _plot_hist(ax, hist):
        """
        Bar graph using the color palette.
        :param hist:  The array of histogram counts (normalized).
        """
        ax.bar(range(len(hist)), hist, color=palette, width=7)
        ax.axis('off')

    def _plot_patch(ax, patch, score=None):
        """
        Show a patch with a score (if given)
        """
        ax.imshow((patch))
        ax.axis('off')
        if score is not None:
            ax.text(0, window_size, '%.2f' % score, color='k', fontsize=10, va='top')

    # Plot a few corners from image1 and their best and worst matches in image2
    n_corner_examples = 8
    n_match_examples = 4
    n_worst_examples = 2
    n_cols = np.min([n_corner_examples, len(corners1)]) * 2  # for image and histogram
    n_rows = 1 + n_worst_examples + n_match_examples  # for best and worst matches
    fig, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_corner_examples):

        # show the corner at the top
        window = qimg1.get_patch(corners1[i], window_size, which='rgb')
        _plot_patch(ax[0, i*2], window)
        _plot_hist(ax[0, i*2+1], hist1[i])

        # show the best
        for j, idx in enumerate(np.argsort(similarity[i])[:n_match_examples]):
            window = qimg2.get_patch(corners2[idx], window_size, which='rgb')
            _plot_patch(ax[j+1, i*2], window, similarity[i][idx])
            _plot_hist(ax[j+1, i*2+1], hist2[idx])

        # show the worst
        for j, idx in enumerate(np.argsort(similarity[i])[-n_worst_examples:]):
            window = qimg2.get_patch(corners2[idx], window_size, which='rgb')
            _plot_patch(ax[j+1+n_match_examples, i*2], window, similarity[i][idx])
            _plot_hist(ax[j+1+n_match_examples, i*2+1], hist2[idx])

    # Annotate plots window with separation lines under first row and between best/worst rows.
    # get the y-coordinates between the first and second row in figure coordinates.
    _, y1 = ax[1, 0].transAxes.transform([0, 1.15])
    _, y2 = ax[n_match_examples, 0].transAxes.transform([0, -.20])
    y1 = fig.transFigure.inverted().transform([0, y1])[1]
    y2 = fig.transFigure.inverted().transform([0, y2])[1]
    # x limits are 5% and 95% of the figure width
    x1 = 0.05
    x2 = 0.90
    # annotate

    ax[1, 0].annotate('', xy=(x1, y1), xytext=(
        x2, y1), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))
    ax[n_match_examples, 0].annotate('', xy=(x1, y2), xytext=(
        x2, y2), xycoords='figure fraction', arrowprops=dict(arrowstyle='-', color='k'))

    # Add titles to the three row sections (off to the left)
    ax[0, 0].text(-0.1, 0.5, 'image1 corners', fontsize=12, color='r',
                  ha='right', va='center', transform=ax[0, 0].transAxes)
    ax[1, 0].text(-0.1, 0.5, 'best %i matches\nof image2' % (n_match_examples, ), fontsize=12,
                  color='r', ha='right', va='center', transform=ax[1, 0].transAxes)
    ax[n_match_examples+1, 0].text(-0.1, 0.5, 'worst %i matches\nof image2' % (n_match_examples, ), fontsize=12,
                                   color='r', ha='right', va='center', transform=ax[n_match_examples+1, 0].transAxes)

    # remove most space between subplots
    plt.subplots_adjust(wspace=.1, hspace=.4)
    plt.suptitle('Test Corner Matching', fontsize=16)

    # show and return
    plt.show()


if __name__ == "__main__":
    plt.ioff()  # do all windows on startup
    i1 = _test_image(plot=True)
    print("All tests passed.")
