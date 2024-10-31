"""
Demo of RANSAC algorithm, finding the affine transform between two images.
"""
import numpy as np
import matplotlib.pyplot as plt
from ransac import RansacModel, RansacDataFeatures
import cv2
from util_affine import Affine
from image_util import TestImage
import logging
from scipy.spatial.distance import cdist

PATCH_SIZE = 21
MARGIN = 15
SMOOTHING = 1.

# for plots
DOT_SIZE = 5
BUBBLE_SIZE = 7
LINEWIDTH = 1
NEON_GREEN = '#39FF14'
NEON_BLUE = '#1422FF'
NEON_RED = '#FF1421'
LEGEND_FONTSIZE = 7


class RansacImageData(RansacDataFeatures):
    """
    A dataset of two images for RANSAC, with an affine transform model.
        RansacImageData.get_features() returns a list of pairs of indices matching above threshold.
        each pair (i,j) corresponds to a pair of corners in the two images, the coordinates and descriptors can be
        accessed via the member variables corners_1, corners_2, desc_1, desc_2.  (Nx2 and N element arrays)

    """

    def __init__(self, data, harris_kwargs=None, matcher_threshold=0.5):
        """
        :param data: a tuple (img1, img2) of TestImages.
        :param harris_kwargs: dictionary of parameters for the Harris corner detector.
        :param matcher_threshold: threshold for matching descriptors (see TestImage.compare_descriptors).
        """
        # publics, set in _extract_features:
        self.corners_1, self.corners_2 = None, None  # x, y
        self.desc_1, self.desc_2 = None, None
        self.size = data[0].gray.shape[::-1]

        self._features = []  # exhaustive list of all features (pairs(c1,c2) matching above threshold
        self._harris_kwargs = harris_kwargs
        self._margin = 15
        self._m_threshold = matcher_threshold
        logging.info("RansacImageData created, extracting features...")
        super().__init__(data, 3)  # calls _extract_features
        logging.info("\textracted %i features (pairs of potentially corresponding corners)." % len(self._features))

    def _extract_features(self):
        """
        Find interest points in both images, extract their descriptors, match them.
        A "feature" for RANSAC is a pair of points with a high matching score.
        Create a list of such pairs (the "matches").
        """
        img1, img2 = self._dataset

        # detect interest points (corners)
        self.corners_1 = img1.find_corners(harris_kwargs=self._harris_kwargs)
        self.corners_2 = img2.find_corners(harris_kwargs=self._harris_kwargs)
        logging.info('\tFound %d corners in image 1.' % len(self.corners_1))
        logging.info('\tFound %d corners in image 2.' % len(self.corners_2))
        self.desc_1 = [img1.get_patch_descriptor(c, PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_1]
        self.desc_2 = [img2.get_patch_descriptor(c, PATCH_SIZE, smoothing=SMOOTHING) for c in self.corners_2]

        # match interest points to get candidates features (corresponding pairs)
        for i, d1 in enumerate(self.desc_1):
            for j, d2 in enumerate(self.desc_2):
                if TestImage.compare_descriptors(d1, d2) < self._m_threshold:
                    self._features.append((i, j))

        logging.info('\tFound %d candidate features.' % (len(self._features)))

    def get_n_features(self):
        return len(self._features)

    def get_min_sample_inds(self):
        """
        Don't pick any two features that have the same corner in common.
        (Should come up with a better way to sample features...)
        :return: indices into self._features for the minimum sample.
        """
        n_failed = 0
        while True:
            sample = np.random.choice(len(self._features), self.min_features, replace=False)
            c1 = [self._features[s][0] for s in sample]
            c2 = [self._features[s][1] for s in sample]
            if len(set(c1)) + len(set(c2)) < self.min_features * 2:
                n_failed += 1
                if n_failed > 100:
                    logging.warning("Couldn't find a sample with no common corners, returning anyway.")
                    break
            else:
                break

        return sample

    def get_features(self,  indices=None):
        if indices is not None:
            return [self._features[i] for i in indices]
        else:
            return self._features

    def plot_side_by_side(self, axis, which='gray'):
        _SPACING = 20
        img1, img2 = self._dataset
        if img1.size != img2.size:
            raise ValueError("Images must be the same size to plot side-by-side")

        bbox1 = img1.get_bbox()
        bbox2 = img2.get_bbox()
        img2_width = bbox2['x'][1] - bbox2['x'][0]
        both_bbox = {'x': (bbox1['x'][0], bbox2['x'][1] + img2_width + _SPACING),
                     'y': (bbox1['y'][0], bbox1['y'][1])}

        extent = (both_bbox['x'][0], both_bbox['x'][1],
                  both_bbox['y'][0], both_bbox['y'][1])

        if which == 'gray':
            spacer = np.zeros((img1.gray.shape[0], _SPACING), dtype=np.uint8) + 255
            axis.imshow(np.concatenate((img1.gray, spacer, img2.gray), axis=1)[::-1, :], cmap='gray', extent=extent)
        elif which == 'color':
            spacer = np.zeros((img1.rgb_img.shape[0], _SPACING, 3), dtype=np.uint8) + 255
            axis.imshow(np.concatenate((img1.rgb_img, spacer, img2.rgb_img), axis=1)[::-1, :, :], extent=extent)

        x_offset = img1.gray.shape[1] + _SPACING

        return x_offset

    def plot_features(self, axis, sample_inds=None, draw_image='gray'):
        """
        Draw images side-by-side with all corners detected, and lines of the same color from each corner in image2 to the corresponding corner in image1.
        If "sample_inds" is specified, draw the sample correspondences (points/lines) in green and only show these
        """

        x_offset = self.plot_side_by_side(axis, draw_image)

        def _draw_features(f_list, color=None):
            """
            Draw f_lists's points in both images, indicated color or different colors.
            Draw each line between corresponding points in the same color as the point in the first image.
            """
            c1_inds = list(set([f[0] for f in f_list]))

            if sample_inds is None:
                # Draw corners in image 1 in all colors
                points = [axis.plot(*corner_coord, 'o', markersize=BUBBLE_SIZE)[0]
                          for corner_coord in self.corners_1]
                point_colors = {i: point.get_color() for i, point in enumerate(points)}

                # Draw corners in image 2 in red
                axis.plot(self.corners_2[:, 0] + x_offset,
                          self.corners_2[:, 1], '.', markersize=DOT_SIZE, color='r')
            else:
                # Draw corners in image 1 green if not in the sample,
                other_inds = [i for i in range(len(self.corners_1)) if i not in c1_inds]
                [axis.plot(*self.corners_1[c, :], 'o', color=NEON_GREEN, markersize=DOT_SIZE)[0] for c in other_inds]
                # and sample points in red
                _ = [axis.plot(*self.corners_1[c, :], 'o', markersize=DOT_SIZE, color='r')[0] for c in c1_inds]
                point_colors = {c1_inds[i]: 'r' for i in range(len(c1_inds))}
                point_colors.update({other_inds[i]: NEON_GREEN for i in range(len(other_inds))})

                # Draw corners in image 2 in red/green similarly
                c2_inds = list(set([f[1] for f in f_list]))
                c2_other_inds = [i for i in range(len(self.corners_2)) if i not in c2_inds]
                [axis.plot(self.corners_2[c, 0] + x_offset, self.corners_2[c, 1], 'o', color=NEON_GREEN, markersize=DOT_SIZE)[0]
                    for c in c2_other_inds]
                _ = [axis.plot(self.corners_2[c, 0] + x_offset, self.corners_2[c, 1], 'o', markersize=DOT_SIZE, color='r')[0]
                     for c in c2_inds]

            # and draw the line:
            for (c_ind_1, c_ind_2) in f_list:
                c1, c2 = self.corners_1[c_ind_1], self.corners_2[c_ind_2]
                axis.plot([c1[0], c2[0] + x_offset], [c1[1], c2[1]], color=point_colors[c_ind_1], linewidth=LINEWIDTH)

        if sample_inds is not None:

            features = [self._features[i] for i in sample_inds]
            _draw_features(features, color='g')
            title = "Minimum feature sample, 3 random correspondences "
        else:

            _draw_features(self._features)
            title = "All %i corner correspondences\nwith match score > %.3f" % (len(self._features),
                                                                                self._m_threshold)
        axis.set_title(title)
        axis.axis('off')

    def get_imgs(self):
        return self._dataset


def plot_pointset_overlap(axis, pts1, pts2, min_dist, label1=None, label2_in=None, label2_out=None):
    """
    Plot pts1 in green dots, calculate which of pts2 are within min_dist of a point
    in pts1, and plot them in blue circles, plot the rest of pts2 in red circles.
    :returns the number of points in pts2 that are within min_dist of any point in pts1.
    """
    # plot pts1 in green
    axis.plot(pts1[:, 0], pts1[:, 1], '.', markersize=DOT_SIZE, color=NEON_GREEN, label=label1)

    # find points in pts2 that are within min_dist of any point in pts1
    distances = cdist(pts1, pts2)
    min_dists = distances.min(axis=0)
    inliers = min_dists < min_dist
    outliers = np.logical_not(inliers)
    axis.plot(pts2[inliers, 0], pts2[inliers, 1], 'o', markersize=BUBBLE_SIZE,
              color=NEON_BLUE, fillstyle='none', label=label2_in)
    axis.plot(pts2[outliers, 0], pts2[outliers, 1], 'o', markersize=BUBBLE_SIZE,
              color=NEON_RED, fillstyle='none', label=label2_out)
    return inliers.sum()


class RansacAffine(RansacModel):
    """
    An affine transform model for RANSAC.
    """
    _N_MIN_FEATURES = 3

    def __init__(self, data, inlier_threshold, training_inds, iter=None):
        """
        Fit the model with the given features.
        :param data: RansacImageData object containing the data & extracted features.
        :param inlier_threshold: threshold for inliers (will depend on implementation of RansacModel.evaluate)
        :param training_inds: list of indices of the features used to fit this RansacModel.
        :param iter: iteration number, for bookkeeping
        """
        self._training_inds = training_inds
        super().__init__(data, inlier_threshold, training_inds, iter)

    def _corner_coords_from_inds(self, feature_inds):
        train_features = self.data.get_features(indices=feature_inds)
        src_pts = np.array([self.data.corners_1[f[0]] for f in train_features])
        dst_pts = np.array([self.data.corners_2[f[1]] for f in train_features])
        return src_pts, dst_pts

    def _fit(self):
        """
        Fit an affine transform to this model's training data, score all featues, find in/outliers.
        """
        # Extract the coordinates of the features
        logging.info("Fitting model to %d features" % len(self._training_inds))

        src_pts, dst_pts = self._corner_coords_from_inds(self._training_inds)
        self._model_params = Affine.from_point_pairs(src_pts, dst_pts)

    def _get_inliers(self):
        features = self.data.get_features()
        corners1_moved = self._model_params.apply(self.data.corners_1)
        c2c_distances = cdist(corners1_moved, self.data.corners_2)
        scores = [c2c_distances[f[0]][f[1]] for f in features]
        return np.array(scores) < self.thresh

    @staticmethod
    def _animation_setup():
        """
        Set up the animation for plotting each iteration (2 windows, 3 subplots)
        Layout has the "training features" on top, (both images side-by-side with lines
        showing the candidate correspondences) and the "inliers/outliers on the bottom,
        showing the detected corners in each image and the corresponding corners induced by the 
        (inverse) transformation.
        """
        best_fig = plt.figure(figsize=(5, 5.5))
        current_fig = plt.figure(figsize=(5, 5.5))
        best_gs = best_fig.add_gridspec(2, 2)
        current_gs = current_fig.add_gridspec(2, 2)
        RansacModel._FIG = {'best': best_fig, 'current': current_fig, 'best_iter': -1}

        RansacModel._AXES = {'best': {'features': best_fig.add_subplot(best_gs[0, :]),
                                      'img1': best_fig.add_subplot(best_gs[1, 0]),
                                      'img2': best_fig.add_subplot(best_gs[1, 1])},

                             'current': {'features': current_fig.add_subplot(current_gs[0, :]),
                                         'img1': current_fig.add_subplot(current_gs[1, 0]),
                                         'img2': current_fig.add_subplot(current_gs[1, 1])}}

    def plot_nesting_bboxes(self, axis, img, c1, c2, back_transf, which='gray'):
        """
        Show the image, plot detected corners as green dots, plot back-transformed corners
        as circles, blue if within threshold of a green dot, else red.
        Show the bounding box of the other image back-transformed.  (assume both are the same shape)

        :param axis: axis to plot in
        :param img: TestImage object
        :param c1: corners in img
        :param c2: corners in the other image
        :param back_transf: Affine object, mapping 2->1
        """
        img.plot(axis, which=which)

        # Plot the image 1 detected corners and corners 2 transformed to image 1 space
        c2_back = back_transf.apply(c2)
        n_match = plot_pointset_overlap(axis, c1, c2_back, 10.0, label1='Image 1 corners',
                                        label2_in='Img2C-match', label2_out='Img2C-no match')

        # add legend
        # axis.legend(fontsize=LEGEND_FONTSIZE, loc='lower left')
        # draw the bounding box of image 2 in image 1 space
        bbox_2 = img.get_bbox()
        bbox_2_corners = np.array([[bbox_2['x'][0], bbox_2['y'][0]],
                                   [bbox_2['x'][1], bbox_2['y'][0]],
                                   [bbox_2['x'][1], bbox_2['y'][1]],
                                   [bbox_2['x'][0], bbox_2['y'][1]],
                                   [bbox_2['x'][0], bbox_2['y'][0]],])
        bbox_2_back = back_transf.apply(bbox_2_corners)
        axis.plot(bbox_2_back[:, 0], bbox_2_back[:, 1], 'g-', linewidth=LINEWIDTH*2)
        axis.axis('off')
        axis.set_aspect('equal')

        # zoom out a bit
        margin = 0.1
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x_marg, y_marg = x_span * margin, y_span * margin
        axis.set_xlim(xlim[0] - x_marg, xlim[1] + x_marg)
        axis.set_ylim(ylim[0] - y_marg, ylim[1] + y_marg)

        return n_match

    def plot_iteration(self, data, best_model, which='gray', is_final=False, max_iter=0):
        """
        if is_final = False:
            four plots, above are plots for the current model and below are plots for the best model so far

        if is_final = True:
            Show the same four windows, but replacing the upper two with the "final" model, estimated from the consensus set.
            Open a second window with two plots, the first showing image 1 in grayscale and the green outline of the bounding box of image 2, transformed backwards,
            and the second showing image 2 transformed into image 1's space (i.e. filling in the green box in image21 with the actual warped image2)

        """

        if RansacModel._AXES is None:
            RansacAffine._animation_setup()

        if not is_final:
            # plot the current model
            self._plot_model(data, self, RansacModel._AXES['current'], which=which,
                             title_prefix="")
            plt.suptitle("RANSAC iteration %i / %i" % (self.iter+1, max_iter))
            if RansacModel._FIG['best_iter'] in [-1, self.iter]:
                self._plot_model(data, best_model, RansacModel._AXES['best'], which=which,
                                 title_prefix='Best iteration (%i) had' % best_model.iter)
                plt.show()
        else:
            # plot the final model
            self._plot_model(data, self,  RansacModel._AXES['current'],
                             title_prefix='Final model using all', which=which)
            plt.suptitle("RANSAC completed %i iterations." % (self.iter,))
            plt.show()

    @staticmethod
    def _plot_model(data, model, axes, which='gray', title_prefix=""):
        """
        Plot the model (affine transform) in the given axes.
        On the left, show image1, image2's recovered bounding box,
        on the right, image 2.   IOn both images, plot the detected corners and the other image's
        detected corners transformed to its space and colored to indicate whether each is close to 
        a corner in the image.
        :param data: RansacImageData object
        :param model: RansacAffine object (e.g. self)
        :param axes: dictionary of axes to plot in with, 'features' , 'img1', and 'img2'
        :param which: 'gray' or 'color' for the image to plot
        """
        axes['img1'].clear()
        axes['img2'].clear()
        axes['features'].clear()

        # plot minimum sample features:
        data.plot_features(axes['features'], sample_inds=model._training_inds)

        img1, img2 = data.get_imgs()
        transf = model._model_params
        inv_tr = model._model_params.invert()

        # plot the images, transformed corners, and bounding boxes:
        model.plot_nesting_bboxes(axes['img1'], img1, model.data.corners_1, model.data.corners_2, inv_tr, which=which)
        model.plot_nesting_bboxes(axes['img2'], img2, model.data.corners_2, model.data.corners_1, transf, which=which)

        axes['img1'].set_title('Image 1')
        axes['img2'].set_title('Image 2')
        axes['features'].set_title("%s %i inliers (%.2f %%)" % (title_prefix,
                                                                np.sum(model.inlier_mask),
                                                                100 * np.mean(model.inlier_mask)))


def _test(plot=False):
    # Freeze random state
    # np.random.seed(210)
    # if plot:
    #    plt.ion()

    matcher_threshold = 0.4
    size = 500, 500
    noise_level = 0.1
    args = dict(blockSize=8,
                ksize=23,
                k=0.045)

    # data
    img1 = TestImage(size)
    img2, orig_transf = img1.transform(noise_level)
    data = (img1, img2)
    print(orig_transf)

    # test RansacImageData, feature extraction and plotting
    ransac_data = RansacImageData(data, harris_kwargs=args, matcher_threshold=matcher_threshold)
    # fig, ax = plt.subplots(2, 1)
    # ransac_data.plot_features(ax[0], draw_image='color')
    train_inds = ransac_data.get_min_sample_inds()
    # ransac_data.plot_features(ax[1], train)
    # plt.show()

    # show image2, corners in image2, corners in image1 transformed to image2 space
    img1_corners = ransac_data.corners_1
    transf_corners = orig_transf.apply(img1_corners)
    distances = cdist(transf_corners, ransac_data.corners_2)
    score = np.sum(distances.min(axis=1) < 10.0)

    # test RansacAffine
    inlier_threshold = 5.0

    model = RansacAffine(ransac_data, inlier_threshold, train_inds, iter=0)
    # set model params to the original transformation for debugging
    model._model_params = orig_transf
    model.inlier_mask = model._get_inliers()

    if plot:
        model._animation_setup()
        # fig, ax = plt.subplots(1, 2)

        # plot separately
        # model.plot_nesting_bboxes(ax[0])
        # model.plot_induced_transform(ax[1])
        # ax[0].set_title('Image 1 with recovered\nb-box of Image 2')
        # ax[1].set_title('Image 2')

        # plot together
        model.plot_iteration(ransac_data, model, is_final=False, max_iter=2)
        plt.show()

        plt.pause(0)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    _test(plot=True)
