import numpy as np
from scipy.interpolate import interp2d


class Affine(object):
    """
    Utilities for daeling with affine transforms in 2d.

    Internally, this is stored as an arbitrary 2x3 matrix.

    Affine vectors are created from these components, applied in this order:
        * Scaling (x, y) - scale the x and y axes by different amounts
        * Rotation (theta) - rotate the x and y axes by the same amount
        * Translation (x, y) - move the x and y axes by different amounts

    i.e.: 

            M*x = T*R*S*[x,y,1]^T = [x', y', 1] 

    where:

        * S is a scaling matrix (diag(scale_x, scale_y), ...)
        * R is a rotation matrix (angle)
        * T is a translation matrix (translate_x, translate_y)
    """

    def __init__(self, rotate=0., scale=(1., 1.), translate=(0., 0.)):
        """
        Initialize an affine* transform of the 2d plane. 

        The order of operations is:

           transform = translate(scale*rotate(x))

        :param rotate: rotation angle in radians
        :param scale: scaling factor (x, y)
        :param translate: translation vector in pixels
        """

        self.M = get_matrix(np.array(scale), rotate, np.array(translate))

    def __eq__(self, other):
        """
        Return true if the two matrices are close enough to equal.
        """
        return np.allclose(self.M, other.M)

    def __str__(self):
        return 'Affine: \n' + str(self.M) + "\n"

    @staticmethod
    def from_random(rand_transf_params='demo'):
        _scale_range = 0.2
        scale = 1.0
        if rand_transf_params in ['detection_tuner', 'demo']:

            angle = np.deg2rad(30) #np.random.rand() * np.pi*2
            print(np.rad2deg(angle))
            translation = np.random.randn(2) * 40  # not too far away
            scale_x = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2
            scale_y = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2

        elif rand_transf_params == 'test_affine':
            angle = np.radians(20.)
            translation = np.array((-18, 20))
            scale_x = 1.3
            scale_y = .8

        elif rand_transf_params in ['test_detection_tuner']:
            angle = np.pi / 6
            scale_x = 1.333
            scale_y = 1.2
            translation = np.array([75, 42])

        #elif rand_transf_params in ['detection_tuner']:
        #    angle = np.pi / 6 + np.random.rand(1)[0] * np.pi / 4
        #    scale_x = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2
        #    scale_y = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2
        #    translation = np.random.rand(2) * 20

        elif rand_transf_params == 'tune_matching':
            angle = np.pi / 6
            scale_x = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2
            scale_y = scale + np.random.rand(1)[0] * _scale_range - _scale_range/2
            translation = np.array([150, 0])
        else:
            raise ValueError('Unknown kind of random transform to make: %s' % rand_transf_params)
        t_scale = (scale_x, scale_y)
        return Affine(angle, t_scale, translation)

    @staticmethod
    def from_matrix(m):
        """
        Create an Affine object from a 2x3 matrix.
        """
        a = Affine()
        a.M = m
        return a

    @staticmethod
    def from_point_pairs(src_pts, dst_pts):
        """
        Fit an affine transform to the corresponding points in src_pts and dst_pts.

        Let, M be an affine transformation matrix, then:

            M [src_pts, 1]' = [dst_pts]'

        We recover the matrix M using least squares.

        :param src_pts: Nx2 array of source points
        :param dst_pts: Nx2 array of destination points
        """
        # Add a column of ones to the src_pts
        src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))

        # Solve the least squares problem to get the Affine matrix
        m_inv = np.linalg.lstsq(src_pts, dst_pts, rcond=None)[0].T

        a = Affine.from_matrix(m_inv)
        return a

    def invert(self):
        """
        Find an affine transform that inverts self's transform.
        """
        m_sqare = np.vstack((self.M, np.array([0, 0, 1])))
        m_inv = np.linalg.inv(m_sqare)
        a = Affine()
        a.M = m_inv[:2, :]
        return a

    def apply(self, pts):
        """
        Apply the affine transform to the given points.
        (center wrt 'size', apply transform, un-center)
        """
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        pts_out = np.dot(self.M, pts_h.T).T
        return pts_out

    def warp_image(self, image):
        """
        Apply the affine transform to the given image.
        It's a palette image with color indices not intensities, so use nearest-neighbors instead of interpolating.



        (Find where coords in bounding box of new image originate in the old image and use the nearest pixel value
        in the old.)
        :param image: HxW or HxWx3 image
        """
        fill = image[0, 0]  # out of bounds areas arbitrarily take value of this pixel
        size = image.shape[1], image.shape[0]
        x_y_offset = np.array([size[0] / 2, size[1] / 2], dtype=np.float64) - 0.5

        # original & new bounding box coords
        x_img, y_img = np.meshgrid(np.arange(size[0], dtype=np.float64),
                                   np.arange(size[1], dtype=np.float64))
        x_img -= x_y_offset[0]
        y_img -= x_y_offset[1]

        # invert transform, see where each new image pixel gets its value
        inv = self.invert()
        pts_orig = np.vstack((x_img.flatten(), y_img.flatten())).astype(float).T
        pts_transformed = inv.apply(pts_orig)

        # equivalent to nearest-neighbor interpolation
        pts_transformed = np.round(pts_transformed).astype(int).reshape(-1, 2)

        # Un-center:
        pts_transf_px = (pts_transformed + x_y_offset).astype(np.int32)

        # in/out of bounds
        valid = (pts_transf_px[:, 0] >= 0) & (pts_transf_px[:, 0] < size[0]) & \
                (pts_transf_px[:, 1] >= 0) & (pts_transf_px[:, 1] < size[1])

        # fill in valid/invalid values
        if len(image.shape) == 2:
            img_flat = np.zeros(size[0] * size[1], dtype=image.dtype)
            img_flat[valid] = image[pts_transf_px[valid, 1], pts_transf_px[valid, 0]]
            img_flat[~valid] = fill
            img = img_flat.reshape(size)
        else:
            img_flat = np.zeros((size[0] * size[1], image.shape[2]), dtype=image.dtype)
            for i in range(image.shape[2]):
                img_flat[valid, i] = image[pts_transf_px[valid, 1], pts_transf_px[valid, 0], i]
            img_flat[~valid] = fill
            img = img_flat.reshape(size[1], size[0], image.shape[2])

        return img


def get_matrix(scale, rotate, translate):
    """
    Return the 2x3 matrix representation of the affine transform.
    """

    m = np.array([[np.cos(rotate), -np.sin(rotate), translate[0]],
                  [np.sin(rotate), np.cos(rotate), translate[1]]])

    m[:, 0] *= scale[0]
    m[:,] *= scale[1]

    return m


def test_affine(plot=True):
    size = (100, 100)
    aff = Affine.from_random('test_affine')
    print('Original transf:\n\t', aff)
    pts = np.random.rand(20, 2) * size
    pts -= size[0] / 2

    pts_transformed = aff.apply(pts)

    # fit an affine transform the (point, transformed_point) pairs, and recover the original points.
    aff_recovered = Affine.from_point_pairs(pts, pts_transformed)
    print('Transform recovered from %i random points:\n\t' % (pts.shape[0],), aff_recovered)

    pts_transformed_w_aff_rec = aff_recovered.apply(pts)

    # invert the original transform & move the points back.
    aff_inverted = aff.invert()
    print("Transform inverted: %s" % aff_inverted)
    pts_returned = aff_inverted.apply(pts_transformed)

    # finally invert the inverted transform and make sure it is the same as the original transform.
    aff_inverted_inverted = aff_inverted.invert()

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(pts[:, 0], pts[:, 1], 'o')
        plt.plot(pts_transformed[:, 0], pts_transformed[:, 1], 's')
        plt.plot(pts_returned[:, 0], pts_returned[:, 1], 'x')
        plt.legend(['Original', 'Transformed', 'Returned'])
        plt.show()

    # Print original, recovered and double-inverted transforms:
    print('Twice inverted:', aff_inverted_inverted)

    # make sure estimated affine transform is close to the original, and points were moved the same by both
    print("\n", aff, "\n", aff_recovered)

    # make sure transformed points are mapped back to the original points.
    assert np.allclose(pts, pts_returned), 'Affine transform failed to invert correctly.'

    assert aff == aff_recovered, 'Affine transform failed to estimate correctly.'

    assert np.allclose(pts_transformed, pts_transformed_w_aff_rec), 'Affine transform failed to recover points correctly.'

    # make sure doubly inverted is the same as the original:
    assert aff == aff_inverted_inverted, 'Affine transform failed to invert twice and remain constant.'

    print('Affine transform passed.')


def in_bbox(bbox, x):
    """
    Return True for 2d points in the bounding box bbox.
    :param bbox: dict(x=(min_x, min_y), y=(max_x, max_y))
    :param x: Nx2 array of points
    """
    return (bbox['x'][0] <= x[:, 0]) & (x[:, 0] <= bbox['x'][1]) & \
           (bbox['y'][0] <= x[:, 1]) & (x[:, 1] <= bbox['y'][1])


if __name__ == '__main__':

    # print(Affine.from_random('demo'))
    # print(Affine.from_random('test_detection_tuner'))
    # print(Affine.from_random('detection_tuner'))
    # print(Affine.from_random('tune_matching'))

    test_affine()
    print('All tests passed.')
