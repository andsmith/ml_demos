import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import interp1d


def pca(points, d):
    """
    Return the first d principal axes of the point cloud.
    :param points: N x M array of points
    :param d: integer, number of principal components to return
    :return: M x d array of principal axes, vector of standard deviations along each axis
    """
    points = points - np.mean(points, axis=0)
    covar = np.cov(points, rowvar=False)
    evals, evecs = np.linalg.eigh(covar)
    idx = np.argsort(evals)[::-1][:d]
    p_axes = evecs[:, idx]
    vars = evals[idx]
    if np.sum(vars<=0) > 0:
        logging.warning("Some eigenvalues are <= 0, returning 0s for those axes.")
        vars[vars<=0] = 0
    sds = np.sqrt(vars)
    return p_axes, sds


def test_pca():
    n_clusters = 5
    n_points = 2100
    for _ in range(n_clusters):
        center = np.random.rand(2) * 20
        covar =  np.random.randn(2, 2)
        covar = np.dot(covar, covar.T)
        points = np.random.multivariate_normal(center, covar, n_points)
        evecs, lengths = pca(points, 2)
        plt.scatter(points[:, 0], points[:, 1], s=2, alpha=.9)
        for length, vec in zip(lengths, evecs.T):
            plt.plot([center[0]-vec[0]*length, center[0] + vec[0]*length],
                     [center[1]-vec[1]*length, center[1] + vec[1]*length], lw=3)
    # equal axis
    plt.axis('equal')
    plt.show()


def image_from_floats(floats, small=None, big=None):
    small = floats.min() if small is None else small
    big = floats.max() if big is None else big

    values = (floats - small) / (big - small) * 255
    return values.astype(np.uint8)


def apply_colormap(floats, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to an image.
    :param floats: 2D array of floats
    :param colormap: cv2 colormap
    :return: 3D array of uint8
    """
    image = image_from_floats(floats)
    return cv2.applyColorMap(image, colormap)


def test_image_from_floats():
    image = np.random.randn(100, 100)
    image_uint8 = image_from_floats(image)
    cv2.imwrite('test_image_from_floats.png', image_uint8)


def scale_bbox(bbox_rel, bbox_abs):
    """
    Get absolute coordinates of a bounding box from relative coordinates.
    Boxes are dicts with keys 'x' and 'y' and values (x0, x1) and (y0, y1)

    :param bbox_rel: bbox in unit square
    :param bbox_abs:  bbox in pixels
    :return: bbox in pixels, the subset of bbox_abs that corresponds to bbox_rel
    """
    x0, x1 = bbox_rel['x']
    y0, y1 = bbox_rel['y']
    x0_abs, x1_abs = bbox_abs['x']
    y0_abs, y1_abs = bbox_abs['y']
    return {'x': (int(x0_abs + x0 * (x1_abs - x0_abs)),
                  int(x0_abs + x1 * (x1_abs - x0_abs))),
            'y': (int(y0_abs + y0 * (y1_abs - y0_abs)),
                  int(y0_abs + y1 * (y1_abs - y0_abs)))}


def unscale_coords(bbox, coords):
    """
    Get relative coordinates of a point from absolute coordinates.
    :param bbox: bbox in pixels
    :param coords: N x 2 array of points in pixel coords
    :return: points in unit square
    """
    x0, x1 = bbox['x']
    y0, y1 = bbox['y']
    return np.array([(coords[:, 0] - x0) / (x1 - x0),
                     (coords[:, 1] - y0) / (y1 - y0)]).T


def test_bbox_scaling():
    bbox_rel = {'x': (.1, .9), 'y': (.1, .9)}
    bbox_abs = {'x': (0, 100), 'y': (0, 100)}
    bbox_abs_scaled = scale_bbox(bbox_rel, bbox_abs)
    assert bbox_abs_scaled == {'x': (10, 90), 'y': (10, 90)}
    unscalled_coords = unscale_coords(bbox_abs, np.array([[10, 10], [90, 90]]))
    assert np.allclose(unscalled_coords, np.array([[.10, .10], [.9, .9]]))


def get_n_disp_colors(n):
    """
    Get colors to display n clusters.
    Should be somewhat dark.
    """
    if n < 5:
        logging.info("Making primary colors")
        colors = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        colors = colors[:n]
    elif n < 27:
        logging.info("Making 27 colors")
        colors = np.array([[r, g, b] for r in [0, .5, 1.0] for g in [0, .5, 1.0] for b in [0, .5, 1.0]])
        colors = colors[:n]
    else:
        logging.info("Making random colors")
        colors = np.random.rand(n, 3) * .5 + .2
    colors = (colors * 255).astype(np.uint8)
    return colors


def get_ellipse_points(center, p0, p1, n_points):
    """
    Get n_points on the ellipse with given center and passing through 
    major, and minor axes points p0 and p1.
    """
    # get the major and minor axes
    major_axis = np.linalg.norm(p0 - center)
    minor_axis = np.linalg.norm(p1 - center)
    # get the angle of the major axis
    angle = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    # get the points on the ellipse
    angles = np.linspace(0, 2 * np.pi, n_points)
    points = np.array([center[0] + major_axis * np.cos(angle) * np.cos(angles) - minor_axis * np.sin(angle) * np.sin(angles),
                       center[1] + major_axis * np.sin(angle) * np.cos(angles) + minor_axis * np.cos(angle) * np.sin(angles)]).T
    return points


def calc_font_size(lines, bbox, font, item_spacing_px, n_extra_v_spaces=0, search_range=(.1, 10)):
    """
    Calculate the largest font size to fit the text in the bbox.
    :param lines: list of strings, one per line, or a single string
    :param bbox: dict(x=(x0, x1), y=(y0, y1))
    :param font: cv2 font constant
    :param item_spacing_px: int, vertical spacing between lines, left and right horizontal spacing within box.
    :param n_extra_v_spaces: int, number of extra item_spacing_px to add to the bottom.
    """
    if isinstance(lines, str):
        lines = [lines]
    font_sizes = np.linspace(.2, 3.0, 100)[::-1]

    text_area_width = bbox['x'][1] - bbox['x'][0] - item_spacing_px * 2
    text_area_height = bbox['y'][1] - bbox['y'][0] - item_spacing_px * 2 - n_extra_v_spaces * item_spacing_px
    # logging.info("Fitting text in %i x %i"%( text_area_width, text_area_height))
    for font_size in font_sizes:
        (title_width, title_height), baseline = cv2.getTextSize(lines[0], font, font_size, 1)
        header_height = title_height + item_spacing_px * 2 + 1 + baseline  # space, title, space, line
        if len(lines) > 1:

            text_sizes = [cv2.getTextSize(text, font, font_size, 1)[0] for text in lines[1:]]
            text_widths, text_heights = zip(*text_sizes)
        else:
            text_widths, text_heights = [0], [0]

        total_height = header_height + np.sum(text_heights) + item_spacing_px * \
            (len(text_heights) + 1) + baseline * len(text_heights)
        total_width = max(title_width, max(text_widths))
        item_height = text_heights[0]
        if total_height < text_area_height and total_width < text_area_width:
            return font_size, int(item_height/2)
    logging.warning("Failed to find good font size, using smallest.")
    return np.min(font_sizes), np.min(font_sizes)//2


def bbox_contains(box, x, y):
    """
    :param box: dict(x=(x0, x1), y=(y0, y1))
    :param x: float or np.array of x coordinates
    :param y: float or np.array of same length as x
    :return: bool or np.array of bools
    """
    if isinstance(x, np.ndarray):
        x_valid = np.logical_and(box['x'][0] <= x, x <= box['x'][1])
        y_valid = np.logical_and(box['y'][0] <= y, y <= box['y'][1])
        return np.logical_and(x_valid, y_valid)
    return box['x'][0] <= x <= box['x'][1] and box['y'][0] <= y <= box['y'][1]


def get_good_point_size(n_points, bbox):
    # TODO: make this scale wrt bbox size
    # returns: number of pixels
    if n_points > 10000:
        pts_size = 2
    elif n_points > 1000:
        pts_size = 3
    elif n_points > 100:
        pts_size = 4
    else:
        pts_size = 5
    return pts_size


def sample_canonical_ellipse(n, minor, random_state, empty_frac=0.):
    """
    Sample an ellipse at the origin with major axis 1, and no rotation
    """
    rands = random_state.uniform(0, 1, (n, 2))
    angles = rands[:, 0] * np.pi * 2
    radii = rands[:, 1]
    if empty_frac > 0:
        # is this slowing things down?
        radii = (1-np.sqrt(1-empty_frac) * radii)
    radii = np.sqrt(radii)  # xform for circular dist.
    x = radii * np.cos(angles)
    y = radii * np.sin(angles) * minor

    return np.hstack([x[:, None], y[:, None]])


def rotate_points(points, angle):
    """
    Rotate points by angle.
    TODO: THis is slow, remove need for canonical+rotation
    """
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rot_matrix.T)


def sample_ellipse(center, p0, p1, n, random_state, empty_frac=0.):
    """
    Generate N random points inside the ellipse.
    """
    # get the major and minor axes
    major_axis = np.linalg.norm(p0 - center)
    minor_axis = np.linalg.norm(p1 - center)
    # degenerate cases
    if major_axis == 0:
        if minor_axis == 0:
            return np.tile(center, (n, 1))
        else:
            p0, p1 = p1, p0
    # get the angle of the major axis
    angle = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    # get the points on the ellipse
    ratio = minor_axis / major_axis
    points_canonical = sample_canonical_ellipse(n, ratio, random_state, empty_frac)
    points_scaled = points_canonical * major_axis
    points_rotated = rotate_points(points_scaled, angle)
    points = points_rotated + np.array(center)
    return points


def sample_ellipse_old(center, p0, p1, n, random_state, empty_frac=0.):
    """
    Generate N random points inside the ellipse.
    """
    # get the major and minor axes
    major_axis = np.linalg.norm(p0 - center)
    minor_axis = np.linalg.norm(p1 - center)
    # get the angle of the major axis
    angle = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    # get the points on the ellipse
    angles_and_radii_rands = random_state.uniform(0, 1, (n, 2))
    # transform radii rands so points will be uniformly distributed in the ellipse
    angles_and_radii_rands[:, 1] = (angles_and_radii_rands[:, 1])**2

    angles = angles_and_radii_rands[:, 0] * 2 * np.pi
    rad_rands = angles_and_radii_rands[:, 1]
    if empty_frac > 0:
        rad_rands *= (1-empty_frac)
    radii = 1.-(rad_rands)

    points = np.array([center[0] + major_axis * radii * np.cos(angle) * np.cos(angles) - minor_axis * radii * np.sin(angle) * np.sin(angles),
                       center[1] + major_axis * radii * np.sin(angle) * np.cos(angles) + minor_axis * radii * np.cos(angle) * np.sin(angles)]).T
    return points


def vsplit_bbox(bbox, weights, integer=True):
    """
    Split the bounding box vertically, with relative heights given by weights.
    :param bbox: dict(x=(left, right), y=(top, bottom))
    :param weights: list of positive floats
    :return: list of bounding boxes, from top to bottom
    """
    x = bbox['x']
    y = bbox['y']
    total_height = y[1] - y[0]
    bboxes = []
    top = y[0]
    weights = np.array(weights) / np.sum(weights)
    for w in weights:
        bottom = (top + w * total_height)
        if integer:
            bottom = int(bottom)
        bboxes.append({'x': x, 'y': (top, bottom)})
        top = bottom
    return bboxes


def indent_bbox(bbox, n_px):
    return {'x': (bbox['x'][0] + n_px, bbox['x'][1] - n_px),
            'y': (bbox['y'][0] + n_px, bbox['y'][1] - n_px)}


def hsplit_bbox(bbox, weights, integer=True):
    """
    Split the bounding box horizontally, with relative widths given by weights.
    :param bbox: dict(x=(left, right), y=(top, bottom))
    :param weights: list of positive floats
    :return: list of bounding boxes, from left to right
    """
    x = bbox['x']
    y = bbox['y']
    total_width = x[1] - x[0]
    bboxes = []
    left = x[0]
    weights = np.array(weights) / np.sum(weights)
    for w in weights:
        right = left + w * total_width
        if integer:
            right = int(right)
        bboxes.append({'x': (left, right), 'y': y})
        left = right
    return bboxes


def _test_vsplit_bbox():
    import matplotlib.pyplot as plt

    bbox = {'x': (0, 1), 'y': (0, 1)}
    bboxes = vsplit_bbox(bbox, [1, 2, 1])

    def plot_bbox(bbox, color):
        x = bbox['x']
        y = bbox['y']
        plt.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], color)

    for i, b in enumerate(bboxes):
        plot_bbox(b, 'C{}'.format(i))

    plt.show()


def get_tic_positions(n, max_val, pixel_range):
    """
    With an axis spanning pixel_range pixels, where should the tic marks be?
    :param n: number of tic marks (approximate)
    :param max_val: maximum value on the axis (0 is minimum)
    :param pixel_range: number of pixels the axis spans
    """
    mag_val = np.floor(np.log10(max_val))
    step = 10 ** mag_val
    # find the step size
    while max_val / step < n:
        step /= 2
    # find the first tic mark
    first_tic = np.ceil(max_val / step) * step
    # find the last tic mark
    last_tic = np.floor(0 / step) * step
    # get the values
    tic_values = np.arange(first_tic, last_tic - step, -step)
    # scale to pixel range
    tic_positions = (tic_values / max_val) * pixel_range
    return {'values': tic_values,
            'pos_px': tic_positions,
            'n': len(tic_values)}


def add_sub_image(img, sub_img, bbox):
    """
    Blit the sub_image over the image in the bbox.
    The image may be smaller than the bbox, it goes in the top left corner.
    """
    x0, x1 = bbox['x']
    y0, y1 = bbox['y']
    sub_img = sub_img[:y1-y0, :x1-x0]
    img[y0:y0+sub_img.shape[0], x0:x0+sub_img.shape[1]] = sub_img


def test_sample_canonical_ellipse(empty_frac=0):
    n = 1000
    random_state = np.random.RandomState(0)
    points = sample_canonical_ellipse(n, .3, random_state, empty_frac)
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=.9)
    plt.show()


def test_sample_elipse():
    center = np.array([0, 0])
    p0 = np.array([1, 1])
    p1 = np.array([-2, 2])
    n = 1000
    random_state = np.random.RandomState(0)
    points = sample_ellipse(center, p0, p1, n, random_state)
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=.9)
    # now plot the actual ellipse
    n_interp = 10000
    points = get_ellipse_points(center, p0, p1, n_interp)
    plt.plot(points[:, 0], points[:, 1], lw=2)
    plt.show()


def sample_gaussian(center, p0, p1, n, random_state):
    """
    Sample points from a gaussian with params
    :param center: center of the gaussian
    :param p0: endpoint of axis 1
    :param p1: endpoint of axis 2
    :param n: number of points to sample
    :param random_state:  numpy random state for reproducibility
    """
    # get the major and minor axes
    major_axis = np.linalg.norm(p0 - center)
    minor_axis = np.linalg.norm(p1 - center)
    # get the angle of the major axis
    angle = -np.arctan2(p0[1] - center[1], p0[0] - center[0])
    # get the covariance matrix, but scale so major axis is 1.0.
    min_a = minor_axis / major_axis
    maj_a = 1.0
    cov = np.array([[maj_a, 0], [0, min_a]])

    # rotate the covariance matrix
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    cov = np.dot(cov, rot_matrix)

    points = random_state.multivariate_normal(np.zeros(2), np.eye(2), n)
    points = np.dot(points, cov) * major_axis + center  # co-vary, scale and translate
    return points


def test_sample_gaussian():
    n = 10000
    center = 10, 7.5
    p0_offset = np.array((2, 2))
    p1_offset = np.array((-7, 1))
    p0 = np.array(center) + p0_offset
    p1 = np.array(center) + p1_offset
    r0 = np.linalg.norm(p0_offset)
    r1 = np.linalg.norm(p1_offset)

    points = sample_gaussian(center, p0, p1, n, np.random.RandomState(0))
    # plot points
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=.1)
    # plot ellipse around sigma, 2*sigma
    n_interp = 1000
    points = get_ellipse_points(center, p0, p1, n_interp)
    plt.plot(points[:, 0], points[:, 1], lw=3, color='r')
    points = get_ellipse_points(center, p0 + p0_offset, p1+p1_offset, n_interp)
    plt.plot(points[:, 0], points[:, 1], lw=3, color='g')
    plt.show()


def orthornormalize(vectors):
    o_vecs = vectors.copy()
    o_vecs[0] = o_vecs[0] / np.linalg.norm(o_vecs[0])
    for i in range(1, len(vectors)):
        for j in range(i):
            o_vecs[i] -= np.dot(o_vecs[i], o_vecs[j]) * o_vecs[j]
        o_vecs[i] /= np.linalg.norm(o_vecs[i])
    return o_vecs


def test_orthornormalize():
    vecs = np.random.randn(3, 3)
    vecs = orthornormalize(vecs)
    print(vecs)
    print(np.linalg.norm(vecs[0]))
    print(np.linalg.norm(vecs[1]))
    print(np.linalg.norm(vecs[2]))
    print(np.dot(vecs[0], vecs[1]))
    print(np.dot(vecs[0], vecs[2]))
    print(np.dot(vecs[1], vecs[2]))


if __name__ == '__main__':
    # test_sample_canonical_ellipse(empty_frac=.667)
    # test_sample_canonical_ellipse(empty_frac=0)
    # test_sample_elipse()
    # test_sample_gaussian()
    test_orthornormalize()
