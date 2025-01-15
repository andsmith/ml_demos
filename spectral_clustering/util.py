import numpy as np
import cv2

def image_from_floats(floats, small=None, big=None):
    small = floats.min()  if small is None else small
    big =  floats.max() if big is None else big
    
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
        print("Making primary colors")
        colors = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        colors= colors[:n]
    elif n< 27:
        print("Making 27 colors")
        colors = np.array([[r, g, b] for r in [0, .5, 1.0] for g in [0, .5, 1.0] for b in [0, .5, 1.0]])
        colors= colors[:n]
    else:
        print("Making random colors")
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
    # print("Fitting text in %i x %i"%( text_area_width, text_area_height))
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
    # print("Failed to find good font size, using smallest.")
    return np.min(font_sizes), np.min(font_sizes)//2


def bbox_contains(box, x, y):
    """
    :param box: dict(x=(x0, x1), y=(y0, y1))
    :param x: float or np.array of x coordinates
    :param y: float or np.array of same length as x
    :return: bool or np.array of bools
    """
    if isinstance(x,np.ndarray):
        x_valid = np.logical_and(box['x'][0] <= x, x <= box['x'][1])
        y_valid = np.logical_and(box['y'][0] <= y, y <= box['y'][1])
        return np.logical_and(x_valid, y_valid)
    return box['x'][0] <= x <= box['x'][1] and box['y'][0] <= y <= box['y'][1]

def get_good_point_size(n_points, bbox):
    # TODO: make this scale wrt bbox size
    if n_points > 10000:
        pts_size = 2
    elif n_points > 1000:
        pts_size = 3
    elif n_points > 100:
        pts_size = 4
    else:
        pts_size = 5
    return pts_size

def sample_ellipse(center, p0, p1, n, random_state, empty_frac=0.):
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

    angles = angles_and_radii_rands[:, 0] * 2 * np.pi
    rad_rands = angles_and_radii_rands[:, 1]
    if empty_frac>0:
        rad_rands *= (1-empty_frac) 
    radii  = 1.-(rad_rands)


    points = np.array([center[0] + major_axis * radii * np.cos(angle) * np.cos(angles) - minor_axis * radii * np.sin(angle) * np.sin(angles),
                       center[1] + major_axis * radii * np.sin(angle) * np.cos(angles) + minor_axis * radii * np.cos(angle) * np.sin(angles)]).T
    return points

def vsplit_bbox(bbox, weights):
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
    return {'values':tic_values,
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


if __name__ == '__main__':
    _test_vsplit_bbox()

    print("All tests passed!")