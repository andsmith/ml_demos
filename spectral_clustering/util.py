import numpy as np
import cv2
import matplotlib.pyplot as plt

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



def make_data(n):
    """
    Return an N x N matrix.
    """
    data = np.random.rand(n, n)
    return data

def _get_good_markersizes(sizes, img_size, density=0.8):
    """
    The goal is to have a certain fraction of the canvas colored by points.
    Make each cluster cover approximately the same area, so scale marker sizes
    by the square root of the number of points.   (i.e. assume no overlapping
    points, but plot them anyway.)  
        NOTE:  this is too big, return sqrt(area)

    :param sizes: list of integers, number of points in each cluster
    :param img_size: tuple of integers, w,h of pixels to cover
    :param density: float, fraction of the canvas to be covered by points
    :return: list of integers, marker sizes for each cluster
    """
    sizes = np.array(sizes)
    total_points = np.sum(sizes)
    n_clusters = len(sizes)

    # pixels squared per cluster
    px2_per_cluster = img_size[0] * img_size[1] * density / n_clusters 
    area_per_marker = px2_per_cluster / sizes

    return np.sqrt(area_per_marker)


def test_get_good_markersizes():
    sizes = [10, 20]
    img_size = (100, 100)
    markersizes = _get_good_markersizes(sizes, img_size, density=0.5)
    print(markersizes)
    assert markersizes[0] == 5
    assert markersizes[1] == 7
    print("All tests passed!")

    
def add_alpha(colors, alpha):
    """
    Add an alpha channel to the colors.
    :param colors: M x 3 array of colors
    :param alpha: float in [0, 1]
    :return: M x 4 array of colors
    """
    return np.concatenate([colors, np.ones((colors.shape[0], 1)) * alpha], axis=1)



def plot_clustering(ax, points, colors, cluster_ids, image_size, alpha=.5):
    """
    Plot the points colored by cluster.
    :param ax: matplotlib axis object
    :param points: N x 2 array of points
    :param colors: M x 3 array of colors
    :param cluster_ids: N array of integers in [0, M-1]
    """
    id_list = np.unique(cluster_ids)
    cluster_sizes = [np.sum(cluster_ids == i) for i in id_list]
    point_sizes = _get_good_markersizes(cluster_sizes, image_size)
    for i in id_list:
        cluster_points = points[cluster_ids == i]
        #print("Plotting cluster %i with %i points, markersize %i"%(i, len(cluster_points), point_sizes[i]))
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], s=point_sizes[i], alpha=alpha)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_eigenvecs(fig,axes, vecs, n_max, k, colors=None, *args, **kwargs):
    """
    Plot the first n_max eigenvectors aranged vertically.
    Draw a red line between plots k and k+1.
    :param axes: list of n_max axes objects
    :param vecs: eigenvectors, N x N matrix
    :param n_max: number of eigenvectors to plot
    :param k: draw a line after this many plots 
    :param colors: dictionary with:
        'colors': list of M colors [r, g, b] in [0, 255]
        'ids': list of N integers in [0, M-1]
        If this is present, draw component j of each eigenvector in colors['colors'][colors['ids'][j]]
    """
    if axes is None or fig is None:
        fig, axes = plt.subplots(n_max, 1, figsize=(5, 5))
    
    fixed_colors = [c/255. for c in colors['colors']]

    comps_by_color = {i: np.where(colors['ids'] == i)[0] for i in range(len(colors['colors']))}


    for i in range(n_max):
        if colors is None:
            axes[i].plot(vecs[:, i])
        else:
            for c_i, color in enumerate(fixed_colors):
                # if points are ever out of order (i.e. not contiguous in color), this will look messed up, use 'o' or '.' instead then.
                axes[i].plot(comps_by_color[c_i], vecs[:, i][comps_by_color[c_i]],color=color, *args, **kwargs)

        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].tick_params(axis='y', labelsize=6)
        axes[i].set_ylabel("%i" % i, fontsize=8)
        for pos in ['right', 'top', 'bottom', 'left']: 
            axes[i].spines[pos].set_visible(False)
    fig.suptitle('Eigenvectors')
    fig.tight_layout(rect=[0, 0, 1, 1])

    # Draw lines between plots k-1 and k
    if k > 0 and k < n_max:
        above_bbox = axes[k-1].get_position()
        below_bbox = axes[k].get_position()
        line_y = (above_bbox.y1 + below_bbox.y0) / 2
        fig.add_artist(plt.Line2D((0, 1), (line_y, line_y), color='red', linewidth=1))
    

def test_plot_eigenvecs():
    from colors import COLORS
    data = make_data(100)
    colors = [COLORS['red']/255.,
              COLORS['green']/255.,
              COLORS['blue']/255.]
    ids = np.zeros(100, dtype=np.int32)
    ids[34:67] = 1
    ids[67:] = 2
    fig, axes = plt.subplots(8, 1, figsize=(5, 5))
    plot_eigenvecs(fig,axes,data, 8, 3  , colors={'colors': colors, 'ids': ids})
    plt.show()

def test_plot_clustering():
    cluster_size_range = (2, 600)
    #cluster_sizes = [1, 2, 5, 10, 100, 500, 1000, 10000]
    n_clusters = 10#len(cluster_sizes)  #8

    points,ids = [],[]
    for c_id in range(n_clusters):
        n = np.random.randint(*cluster_size_range) # cluster_sizes[c_id]
        center = np.random.randn(2)*7
        sigma = np.random.rand(1)*2
        points.append(np.random.randn(n, 2)*sigma + center)
        ids.append(np.ones(n, dtype=np.int32)*c_id)
    colors = get_n_disp_colors(n_clusters) / 255.
    points = np.concatenate(points)
    ids = np.concatenate(ids)
    print(points.shape, ids.shape, colors.shape)
    print(colors)
    print(ids)
    print(points)
    fig_size = (5,5)
    dpi = 100
    _, axes = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
    img_size = (fig_size[0]*dpi, fig_size[1]*dpi)

    plot_clustering(axes,points, colors, ids,img_size)
    plt.title("Clustering")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test_plot_eigenvecs()
    test_plot_clustering()
    #test_get_good_markersizes()
    print("All tests passed!")