import cv2
import numpy as np
import logging
import os


def tk_color_from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
       https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def get_clobber_free_filename(filename, clobber = False):
    """
    Append _1, _2, etc. to the filename if it already exists, before the extension.
    """
    if clobber:
        return filename
    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(filename):
        filename = f"{base}_{i}{ext}"
        i += 1
    return filename

def get_annulus_polyline(r_outer, r_inner, n_points=50):
    """
    Return a list of points that form a polyline of an annulus.
    :param r_outer: outer radius
    :param r_inner: inner radius
    :param n_points: number of points
    :return: list of (x, y) points
    """
    # left-right symmetry w/even number of points:
    n_points = n_points + 1 if (n_points % 2) == 1 else n_points

    angles = np.linspace(0, 2*np.pi, n_points+1)
    x_outer = r_outer*np.cos(angles)
    y_outer = r_outer*np.sin(angles)
    x_inner = r_inner*np.cos(angles[::-1])
    y_inner = r_inner*np.sin(angles[::-1])

    ring_x = np.concatenate([x_outer, x_inner[::-1]])
    ring_y = np.concatenate([y_outer, y_inner[::-1]])
    return list(zip(ring_x, ring_y))

def float_rgb_to_int(rgb):  
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def frame_bbox_from_rel(window_size, rel_bbox):
    """
    Convert a relative bounding box to an absolute one.
    :param window_size: tuple (width, height) of the window
    :param rel_bbox: dict(x=(x0, x1), y=(y0, y1)) with relative coordinates (0-1)
    :return: dict(x=(x0, x1), y=(y0, y1)) with absolute coordinates (ints)
    """
    abs_bbox = {}
    abs_bbox['x'] = (int(rel_bbox['x'][0] * window_size[0]), int(rel_bbox['x'][1] * window_size[0]))
    abs_bbox['y'] = (int(rel_bbox['y'][0] * window_size[1]), int(rel_bbox['y'][1] * window_size[1]))
    return abs_bbox


def show_annulus():
    SHIFT = 6
    SHIFT_M = 1 << SHIFT
    points = get_annulus_polyline(20, 15, 100)
    img = np.zeros((50, 50, 3), dtype=np.uint8) + 255
    offset = np.array([25, 25])
    points = np.array((points + offset.T) * SHIFT_M, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color=(255, 0, 0), lineType=cv2.LINE_AA, shift=SHIFT)

    img_r = img.copy()
    img_r[img[:, :, 0] == 255] = [0, 0, 255]

    img_both = (img + img_r[:, ::-1, :]) // 2
    cv2.imshow('Annulus w/mirror image.', img_both)
    cv2.waitKey(0)

    cv2.imshow('Annulus', img)
    cv2.waitKey(0)


def calc_font_size(lines, bbox, font, item_spacing_px, n_extra_v_spaces=0, max_font_size=3.0):
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
    font_sizes = np.linspace(.2, max_font_size, 100)[::-1]

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

def get_font_scale(font, max_height, max_width=None, incl_baseline=False, text_lines = None):
    """
    Find the maximum font scale that fits a number in the given height.
    :param font_name: Name of the font to use.
    :param max_height: Maximum height of the text.
    :return: The maximum font scale that fits the text in the given height.
    """
    text_lines = ["0"] if text_lines is None else text_lines

    scale = 5
    while True:

        def _check(test_text):
            
            (text_width, text_height), baseline = cv2.getTextSize(test_text, font, scale, 1)
            text_height = text_height + baseline if incl_baseline else text_height

            #print("Text scales vs maximums: Scale:  %.3f, needs %s <= %s, %s <= %s" % (scale,text_width, max_width, text_height, max_height))
            if (text_height <= max_height) and (max_width is None or (text_width <= max_width)):
                return True
            return False
        
        scores = [_check(line) for line in text_lines]
        if all(scores):
            break

        scale -= 0.01

    return scale


def write_lines_in_bbox(img, lines, bbox, font, color, spacing_frac=0.85):
    n_lines = len(lines)
    v_space = bbox['y'][1] - bbox['y'][0]
    h_space = bbox['x'][1] - bbox['x'][0]
    max_h_per_line = int(v_space / n_lines)

    font_scale = get_font_scale(font, max_h_per_line, incl_baseline=True, max_width = h_space,text_lines=lines) * spacing_frac
    y0 = bbox['y'][0] 
    x0 = bbox['x'][0] 
    for i, line in enumerate(lines):
        (width, height), baseline = cv2.getTextSize(line, font, font_scale, 1)
        text_height = height + baseline
        pos = (x0, y0+text_height)
        cv2.putText(img, line, pos, font, font_scale, color, thickness=1, lineType=cv2.LINE_AA)
        y0 +=  max_h_per_line

def test_write_lines_in_bbox():
    lines = ["This is a test",
             "of the emergency",
             "broadcast system.",
             "This is only a test but it has a long line."]
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    bbox = {'x': (10, 390), 'y': (10, 190)} 
    font = cv2.FONT_HERSHEY_SIMPLEX
    write_lines_in_bbox(img, lines, bbox, font, (255, 255, 255), spacing_frac=0.85,)
    cv2.imshow("Test", img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    test_write_lines_in_bbox()
