import numpy as np
import cv2
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

def calc_font_size(lines, bbox, font, item_spacing_px, n_extra_v_spaces=0, search_range=(.1, 10)):
    """
    Calculate the largest font size to fit the text in the bbox.
    :param lines: list of strings, one per line
    :param bbox: dict(x=(x0, x1), y=(y0, y1))
    :param font: cv2 font constant
    :param item_spacing_px: int, vertical spacing between lines, left and right horizontal spacing within box.
    :param n_extra_v_spaces: int, number of extra item_spacing_px to add to the bottom.
    """
    font_sizes = np.linspace(.2, 3.0, 100)[::-1]

    text_area_width = bbox['x'][1] - bbox['x'][0] - item_spacing_px * 2
    text_area_height = bbox['y'][1] - bbox['y'][0] - item_spacing_px * 2 - n_extra_v_spaces * item_spacing_px
    #print("Fitting text in %i x %i"%( text_area_width, text_area_height))
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
    #print("Failed to find good font size, using smallest.")
    return np.min(font_sizes), np.min(font_sizes)//2


def bbox_contains(box, x, y):
    return box['x'][0] <= x <= box['x'][1] and box['y'][0] <= y <= box['y'][1]
