import cv2
import numpy as np
import logging

def get_annulus_polyline(r_outer, r_inner, n_points=50):
    """
    Return a list of points that form a polyline of an annulus.
    :param r_outer: outer radius
    :param r_inner: inner radius
    :param n_points: number of points
    :return: list of (x, y) points
    """
    # left-right symmetry w/even number of points:
    n_points = n_points +1 if (n_points % 2) == 1 else n_points  

    angles = np.linspace(0, 2*np.pi, n_points+1)
    x_outer = r_outer*np.cos(angles)
    y_outer = r_outer*np.sin(angles)
    x_inner = r_inner*np.cos(angles[::-1])
    y_inner = r_inner*np.sin(angles[::-1])
    
    ring_x = np.concatenate([x_outer, x_inner[::-1]])
    ring_y = np.concatenate([y_outer, y_inner[::-1]])
    return list(zip(ring_x, ring_y))

                
def show_annulus():
    SHIFT = 6
    SHIFT_M = 1 << SHIFT
    points = get_annulus_polyline(20, 15, 100)
    img = np.zeros((50, 50, 3), dtype=np.uint8) + 255
    offset = np.array([25, 25])
    points = np.array((points + offset.T) *SHIFT_M , dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color=(255, 0, 0), lineType=cv2.LINE_AA, shift=SHIFT)

    img_r = img.copy()
    img_r[img[:,:,0] == 255] = [0, 0, 255]

    img_both = (img + img_r[:,::-1,:]) // 2
    cv2.imshow('Annulus w/mirror image.', img_both)
    cv2.waitKey (0)

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


if __name__ == '__main__':
    show_annulus()

