import numpy as np
import cv2
import logging


def scale_y_value(value, value_span, y_span, flip_y=True):
    """
    Map logical range to pixel range.
    :param value: The value to scale.
    :param value_span: The range of values to scale from (min, max).
    :param y_span: The range of pixels to scale to (min, max).
    :param flip_y: If True, the y-axis is flipped (higher at the top, decreasing downwards).
    """
    val_norm = (value - value_span[0]) / (value_span[1] - value_span[0])
    if flip_y:
        return (y_span[0] + val_norm * (y_span[1] - y_span[0]))
    return (y_span[1] - val_norm * (y_span[1] - y_span[0])) + y_span[0]


def scale_counts(all_counts, line_t_range, alpha_range, alpha_adjust=1.0):
    """
    Lines representing counts are to be drawn.
    Scale thickness linearly between the given range.

    Adjust alpha according to the number of lines, counts, etc.,
    to best show the general trends.

    :param all_counts: The counts to scale.
    :param line_t_range: The range of line thicknesses to scale to. 
    :param alpha_range: The range of alpha values to scale to.
        (default is (0.05, 0.95), meaning scaled counts with 1 unique 
        value is mostly opaque, etc.)
    :param alpha_adjust: A factor to adjust the alpha values:


    :return: A dict with keys 't' for thicknesses and 'alpha' for alphas.

    """
    rel_counts = all_counts / np.max(all_counts)
    # Thicknesses:
    ltr = line_t_range
    t_scaled_counts = (rel_counts * (ltr[1] - ltr[0]) + ltr[0]).astype(int)
    t_scaled_counts = np.clip(t_scaled_counts, ltr[0], ltr[1])

    # Alphas:  (decay from opaque to a lower alpha bound, by number of lines)
    n_lines = len(all_counts)
    alpha_span = alpha_range[1] - alpha_range[0]
    alpha_unscaled = 1.0 / (n_lines)
    alpha = alpha_range[0] + alpha_span * alpha_unscaled**alpha_adjust

    return {'t': t_scaled_counts,
            'alpha': alpha}


def test_scale_counts():
    """
    Test the scaling of counts to thicknesses and alphas.
    """
    import matplotlib.pyplot as plt
    counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 150, 250, 1000, 2000, 5000])
    line_t_range = (1, 40)
    alpha_range = (0.1, .9)

    alphas = []
    thicks = []
    n_pts = 2000
    for test_count in counts:
        test_counts = np.ones(test_count) * (n_pts // test_count)
        scales = scale_counts(test_counts, line_t_range, alpha_range=alpha_range)
        alphas.append(scales['alpha'])
    alphas = np.array(alphas)
    print(alphas)
    plt.figure()
    print(counts.shape, alphas.shape)
    plt.bar(x=range(counts.size), height=alphas)
    plt.title('Scaled Alphas')

    plt.show()


def calc_alpha_adjust(n_lines, y_span, line_t, alpha_range):
    """
    Given how many lines there are, the span, and their thickness,
    how transparent should they be?
    """
    est_coverage = n_lines * line_t / y_span
    est_alpha_raw = 1.0 / (est_coverage*2)
    est_alpha = np.clip(est_alpha_raw, alpha_range[0], alpha_range[1])
    return est_alpha


def calc_tick_placement(px_range, val_range, min_minor_spacing_px=20):
    """
    Find the number of ticks that fit in the given pixel range, and their placement.

    for a given span, find the most significant decimal place that is changing, 
      place the first tick at this point of change.
    Set the major interval to whatever power of 10 does not exceed the px_per_tick.
    Set the minor interval to 1/10th of the major interval.


    :param px_range: tuple (px_min, px_max) Where the axis will span.
    :param val_range: tuple (val_min, val_max) The range of values to display.
    :param px_per_tick: The number of pixels per tick (approximately).
    :return: {'major': [{'pos': float,  # pixel position of the tick
                         'value': float, # value of the tick
                         'text': str, # text to display at the tick
                         }],
                'minor':  (... similar ...)
                'font_scale': float,  # scale to use for the text
                }
    """
    px_min, px_max = px_range
    val_min, val_max = val_range

    if px_max <= px_min or val_max <= val_min:
        raise ValueError("Invalid pixel or value range.")

    # Calculate the number of pixels per value unit
    px_per_val = (px_max - px_min) / (val_max - val_min)

    # Determine how many ticks we can fit at most:
    n_minor_ticks = (px_max - px_min) / min_minor_spacing_px

    # What power of 10 are we spanning?
    pow_change = np.floor(np.log10(np.abs((val_max - val_min))))
    # Add 2 to overestimate the magnitude change, shrink until we have enough ticks.
    pow_change += 2

    maj_range = 10 ** pow_change
    min_range = maj_range / 10
    # how many of these powers fit in our value range?
    n_maj = ((val_max - val_min) / maj_range)
    n_minor_mags = ((val_max - val_min) / min_range)
    while n_minor_mags < n_minor_ticks:
        pow_change -= 1
        maj_range = 10 ** (2 + pow_change)
        min_range = maj_range / 10
        n_mags = (val_max - val_min) / maj_range
        n_minor_mags = (val_max - val_min) / min_range

    # Between x and x + 10^i where i is an integer, there is
    # a number whose highest power of 10 that changes is 10^i.
    # We can place a tick at this point, and then space them evenly, etc.

    def truncate_10s_place(values, factor):
        return np.floor(values / factor) * factor

    minor_tick_vals = truncate_10s_place(np.arange(val_min, val_max+min_range, min_range), min_range)

    def _find_majors(minor_vals):

        # most significant power of 10 that changes.
        change_places = [np.floor(np.log10(np.abs(val))) for val in minor_vals if val != 0]
        maj_inds = np.argmax(change_places)
        majors = np.where(change_places == change_places[maj_inds])[0]
        if 0.0 in minor_vals:
            # If 0 is in the minor ticks, we need to add it as a major tick.
            majors = np.append(majors, np.where(minor_vals == 0.0)[0][0])
        return majors
    maj_inds = _find_majors(minor_tick_vals)

    ticks = {'major': [], 'minor': []}
    # print("Ticks for values in range (%0.5f, %0.5f) px in range (%0.2f, %0.2f):" % (val_min, val_max, px_min, px_max))
    for i, minor_val in enumerate(minor_tick_vals):
        # Create the tick.
        pixel_pos = px_min + (minor_val - val_min) * px_per_val
        tick = {'pos': pixel_pos, 'value': minor_val, 'text': f"{minor_val:.2f}"}
        if i in maj_inds:
            ticks['major'].append(tick)
        else:
            ticks['minor'].append(tick)
        # print("\tTick: %0.5f  ->  %.2f px%s" % (minor_val, pixel_pos, (" (major)" if i in maj_inds else "")))
    # print("\n")
    ticks['font_scale'] = 0.8

    return ticks


def get_frame(size):
    tests = [(-1.0, 1.0),
             (0.0, 1.0),
             (0.0123, 0.0156),
             (0.0123, 0.0123456),
             (-1.0, 112.4234),
             (-1.0, -0.01),]
    n_tests = len(tests)
    x_pad = 20

    test_w = (size[0] - x_pad * (n_tests - 1)) // n_tests
    test_size = (test_w, size[1])

    img_size = (test_size[1], n_tests * test_size[0] + (n_tests-1)*x_pad, 3)
    img = np.zeros(img_size, dtype=np.uint8)
    x_offset = 0
    for i, (val_min, val_max) in enumerate(tests):
        # Place show each test vertically, all next to each other.
        px_min, px_max = 10, test_size[1]-10
        test_img = np.zeros((test_size[1], test_size[0], 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)  # white background
        ticks = calc_tick_placement((px_min, px_max), (val_min, val_max), 50)

        for tick in ticks['major']:
            pos_y = int(tick['pos'])
            x0 = 20
            x1 = test_size[0] - 20
            (w, h), _ = cv2.getTextSize(tick['text'], cv2.FONT_HERSHEY_SIMPLEX, ticks['font_scale'], thickness=1)
            pos_y = pos_y
            cv2.putText(test_img, tick['text'], (x0 + 2, pos_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        ticks['font_scale'], (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            x0 += w + 10

            cv2.line(test_img,  (x0, pos_y), (x1, pos_y), (0, 0, 0), 2)

        for tick in ticks['minor']:
            pos_px = int(tick['pos'])
            x0 = 20

            (w, h), _ = cv2.getTextSize(tick['text'], cv2.FONT_HERSHEY_SIMPLEX, ticks['font_scale'], thickness=1)
            pos_y = pos_y + h // 2

            cv2.putText(test_img, tick['text'], (x0 + 2, pos_px + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        ticks['font_scale'], (150, 150, 150), thickness=1, lineType=cv2.LINE_AA)
            x0 += w + 10

            cv2.line(test_img, (x0, pos_px), (x1, pos_px), (150, 150, 150), 1)

        # Place the test image in the main image
        img[:, x_offset:x_offset + test_size[0], :] = test_img
        x_offset += test_size[0] + x_pad

    return img


def test_calc_tick_placement():
    size = (800, 600)
    img = get_frame(size)
    cv2.imshow("Tick Placement Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_alpha_line(img, x_range, y_range, color, thickness=1, alpha=0.5, separate_y=0):
    # return bottom y coordinate drawn
    # TODO: jax, c, etc. implementation.
    half_thick, _ = divmod(thickness, 2)
    # import ipdb; ipdb.set_trace()
    x_px = np.arange(x_range[0], x_range[1]+1, dtype=np.int32)
    y_px_upper = np.linspace(y_range[0], y_range[1], len(x_px), dtype=np.int32) - half_thick - separate_y//2
    adjusted_thickness = thickness - separate_y
    thick = np.arange(adjusted_thickness)
    y_px_grid = y_px_upper[:, None] + thick[None, :]

    x_px_grid = np.tile(x_px.reshape(-1, 1), (1, adjusted_thickness))  # Repeat x coordinates for each thickness level
    y_px = y_px_grid.flatten()  # Flatten the y coordinates to match the x coordinates
    x_px = x_px_grid.flatten()  # Flatten the x coordinates to match the y coordinates
    if color is None:
        pixels = img[(y_px, x_px)]
        pixels = (pixels * (1.0-alpha)).astype(np.uint8)  # Apply alpha to the existing pixels
        img[(y_px, x_px)] = pixels  # Update the image with the new pixel values
    else:
        color = (np.array(color) * alpha).astype(np.uint8)  # Apply alpha to the color
        new_pixels = (1 - alpha) * img[(y_px, x_px)] + alpha * color
        img[(y_px, x_px)] = new_pixels  # Update the image with the new pixel values

    return y_px.max()


def test_draw_alpha_line():
    """
    Draw a large number of horizontal/diagonal lines from left to right of varying thicknesses and alphas.
    """
    from colors import COLOR_SCHEME
    import time
    img_size = (800, 800)
    bkg_color = COLOR_SCHEME['bg']
    line_color = COLOR_SCHEME['lines']
    img1 = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    img1[:] = bkg_color  # Fill the image with the background color

    img2 = img1.copy()

    margin = 100
    x = (margin, img_size[0] - margin)
    y = (margin, img_size[1] - margin)

    thickness_range = (1, 100)
    alpha_range = (0.1, .30)
    color = line_color

    n_lines = 30
    cv2_time, self_time = 0.0, 0.0
    for _ in range(n_lines):
        thickness = np.random.randint(thickness_range[0], thickness_range[1])
        alpha = np.random.uniform(alpha_range[0], alpha_range[1]) / (n_lines/50)
        y0 = np.random.randint(y[0], y[1])
        y1 = np.random.randint(y[0], y[1])

        # Draw the line with alpha blending

        # CV2 implementation:
        t0 = time.perf_counter()
        overlay = img1.copy()
        cv2.line(overlay, (x[0], y0), (x[1], y1), line_color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img1, 1 - alpha, 0, img1)
        cv2_time += time.perf_counter() - t0

        # Custom implementation:
        t0 = time.perf_counter()
        draw_alpha_line(img2, x, (y0, y1), None, thickness, alpha)
        self_time += time.perf_counter() - t0

    logging.info(f"Drawn {n_lines} lines:")
    logging.info(f"\tOpenCV time: {cv2_time:.4f} seconds")
    logging.info(f"\tCustom time: {self_time:.4f} seconds")

    # Draw a border around the image
    img = np.concatenate((img1, img2), axis=1)
    cv2.rectangle(img, (margin, margin), (img_size[0] - margin, img_size[1] - margin), 0, thickness=20)
    cv2.imshow("Alpha Line Test", img[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_scale_counts()
