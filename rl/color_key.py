"""
Create a colorbar in an image.


Show a spectrum with all colors, in order, print axis w/ticks under it. 
"""

import numpy as np
import cv2
import logging
from colors import COLOR_SCHEME
import matplotlib.pyplot as plt
from gui_base import Key
from util import get_font_scale


class ColorKey(Key):
    """

        +---------------------------------+
        |                                 |
        |  |                           |  |
        |  |##########@@@@@@@@@@$$$$$$$|  |
        |  |       .     |      .      |  |
        | -1.0  -.0.5   0.0    0.5     1  |
        |                                 |
        +---------------------------------+

    """

    def __init__(self, size, cmap, value_range, x_offset=None):
        """
        Create a color key object.
        :param size: width, height
        :param cmap: Color map to use for the color key.
        :param range: Range of values for the color key.
        :param draw_params: Parameters for drawing the color key.
        """
        super().__init__(size=size, x_offset=x_offset)
        self.cmap = cmap
        self.range = value_range
        self.draw_params = {'axis_width_frac': 1./30,
                            'tick_width_frac': 1./35,
                            'tick_height_frac': .2,
                            'dot_size_frac': .02,  # for dot/dashed line of indicator
                            'tick_font': {'big': cv2.FONT_HERSHEY_COMPLEX, 'small': cv2.FONT_HERSHEY_SIMPLEX},
                            'pad_x_frac': .08,  # allow for axis labels to be centered under spectrum endpoints
                            'spacing_y_frac': .15,  # top, axis, ticks, bottom
                            'line_color': COLOR_SCHEME['lines'],
                            'text_color': COLOR_SCHEME['text'],
                            }

        logging.info("Initialized ColorKey(%i x %i) with range %s, at x_offset %i",
                     size[0], size[1], str(value_range), self._x_offset)
        # self.draw_params.update(draw_params)

    def _get_font(self, v_space):
        if v_space > 12:
            return self.draw_params['tick_font']['big']
        else:
            return self.draw_params['tick_font']['small']

    def norm_value(self, value):
        return (float(value) - self.range[0]) / (self.range[1] - self.range[0])

    def map_color(self, value):
        val_norm = self.norm_value(value)
        return np.array(self.cmap(val_norm)[:3])  # returns RGBA, we use RGB

    def draw(self, img,  indicate_value=None):
        """
        Draw the color key on the given image in the upper right (TODO: make this configurable).
        :param img: Image to draw on.
        :param line_color: Color of the lines.
        :param text_color: Color of the text.
        :param indicate_value: If given, draw a line at this value, print it below.
        """
        bkg_color = tuple(img[0, 0].tolist())

        left, top = self._get_draw_pos(img)
        right, bottom = left + self.size[0], top + self.size[1]
        w, h = self.size
        space_x = max(2, int(self.draw_params['pad_x_frac'] * w))  # space on left & right of spectrum
        space_y = max(2, int(self.draw_params['spacing_y_frac'] * h))  # between top & spectrum,
        spectrum_space = max(3, int(space_y / 1.5))  # space between axis and spectrum
        tick_height = max(3, int(self.draw_params['tick_height_frac'] * h))
        tick_space = max(2, int(space_y / 3))  # space between ticks and tick labels

        total_v_padding = space_y * 2 + spectrum_space + tick_space
        leftover_v_space = h - total_v_padding - tick_height
        spectrum_height = leftover_v_space // 2
        label_height = spectrum_height

        spectrum_top = top + space_y
        spectrum_bottom = spectrum_top + spectrum_height
        axis_y = spectrum_bottom + spectrum_space

        spectrum_x = (left + space_x, right - space_x)
        spectrum_y = (spectrum_top, spectrum_bottom)

        tick_top = axis_y
        tick_bottom = tick_top + tick_height
        tick_mid = (tick_top + tick_bottom) // 2
        tick_label_top = tick_bottom + tick_space
        tick_label_bottom = bottom - space_y
        tick_label_h = tick_label_bottom - tick_label_top

        font = self._get_font(tick_label_h)

        font_scale = get_font_scale(font, tick_label_h)

        # Create a gradient image
        spectrum_width = spectrum_x[1] - spectrum_x[0]
        gradient = np.zeros((spectrum_height, spectrum_width, 3), dtype=np.uint8)
        spec_values = np.linspace(self.range[0], self.range[1], spectrum_width)
        colors = [self.map_color_uint8(value) for value in spec_values]
        # Fill the gradient with colors
        gradient[:] = colors

        # Place the gradient in the image
        img[spectrum_y[0]:spectrum_y[1], spectrum_x[0]:spectrum_x[1]] = gradient

        # Draw axis lines
        axis_width = max(2, int(self.draw_params['axis_width_frac'] * h))
        img[axis_y: axis_y + axis_width, spectrum_x[0]:spectrum_x[1]] = self.draw_params['line_color']

        middle_tick_val = 0 if (self.range[0] < 0 and self.range[1] > 0) else \
            (self.range[0] + self.range[1]) / 2.0

        ticks = [self.range[0],
                 middle_tick_val,
                 self.range[1]]
        tick_width = axis_width
        num_ticks = len(ticks)

        def draw_label_at(string, x_pos, color, bg_color=None):
            thickness = 1 if font_scale < 2 else 2
            (width, _), _ = cv2.getTextSize(string, font, font_scale, thickness)
            text_x = x_pos - width // 2
            if text_x < left+2:
                text_x = left+2
            elif text_x + width > right-2:
                text_x = right-2 - width
            if bg_color is not None:
                # Expand box a bit
                img[tick_label_top-1:tick_label_bottom+1, text_x-1:text_x + width+1] = bg_color
            cv2.putText(img, string, (text_x, tick_label_bottom),
                        font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            return text_x, width

        tick_dims = []  # (x_pos, width)
        if indicate_value is not None:

            # Write the indicatee value

            # draw a vertical dotted line at the indicated value
            line_h = tick_mid - spectrum_top
            dot_size = max(2, int(self.draw_params['dot_size_frac'] * h))
            line_w = dot_size
            ind_line = np.zeros((line_h, line_w, 3), dtype=np.uint8)

            # make alternating black and white:
            black, white = (0, 0, 0), (255, 255, 255)
            for i in range(line_h):
                if (i // dot_size) % 2 == 1:
                    ind_line[i, :] = black
                else:
                    ind_line[i, :] = white

            # figure out where it goes
            ind_val_norm = float((indicate_value - self.range[0]) / (self.range[1] - self.range[0]))
            ind_x = int(spectrum_width * ind_val_norm)
            if ind_x < 0:
                ind_line_x = spectrum_x[0] - line_w*2  # move it to the left of the spectrum
            elif ind_x > spectrum_width:
                ind_line_x = spectrum_x[1] + line_w*2 - 2  # move it to the right of the spectrum
            else:
                ind_line_x = spectrum_x[0] + ind_x

            # Add the vertical line to the image
            line_bottom = spectrum_y[0] + line_h
            img[spectrum_y[0]:line_bottom, ind_line_x:ind_line_x+line_w] = ind_line

            # write the indicated value below it's spot in the spectrum
            ind_val_norm = (indicate_value - self.range[0]) / (self.range[1] - self.range[0])
            ind_x = int(spectrum_width * ind_val_norm)
            if ind_x < 0:
                ind_x = 0
            elif ind_x > spectrum_width:
                ind_x = spectrum_width - 1

            ind_label_text = f"{indicate_value:.3f}"
            line_x_left, label_width = draw_label_at(
                ind_label_text, spectrum_x[0] + ind_x, color=self.draw_params['text_color'])
            line_x_right = line_x_left + label_width
            if line_x_left < left + space_x:
                line_x_left += line_w*2
            elif line_x_right > right - space_x:
                line_x_right -= line_w*2

            line_length = line_x_right - line_x_left
            line_height = line_w  # same as the vertical line width
            # now make a horizontal dashed line above the label connecting to the vertical line
            ind_line = np.zeros((line_height, line_length, 3), dtype=np.uint8)
            # make alternating black and white:
            for i in range(line_length):
                if (i // dot_size) % 2 == 1:
                    ind_line[:, i] = black
                else:
                    ind_line[:, i] = white
            # place the horizontal line above the label
            img[line_bottom:line_bottom+line_height, line_x_left:line_x_right] = ind_line
            tick_dims.append((line_x_left, label_width  ))

        def check_tick_room(tick_x, label_text):
            width = cv2.getTextSize(label_text, font, font_scale, 1)[0][0]
            tick_left = tick_x - width // 2
            tick_right = tick_left + width
            room = True

            for x_pos, width in tick_dims:
                other_left, other_right = x_pos, x_pos + width
                if (tick_left < other_right and tick_right > other_left) or \
                        (tick_right > other_left and tick_left < other_right):
                    room = False
                    break
            return room

        # And draw the rest of the ticks:        
        for i in range(num_ticks):
            tick_x_pos = int(spectrum_x[0] + (spectrum_x[1] - spectrum_x[0]) * self.norm_value(ticks[i]))
            tick_left = tick_x_pos - tick_width // 2
            tick_right = tick_left + tick_width
            if ticks[i]==self.range[1]:
                # don't overrun axis
                shift = tick_right - spectrum_x[1]
                tick_left, tick_right = tick_left-shift, tick_right-shift
            elif ticks[i]==self.range[0]:
                shift = tick_left - spectrum_x[0]
                tick_right, tick_left = tick_right-shift, tick_left-shift
            img[tick_top:tick_bottom, tick_left:tick_right] = self.draw_params['line_color']

            label_value = ticks[i]
            label_text = f"{label_value:.1f}"
            show_label= check_tick_room(tick_x_pos, label_text)
            if show_label:
                x_pos, width = draw_label_at(label_text, tick_x_pos,  color=self.draw_params['text_color'])
                tick_dims.append((x_pos, width))
            last_tick_x_txt = tick_x_pos

        return img

    def map_color_uint8(self, value):
        color = self.map_color(value)
        return (color * 255).astype(np.uint8)


class ProbabilityColorKey(ColorKey):
    """
    Uses 'gray' colormap, range [0, 1] and inverts rgb values (so dark is 1.0)
    """

    def __init__(self, size, x_offset=None):
        cmap = plt.get_cmap('gray')
        super().__init__(size=size, cmap=cmap, value_range=(0., 1.), x_offset=x_offset)

    def map_color(self, value):
        return super().map_color(1.0 - value)  # invert the color for probabilities


class SelfAdjustingColorKey(ColorKey):
    """
    Range is intially (-1,1):
        - when first value v is set, range changes to (min(-1, v), max(1, v)).
        - When second value w is set (where w!=v) the range changes to (min(v, w), max(v, w)).
        - Subsequent values expand the range.
    """

    def __init__(self, size, cmap, x_offset=None):
        self._n_set = 0

        super().__init__(size=size, cmap=cmap,
                         value_range=(-1.0, 1.0),
                         x_offset=x_offset)

    def set_values(self, values):
        """
        :param values: list of values to set the color key range to.
        :return: True if either the min or max value changed, False otherwise.
        """
        all_values = np.array([self.range[0], self.range[1], np.min(values), np.max(values)])
        new_range = all_values.min(), all_values.max()
        changed = (new_range[0] != self.range[0]) or (new_range[1] != self.range[1])
        self.range = new_range
        return changed


def test_adjusting_color_key():
    key_size = (300, 50)
    img_size = (640, 480)
    # Create a background image
    bkg = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    x_lims = [100, img_size[0]-100]
    y_lims = [200, 300]
    bar_color = COLOR_SCHEME['lines']
    bkg[:] = COLOR_SCHEME['bg']

    key_img = np.zeros((key_size[1], key_size[0], 3), dtype=np.uint8)

    mouse_val = [None]

    x_val_range = [-10, 10]

    def _x_val(x_pos, y_pos):
        if y_lims[0] > y_pos or y_pos > y_lims[1]:
            return None
        if x_pos < x_lims[0] or x_pos > x_lims[1]:
            return None
        return (x_pos - x_lims[0]) / (x_lims[1] - x_lims[0]) * (x_val_range[1] - x_val_range[0]) + x_val_range[0]

    def _x_pos(x_val):
        return int((x_val - x_val_range[0]) / (x_val_range[1] - x_val_range[0]) * (x_lims[1] - x_lims[0]) + x_lims[0])

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        x_val = _x_val(x, y)
        if x_val is not None:
            mouse_val[0] = (x_val)
        else:
            mouse_val[0] = None
    #  Display an image and a bar, the mouse over the bar sends its x-coordinate to the color key as
    #  a value in the range [-100, 100].
    win_name = "Self Adjusting Color Key Test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, img_size[0], img_size[1])
    cv2.setMouseCallback(win_name, on_mouse)

    sack = SelfAdjustingColorKey(size=key_size, cmap=plt.get_cmap('coolwarm'))

    while True:
        frame = bkg.copy()
        key_img[:] = COLOR_SCHEME['bg']
        sack.draw(key_img, indicate_value=mouse_val[0])
        if mouse_val[0] is not None:
            sack.set_values([mouse_val[0]])
            color = sack.map_color_uint8(mouse_val[0])
        else:
            color = bar_color
        frame[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]] = color
        frame[300:key_size[1]+300, 100:100+key_size[0]] = key_img
        x_minus = int(_x_pos(-1))
        x_plus = int(_x_pos(1))
        frame[y_lims[0]-10:y_lims[0]+10, x_minus] = bar_color
        frame[y_lims[0]-10:y_lims[0]+10,  x_plus] = bar_color

        cv2.imshow(win_name, frame[:, :, ::-1])
        k = cv2.waitKey(1)
        if k == 27:  # ESC key
            break
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()


def test_color_key():
    """
    Test the ColorKey class.
    """
    box_w = 400
    keys = []
    indicated = [-1.05, -1.0, -0.75, None,  0.0, 0.5, 1.0, 1.05]
    for i, box_h in enumerate([40, 50, 60, 70, 70, 100, 120, 150]):
        box_size = (box_w, box_h)
        img = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        img[:] = (127, 127, 127)
        cmap = plt.get_cmap('coolwarm')
        ck = ColorKey(size=box_size, cmap=cmap, value_range=(-1.0, 1.0))

        ck.draw(img, indicate_value=indicated[i])
        keys.append(img)
        keys.append(np.zeros((10, box_w, 3), dtype=np.uint8))  # add a spacer between keys

    all_keys_img = np.concatenate(keys, axis=0)

    print("Color key image shape:", all_keys_img.shape)

    # img_big = cv2.resize(img, (box_w*3, box_h*3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Color Key", all_keys_img[:, :, ::-1])
    cv2.waitKey(0)


def test_probability_color_key():
    """
    Test the ProbabilityColorKey class.
    """
    COLOR_LINES = COLOR_SCHEME['lines']
    COLOR_TEXT = COLOR_SCHEME['text']
    box_w = 300
    box_h = 60
    box_size = (box_w, box_h)
    imgs = []
    img_blank = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    img_blank[:] = (127, 127, 127)

    ck = ProbabilityColorKey(size=box_size)
    img = img_blank.copy()
    imgs.append(ck.draw(img))

    for prob in [0.0, 0.25, 0.1234123512315, .9999, 1.0]:
        print(f"Drawing color key for probability {prob:.4f}")
        img = img_blank.copy()
        imgs.append(ck.draw(img, indicate_value=prob))

    img = np.concatenate(imgs, axis=0)
    cv2.imshow("Probability Color Key", img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_color_key()
    test_probability_color_key()
    test_adjusting_color_key()
