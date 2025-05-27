"""
Create a colorbar in an image.


Show a spectrum with all colors, in order, print axis w/ticks under it. 
"""

import numpy as np
import cv2
import logging
from colors import COLOR_BG, COLOR_LINES
import matplotlib.pyplot as plt


def get_font_scale(font, max_height):
    """
    Find the maximum font scale that fits a number in the given height.
    :param font_name: Name of the font to use.
    :param max_height: Maximum height of the text.
    :return: The maximum font scale that fits the text in the given height.
    """
    scale = 5.0
    while True:
        (_, text_height), _ = cv2.getTextSize('0', font, scale, 1)
        # print("Text height for scale %.2f is %i  (should be under %i)" % (scale, text_height , max_height))
        if text_height < max_height:
            break
        scale -= 0.01
    return scale


class ColorKey(object):
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

    def __init__(self, cmap, range, size, draw_params={}):
        """
        Create a color key object.
        :param size: width, height
        :param cmap: Color map to use for the color key.
        :param range: Range of values for the color key.
        :param draw_params: Parameters for drawing the color key.
        """
        self.size = size
        self.cmap = cmap
        self.range = range
        self.draw_params = {'axis_width_frac': 1./30,
                            'tick_width_frac': 1./35,
                            'tick_height_frac': .2,
                            'dot_size_frac': .02,  # for dot/dashed line of indicator
                            'tick_font': {'big': cv2.FONT_HERSHEY_COMPLEX, 'small': cv2.FONT_HERSHEY_SIMPLEX},
                            'pad_x_frac': .08,  # allow for axis labels to be centered under spectrum endpoints
                            'spacing_y_frac': .15,  # top, axis, ticks, bottom

                            }
        self.draw_params.update(draw_params)

    def _get_font(self, v_space):
        if v_space > 12:
            return self.draw_params['tick_font']['big']
        else:
            return self.draw_params['tick_font']['small']

    def draw(self, img, line_color, text_color, indicate_value=None):
        """
        Draw the color key on the given image in the upper right (TODO: make this configurable).
        :param img: Image to draw on.
        :param line_color: Color of the lines.
        :param text_color: Color of the text.
        :param indicate_value: If given, draw a line at this value, print it below.
        """
        bkg_color = tuple(img[0, 0].tolist())

        left, right = img.shape[1] - self.size[0], img.shape[1]
        top, bottom = 0, self.size[1]
        h = bottom - top
        w = right - left
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


        spectrum_x = (left + space_x, right -space_x)
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
        spec_values = []
        for i in range(spectrum_width):
            value = self.range[0] + (self.range[1] - self.range[0]) * i / (w - 1)
            val_norm = (value - self.range[0]) / (self.range[1] - self.range[0])
            color = np.array(self.cmap(val_norm)[:3])  # Get the color from the colormap
            gradient[:, i] = (color[:3] * 255).astype(np.uint8)
            spec_values.append(value)

        # Place the gradient in the image
        img[spectrum_y[0]:spectrum_y[1], spectrum_x[0]:spectrum_x[1]] = gradient

        # Draw axis lines
        axis_width = max(2, int(self.draw_params['axis_width_frac'] * h))
        img[axis_y: axis_y + axis_width, spectrum_x[0]:spectrum_x[1]] = line_color


        ticks = [self.range[0], 0.0, self.range[1]] if self.range[0] < 0 and self.range[1] > 0 else \
            [self.range[0], (self.range[0]+self.range[1])/2, self.range[1]]
        
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
                img[tick_label_top-1:tick_label_bottom+1, text_x-1:text_x + width+1 ] = bg_color

            cv2.putText(img, string, (text_x, tick_label_bottom),
                        font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            
            return text_x, text_x + width

        for i in range(num_ticks):
            tick_x = int(spectrum_x[0] + (spectrum_x[1] - spectrum_x[0]) * i / (num_ticks - 1))
            tick_left = tick_x - tick_width // 2
            tick_right = tick_left + tick_width
            if i == num_ticks - 1:
                # don't overrun axis
                shift = tick_right - spectrum_x[1]
                tick_left, tick_right = tick_left-shift, tick_right-shift
            elif i == 0:
                shift = tick_left - spectrum_x[0]
                tick_right, tick_left = tick_right-shift, tick_left-shift
            img[tick_top:tick_bottom, tick_left:tick_right] = line_color
            if indicate_value is None:
                label_value = self.range[0] + (self.range[1] - self.range[0]) * i / (num_ticks - 1)
                label_text = f"{label_value:.1f}"
                draw_label_at(label_text, tick_x, text_color)

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
                if (i //  dot_size) % 2 ==1:
                    ind_line[i, :] = black
                else:
                    ind_line[i, :] = white

            # figure out where it goes
            ind_val_norm = float((indicate_value - self.range[0]) / (self.range[1] - self.range[0]))
            ind_x = int(spectrum_width * ind_val_norm) 
            if ind_x <0:
                ind_line_x = spectrum_x[0] - line_w*2  # move it to the left of the spectrum
            elif ind_x > spectrum_width:
                ind_line_x = spectrum_x[1] + line_w*2  # move it to the right of the spectrum
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
            line_x_left, line_x_right = draw_label_at(ind_label_text, spectrum_x[0] + ind_x, color=text_color, bg_color=bkg_color)
            if line_x_left < left + space_x:
                line_x_left += line_w*2
            elif line_x_right > right - space_x:
                line_x_right -=  line_w*2

            line_length = line_x_right - line_x_left
            line_height = line_w  # same as the vertical line width
            # now make a horizontal dashed line above the label connecting to the vertical line
            ind_line = np.zeros((line_height, line_length, 3), dtype=np.uint8)
            # make alternating black and white:
            for i in range(line_length):
                if (i //  dot_size) % 2 ==1:
                    ind_line[:, i] = black
                else:
                    ind_line[:, i] = white
            # place the horizontal line above the label
            img[line_bottom:line_bottom+line_height, line_x_left:line_x_right] = ind_line
        

        return img


def test_color_key():
    """
    Test the ColorKey class.
    """
    box_w = 400
    keys = []
    indicated = [-1.05, -1.0, -0.75, 0.0, 0.5, 1.0, 1.05]
    for i, box_h in enumerate([40, 50, 60, 70, 100, 120, 150]):

        box_size = (box_w, box_h)
        img = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        img[:] = (127, 127, 127)
        cmap = plt.get_cmap('coolwarm')
        range = (-1.0, 1.0)
        ck = ColorKey(size=box_size, cmap=cmap, range=(-1.0, 1.0))

        ck.draw(img, line_color=COLOR_LINES, text_color=COLOR_LINES, indicate_value=indicated[i])
        keys.append(img)
        keys.append(np.zeros((10, box_w, 3), dtype=np.uint8))  # add a spacer between keys

    all_keys_img = np.concatenate(keys, axis=0)

    print("Color key image shape:", all_keys_img.shape)

    # img_big = cv2.resize(img, (box_w*3, box_h*3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Color Key", all_keys_img[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_color_key()
