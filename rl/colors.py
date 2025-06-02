_MPL_BLUE_RGB = 31, 119, 180    # X
_MPL_ORANGE_RGB = 255, 127, 14  # O
_MPL_GREEN_RGB = 44, 160, 44    # Draw
_OFF_WHITE_RGB = 246, 238, 227  # Background
_DARK_NAVY_RGB = 0, 4, 51       # Lines
_DARK_RED_RGB = 179, 25, 66
_GREEN = 0, 255, 0
_RED = 255, 0, 0
_DARK_GRAY = 100,100,100 # background against state value images
_SKY_BLUE = 135, 206, 235
_NEON_GREEN = 57, 255, 20
_NEON_BLUE = (31, 81, 255)
_NEON_RED = (255, 20, 47)
import numpy as np

# background and line colors
_COLOR_BG = _OFF_WHITE_RGB
_COLOR_LINES = _DARK_NAVY_RGB
_COLOR_TEXT = _DARK_NAVY_RGB
_COLOR_URGENT = _DARK_RED_RGB 
_COLOR_FUNC_BG = _SKY_BLUE

# UI colors
UI_COLORS = {'selected': _NEON_RED,
             'mouseovered': _NEON_GREEN,
             'current_state': _NEON_GREEN,}

COLOR_SCHEME = {'bg':  _COLOR_BG,
                'lines': _COLOR_LINES,
                'text': _COLOR_TEXT,
                'urgent': _COLOR_URGENT,
                'highlight': _NEON_BLUE,
                'func_bg': _COLOR_FUNC_BG,
                'color_x': _MPL_BLUE_RGB,
                'color_o': _MPL_ORANGE_RGB,
                'color_draw': _MPL_GREEN_RGB,
                'color_draw_shade': _GREEN,  # average draw-states w/this color
                'color_shade': _DARK_GRAY,  # average nonterminal states w/this color
                }


MPL_CYCLE_COLORS = [(31, 119, 180),
                    (255, 127, 14),
                    (214, 39, 40),
                    (148, 103, 189),
                    (227, 119, 194), 
                    (150,75,0), 
                    (188, 189, 34),
                    (150, 150, 150),
                    (44, 160, 44)]


def get_n_colors(n):
    """
    Get up to 9 very different, saturated colors.
    :param n:  The number of colors to get.
    :return:  List of RGB tuples.
    """
    return MPL_CYCLE_COLORS[:n]

def shade_color(color, n_shades):
    color = np.array(color)
    if n_shades == 1:
        return color.reshape(1, 3)
    shades = np.linspace(np.zeros(3), color, n_shades+2)[2:]
    if np.all(color==color.max()):
        # if grayscale, reverse the shades
        shades = shades[::-1]
    shades = shades.astype(int)

    return shades

def test_colors():
    import cv2
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = _MPL_BLUE_RGB
    colors = get_n_colors(9)

    for c in range(9):
        n_shades = c+1

        shades = shade_color(colors[c], n_shades)
        print("shades", shades)
        for s in range(n_shades):
            img[s*50:(s+1)*50, c*50:(c+1)*50, :] = shades[s]
    img[450:500, 0:50, :] = _NEON_GREEN
    img[450:500, 50:100, :] = _NEON_BLUE
    img[450:500, 100:150, :] = _NEON_RED
    img[450:500, 150:200, :] = _COLOR_X

    cv2.imshow("Color Test", img[:,:, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    test_colors()
    # test_colors()
