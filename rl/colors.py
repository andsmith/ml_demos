MPL_BLUE_RGB = 31, 119, 180    # X
MPL_ORANGE_RGB = 255, 127, 14  # O
MPL_GREEN_RGB = 44, 160, 44    # Draw
OFF_WHITE_RGB = 246, 238, 227  # Background
DARK_NAVY_RGB = 0, 4, 51       # Lines
GREEN = 0, 255, 0
RED = 255, 0, 0
DARK_GRAY = 100,100,100 # background against state value images

NEON_GREEN = 57, 255, 20
NEON_BLUE = (31, 81, 255)
NEON_RED = (255, 20, 147)
import numpy as np

# background and line colors
COLOR_BG = OFF_WHITE_RGB
COLOR_LINES = DARK_NAVY_RGB
COLOR_TEXT = DARK_NAVY_RGB

# Player (and draw) colors
COLOR_X = MPL_BLUE_RGB
COLOR_O = MPL_ORANGE_RGB
COLOR_DRAW = MPL_GREEN_RGB  # lines
COLOR_DRAW_SHADE = GREEN  # shading
COLOR_SHADE = [0, 0, 0]  # shading

# UI Colors
COLOR_SELECTED = DARK_NAVY_RGB
COLOR_MOUSEOVERED = RED

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
    img[:] = MPL_BLUE_RGB
    colors = get_n_colors(9)

    for c in range(9):
        n_shades = c+1

        shades = shade_color(colors[c], n_shades)
        print("shades", shades)
        for s in range(n_shades):
            img[s*50:(s+1)*50, c*50:(c+1)*50, :] = shades[s]
    img[450:500, 0:50, :] = NEON_GREEN
    img[450:500, 50:100, :] = NEON_BLUE
    img[450:500, 100:150, :] = NEON_RED
    img[450:500, 150:200, :] = COLOR_X

    cv2.imshow("Color Test", img[:,:, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    test_colors()
    # test_colors()
