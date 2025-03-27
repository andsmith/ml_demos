import matplotlib.pyplot as plt

from tic_tac_toe import Game
from game_base import Mark, Result
import numpy as np
import cv2
from colors import COLOR_LINES, COLOR_BG, COLOR_X, COLOR_O, COLOR_DRAW


def get_test_state():
    chars = ["X", "O", " "]
    rows = ["".join(np.random.choice(chars, 3)) for _ in range(3)]
    return Game.from_strs(rows)


def test_grid_sizes():
    test_size_px = 150  # box this wide
    space_sizes = np.array([[1,2,3,4],[5,6,7,8],[10,13,16,20],[25,30,35,40]])
    img_h_px = test_size_px * space_sizes.shape[1]
    img_w_px = test_size_px * space_sizes.shape[0]
    img = np.zeros((img_h_px, img_w_px, 3), dtype=np.uint8)
    img[:, :] = COLOR_BG
    # Fill each grid with as many random game states as will fit, in rows and columns
    for test_row in range(space_sizes.shape[0]):
        y_top = test_row * test_size_px
        for test_col in range(space_sizes.shape[1]):
            x_left = test_col * test_size_px

            space_size = space_sizes[test_row, test_col]
            dims = Game.get_image_dims(space_size)


            g_size = dims['img_size'] 
            g_pad = int(g_size *.2)

            n_g_rows = test_size_px // (g_size + g_pad)
            n_g_cols = test_size_px // (g_size + g_pad)
            print(f"Test({test_row}, {test_col}) will have {n_g_rows} x {n_g_cols} games, each of size {g_size} and cell_size {space_sizes[test_row, test_col]}")
            
            for r in range(n_g_rows):
                for c in range(n_g_cols):
                    game = get_test_state()
                    state_img = game.get_img(dims)
                    x = x_left + c * (g_size + g_pad) + g_pad//2
                    y = y_top + r * (g_size + g_pad) + g_pad//2
                    
                    img[y:y + g_size, x:x + g_size] = state_img
    plt.imshow(img)
    plt.show()   








if __name__ == "__main__":
    test_grid_sizes()