
from tic_tac_toe import Game
from game_base import Mark, Result
import numpy as np
import cv2
from colors import COLOR_SCHEME
from drawing import GameStateArtist

import matplotlib.pyplot as plt
def get_test_state():
    chars = ["X", "O", " "]
    rows = ["".join(np.random.choice(chars, 3)) for _ in range(3)]
    return Game.from_strs(rows)


def test_grid_sizes():
    test_size_px = 300  # box this wide
    #    space_sizes = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    space_sizes = np.array([[11,12, 13, 14], [15,16,17,18]  , [19, 20,  22, 25], [30, 35, 40,50]])

    # space_sizes = np.array([[7,8,6],[3,2,4],[6,2,1]])
    img_h_px = test_size_px * space_sizes.shape[1]
    img_w_px = test_size_px * space_sizes.shape[0]
    img = np.zeros((img_h_px, img_w_px, 3), dtype=np.uint8)
    img[:, :] = COLOR_SCHEME['bg']
    # Fill each grid with as many random game states as will fit, in rows and columns
    for test_row in range(space_sizes.shape[0]):
        y_top = test_row * test_size_px
        for test_col in range(space_sizes.shape[1]):
            x_left = test_col * test_size_px

            space_size = space_sizes[test_row, test_col]

            artist = GameStateArtist(space_size=space_size)
            g_size = artist.dims['img_size']
            g_pad = int(g_size * .2)

            n_g_rows = test_size_px // (g_size + g_pad)
            n_g_cols = test_size_px // (g_size + g_pad)
            for r in range(n_g_rows):
                for c in range(n_g_cols):
                    game = get_test_state()
                    state_img = artist.get_image(game)
                    x = x_left + c * (g_size + g_pad) + g_pad//2
                    y = y_top + r * (g_size + g_pad) + g_pad//2

                    img[y:y + g_size, x:x + g_size] = state_img
    cv2.imshow("Test Grid Sizes", img[:, :, ::-1])  # BGR to RGB
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_random_act_distribution(uniform=False):
    actions = [(row, col) for row in range(3) for col in range(3)]

    n_chosen = np.random.randint(1, len(actions))  # choose between 1 and 9 actions
    chosen_action_inds = np.random.choice(len(actions), n_chosen, replace=False)
    chosen_actions = [actions[i] for i in chosen_action_inds]
    probs = np.random.rand(n_chosen) if not uniform else np.ones(n_chosen)
    probs /= np.sum(probs)  # normalize to sum to 1
    action_dist = [(action, prob) for action, prob in zip(chosen_actions, probs)]
    return action_dist, np.random.choice([Mark.X, Mark.O])


def test_action_distributions():
    test_size_px = 280  # box this wide
    space_sizes = np.array([[2,5, 8, 13], [15,16,17,18]  , [19, 20,  22, 25], [30, 35, 40,50]])
    # space_sizes = np.array([[7,8,6],[3,2,4],[6,2,1]])
    img_h_px = test_size_px * space_sizes.shape[0]
    img_w_px = test_size_px * space_sizes.shape[1]
    img = np.zeros((img_h_px, img_w_px, 3), dtype=np.uint8)
    img[:, :] = COLOR_SCHEME['bg']
    cmap = plt.get_cmap('gray')  # Use a colormap for the action distributions
    # Fill each grid with as many random game states as will fit, in rows and columns
    for test_row in range(space_sizes.shape[0]):
        y_top = test_row * test_size_px
        for test_col in range(space_sizes.shape[1]):
            x_left = test_col * test_size_px

            space_size = space_sizes[test_row, test_col]

            artist = GameStateArtist(space_size=space_size)
            g_size = artist.dims['img_size']

            print("Space size: ", space_size)
            g_pad = int(g_size * .2)

            n_g_rows = test_size_px // (g_size + g_pad)
            n_g_cols = test_size_px // (g_size + g_pad)
            for r in range(n_g_rows):
                for c in range(n_g_cols):
                    action_dist, mark = get_random_act_distribution(uniform=True)
                    hc=None
                    if space_size >= 11 and np.random.rand()<.5:
                        hc = np.random.choice(len(action_dist))  # highlight a random action
                    state_img, _ = artist.get_action_dist_image(action_dist,mark, cmap,highlight_choice = hc)
                    x = x_left + c * (g_size + g_pad) + g_pad//2
                    y = y_top + r * (g_size + g_pad) + g_pad//2

                    img[y:y + g_size, x:x + g_size] = state_img
    cv2.imshow("Test Grid Sizes", img[:, :, ::-1])  # BGR to RGB
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    test_grid_sizes()
    test_action_distributions()
