"""
When user mouses over a state on the value function representation, 
the state key shows the game board/state and prints the 
value of the square under the mouse.
"""

import numpy as np
import cv2
from drawing import GameStateArtist
from tic_tac_toe import Game, Mark, Result

from colors import COLOR_BG
class StateKey(object):
    def __init__(self, size):
        self._size=None
        self._state_artist = None
        self._indent = None
        self.resize(size)

    def resize(self,new_size):
    
        if self._size is None or self._size != new_size:
            height = new_size[1]
            space_size = height//4
            self._state_artist = GameStateArtist(space_size=space_size)
            self._size = new_size
            state_img_side_len = self._state_artist.dims['img_size']
            print("Image size:  %s,  cell_size:  %s" %(state_img_side_len, space_size))
            v_room = height - state_img_side_len
            h_room = new_size[0] - state_img_side_len
            self._indent = (h_room // 2, v_room // 2)

    def draw(self, img, game_state, pos=(0,0)):
        """
        Draw the game state in the state key.
        :param img:  The image to draw on.
        :param game_state:  The game state to draw.
        :param pos:  Where in the image to draw the state key.
        """
        state_img = self._state_artist.get_image(game_state)
        img[pos[1]+self._indent[1]:pos[1]+self._indent[1] + state_img.shape[0],
            pos[0]+self._indent[0]:pos[0]+self._indent[0] + state_img.shape[1]] = state_img




def test_state_key():
    size = (170, 70)
    #size = (170, 150)


    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = COLOR_BG

    sk = StateKey(size)

    game_state =Game.from_strs(["XOX", "OOX", "XOO"])

    sk.draw(img,game_state)
    
    cv2.imshow("State Key", img[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_state_key()
