"""
When user mouses over a state on the value function representation, 
the state key shows the game board/state and prints the 
value of the square under the mouse.
"""

import numpy as np
import cv2
from drawing import GameStateArtist
from tic_tac_toe import Game, Mark, Result
from gui_base import Key
import logging

class StateKey(Key):
    def __init__(self,  size, x_offset=0):
        super().__init__(size, x_offset)
        self._state_artist = None
        self._indent = None
        height = size[1]
        space_size = height//4
        self._state_artist = GameStateArtist(space_size=space_size)
        self._icon_size = self._state_artist.dims['img_size']
        if self._icon_size > height:
            raise ValueError("Use a smaller space_size")

        logging.info("StateKey (%i x %i) initialized with icon size %d  (X offset:  %i)", size[0], size[1],self._icon_size, x_offset)

    def draw(self, img, indicate_value=None):
        """
        Draw the game state in the state key.
        :param img:  The image to draw on.
        :param indicate_value:  A Game state object or None.
        :param pos:  Where in the image to draw the state key.
        """
        if indicate_value is None:
            return
        state_img = self._state_artist.get_image(indicate_value)
        pos= self._get_draw_pos(img, center_rect=(self._icon_size, self._icon_size))
         
        img[pos[1]:pos[1]+state_img.shape[0], pos[0]:pos[0]+state_img.shape[1]] = state_img
        self.draw_bbox(img, color=0)




def test_state_key():
    size = (170, 70)
    #size = (170, 150)

    from colors import COLOR_SCHEME

    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = COLOR_SCHEME['bg']

    sk = StateKey(size)

    game_state =Game.from_strs(["XOX", "OOX", "XOO"])

    sk.draw(img,game_state)
    
    cv2.imshow("State Key", img[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_state_key()
