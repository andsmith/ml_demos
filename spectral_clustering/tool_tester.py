"""
Lightweight interface with tools.
"""

import cv2
import numpy as np
from toolbar import RadioButtons
from windows import COLORS

class ToolTester(object):
    """
    Test a tool.
    """

    def __init__(self, tool_class, tool_params, img_size = (640, 480)):
        """
        Create a tool and test it.
        """
        self._img_size = img_size
        self._tool = tool_class(**tool_params)
        self._test()

    def _test(self):
        """
        Test the tool.
        """

        blank = np.zeros((500, 500, 3), dtype=np.uint8) + COLORS['white']
        #draw_color = COLORS['black'].tolist()
        cv2.namedWindow("Test")
        cv2.setMouseCallback("Test", self._mouse_callback)
        while True:
            img = blank.copy()
            self._tool.render(img)
            cv2.imshow("Test", img)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            
        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tool.mouse_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._tool.mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._tool.mouse_unclick(x, y)
    

def test_radio():
    params = {'title': "Cluster Type",
              'options': ['Elliptical', 'Gaussian', 'Annular', 'taco','nacho','burrito supreme'],
              'default_selection': 1,
              'bbox': {'x': [10, 220], 'y': [10, 220]}}
    
    tt = ToolTester(RadioButtons, tool_params=params)


if __name__ == "__main__":
    test_radio()
