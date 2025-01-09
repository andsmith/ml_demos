"""
Toolbox sub-window of the cluster editor.
"""

import cv2
import numpy as np
from windows import COLORS
from util import bbox_contains


COLOR_OPTIONS = {'unselected': COLORS['black'],
                 'selected': COLORS['black'],
                 'mouseover': COLORS['red']}

MIN_SPACING_PX = 5  # min vertical spacing between things


class RadioButtons(object):
    """
    Looks like this:

        Title
        -------
         * option 1
           option 2
           option 3

    Color indicates slection/mouseover (no checkbox, etc.)

    """

    def __init__(self, bbox, title, options, texts=None, default_selection=None):
        """
        Create list of mutually exclusive items to select.
        """
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._title = title
        self._bbox = bbox
        self._options = options
        self._texts = texts if texts is not None else options
        self._selected_ind = default_selection if default_selection is not None else 0  # currently selected index
        self._mouseover_ind = None  # index of item the mouse is over
        self._colors = {c_opt: COLOR_OPTIONS[c_opt].tolist() for c_opt in COLOR_OPTIONS}

        self._calc_dims()

    def _calc_dims(self):
        """
        Determine the text size, the X,Y positions of each option.
        """
        self._font_size, self._mark_height = self._calc_font_size()
        text_dims = [cv2.getTextSize(text, self._font, self._font_size, 1)  # (width, height), baseline
                     for text in self._texts]
        self._baseline = text_dims[0][1]
        title_dims = cv2.getTextSize(self._title, self._font, self._font_size, 1)[0]

        left = self._bbox['x'][0] + MIN_SPACING_PX
        top = self._bbox['y'][0] + title_dims[1] + MIN_SPACING_PX
        self._title_pos = left, top
        line_length= np.max([int(title_dims[0]* .5), 5])
        top += MIN_SPACING_PX+ self._baseline
        self._line_coords = [(left, top),
                             (left+line_length, top)]
        self._text_pos = []
        # indent for indicator-dot
        self._dot_size = 2
        left += self._dot_size  + MIN_SPACING_PX* 2

        #top += MIN_SPACING_PX
        self._div_lines = [top]   # y-coordinate between each selection (for determining which one the mouse is over)

        for text_dim in text_dims:
            top += text_dim[0][1] + MIN_SPACING_PX +self._baseline
            self._text_pos.append((left, top))
            self._div_lines.append(top+self._baseline)
            

    def _calc_font_size(self):
        """
        Determine the font size that will fit the text in the available space.
        """
        font_sizes = np.linspace(.2, 3.0, 100)[::-1]
        for font_size in font_sizes:
            (title_width, title_height), baseline = cv2.getTextSize(self._title, self._font, font_size, 1)
            header_height = title_height + MIN_SPACING_PX * 2 + 1  + baseline# space, title, space, line

            text_sizes = [cv2.getTextSize(text, self._font, font_size, 1)[0] for text in self._texts]
            text_widths, text_heights = zip(*text_sizes)
            total_height = header_height + np.sum(text_heights) + MIN_SPACING_PX * (len(text_heights) + 1)+ baseline * len(text_heights)
            total_width = max(title_width, max(text_widths)) + MIN_SPACING_PX * 2
            item_height = text_heights[0] 
            if total_height < self._bbox['y'][1] - self._bbox['y'][0] and total_width < self._bbox['x'][1] - self._bbox['x'][0]:

                return font_size, int(item_height/2)

        return 0.1 , .05

    def render(self, img):
        """
        Render the radio buttons.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        cv2.rectangle(img, p0, p1, self._colors['unselected'], 1)
        cv2.putText(img, self._title, self._title_pos, self._font,
                    self._font_size, self._colors['unselected'])
        cv2.line(img, self._line_coords[0], self._line_coords[1], self._colors['unselected'])
        for i, text_pos in enumerate(self._text_pos):
            if i == self._selected_ind:
                color = self._colors['selected']  
                # indicator dot
                dot_y = (self._div_lines[i+1] - self._baseline-self._mark_height ) 
                cv2.circle(img, (self._text_pos[i][0] - MIN_SPACING_PX, dot_y), self._dot_size, color, -1)
            elif i == self._mouseover_ind:
                color = self._colors['mouseover']
            else:
               color=self._colors['unselected']
            cv2.putText(img, self._texts[i], text_pos, self._font, self._font_size, color)

    def _get_item_at(self, y):
        """
        Return the index of the item at the given y-coordinate.
        """
        for i in range(len(self._div_lines)-1):
            if self._div_lines[i] < y < self._div_lines[i+1]:
                return i
        return None
    
    def mouse_click(self, x, y):
        """
        Check if the click is within the radio buttons.
        :
        """
        if not bbox_contains(self._bbox, x, y):
            return
        ind = self._get_item_at(y)
        if ind is not None:
            self._selected_ind = ind


    def mouse_move(self, x, y):
        """
        Check if the mouse is over the radio buttons.
        
        """
        if not bbox_contains(self._bbox, x, y):
            self._mouseover_ind = None
        else:        
            for i, text_pos in enumerate(self._text_pos):
                self._mouseover_ind = None
                item = self._get_item_at(y)
                if item is not None:
                    self._mouseover_ind = item
                    break

    def mouse_unclick(self, x, y):
        pass  

    def get_selection(self, name=False):
        """
        Return the selected index.
        """
        return self._selected_ind   if not name else self._texts[self._selected_ind]
    


