"""
Toolbox sub-window of the cluster editor.
"""

import cv2
import numpy as np
from colors import COLORS
from util import bbox_contains, calc_font_size
from abc import ABC, abstractmethod

COLOR_OPTIONS = {'unselected': COLORS['black'],
                 'selected': COLORS['black'],
                 'mouseover': COLORS['red'],
                 'idle': COLORS['black'],
                 'held': COLORS['blue'],
                 'tab': COLORS['gray'],
                 'border': COLORS['gray'],}


class Tool(ABC):
    """
    Abstract class for tools in the cluster creator app.
    """

    def __init__(self, bbox, name):
        self._bbox = bbox
        self._name = name
        self._spacing_px = 7

    @abstractmethod
    def render(self, img):
        """
        Render the tool onto the image.
        """
        pass

    @abstractmethod
    def mouse_click(self, x, y):
        """
        Handle a mouse click.
        """
        pass

    @abstractmethod
    def mouse_move(self, x, y):
        """
        Handle a mouse move.
        """
        pass

    @abstractmethod
    def mouse_unclick(self, x, y):
        """
        Handle a mouse unclick.
        """
        pass

    def move_to(self, bbox):
        """
        Move the tool to the new bbox.
        """
        self._bbox = bbox
    def get_value(self):
        """
        Return the current value.
        """
        pass

class Slider(Tool):
    """
    Vertical slider, looks like this:

        Value = 0.4
        [---|------]
    """

    def __init__(self, bbox, name, range, default=None, format_str='=%.2f'):
        super().__init__(bbox, name)
        self._format_str = format_str
        self._t_vert_frac = 0.6   # fraction of slider that is for title
        self._slider_width_px = 10

        self._range = range
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._colors = {c_opt: COLOR_OPTIONS[c_opt].tolist() for c_opt in COLOR_OPTIONS}

        self._calc_dims()

        self._moused_over = False
        self._held = False
        if default is None:
            self._slider_pos = 0.0  # smallest (biggest is 1.0)
        else:
            self._slider_pos = (default - self._range[0]) / (self._range[1] - self._range[0])

    def get_value(self):
        """
        Return the current value.
        """
        return self._range[0] + self._slider_pos * (self._range[1] - self._range[0])

    def _calc_dims(self):
        self._y_midline = int((self._bbox['y'][1] * self._t_vert_frac + self._bbox['y'][0] * (1 - self._t_vert_frac)))
        # title position
        self._text_bbox = {'x': (self._bbox['x'][0]+self._spacing_px, self._bbox['x'][1]-self._spacing_px),
                           'y': [self._bbox['y'][0]+self._spacing_px, self._y_midline]}

        # test font size with random value
        test_val =self._range[1] 
        test_str = self._name + self._format_str%(test_val,)

        self._font_size, _ = calc_font_size([test_str], self._text_bbox, self._font, 0)
        title_dims, baseline = cv2.getTextSize(test_str, self._font, self._font_size, 1)
        left = self._bbox['x'][0] + self._spacing_px
        top = self._bbox['y'][0] + title_dims[1] + self._spacing_px + baseline
        self._title_pos = (left, top)
        # slider dims
        self._s_left = left
        self._s_right = self._bbox['x'][1] - self._spacing_px
        self._s_top = self._y_midline + self._spacing_px
        self._s_bottom = self._bbox['y'][1] - self._spacing_px

        self._slider_bbox = {'x': [self._s_left, self._s_right],
                             'y': [self._s_top, self._s_bottom]}

    def render(self, img):
        """
        Render the slider.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        if False:  # debug graphics
            # text bbox
            cv2.rectangle(img, (self._text_bbox['x'][0], self._text_bbox['y'][0]),
                        (self._text_bbox['x'][1], self._text_bbox['y'][1]), COLORS['red'].tolist(), 1)
            # midline in violet
            cv2.line(img, (self._bbox['x'][0], self._y_midline), (self._bbox['x'][1], self._y_midline), COLORS['violet'].tolist(), 1)
            # slider bbox in turquoise
            cv2.rectangle(img, (self._slider_bbox['x'][0], self._slider_bbox['y'][0]),
                        (self._slider_bbox['x'][1], self._slider_bbox['y'][1]), COLORS['turquoise'].tolist(), 1)
      
        # bbox, in border color
        #cv2.rectangle(img, p0, p1, self._colors['border'], 2)
            
        # title
        val = self.get_value()
        slider_str = self._name + self._format_str%(val,)
        cv2.putText(img, slider_str, self._title_pos, self._font, self._font_size, self._colors['idle'])
        # slider bar
        slider_y = (self._slider_bbox['y'][0] + self._slider_bbox['y'][1]) // 2
        cv2.line(img, (self._s_left, slider_y), (self._s_right, slider_y), self._colors['idle'], 1)
        # slider tab
        slider_xpos = int(self._slider_pos * (self._s_right - self._s_left)) + self._s_left - self._slider_width_px // 2

        tab_color = self._colors['tab']
        if self._held:
            tab_color = self._colors['held']
        elif self._moused_over:
            tab_color = self._colors['mouseover']
        
        img[self._slider_bbox['y'][0] :self._slider_bbox['y'][1],
            slider_xpos: slider_xpos + self._slider_width_px] = np.array(tab_color, dtype=np.uint8)


    def mouse_click(self, x, y):
        """
        Check if the click is within the slider.
        """
        #print("Slider mouse click")
        if bbox_contains(self._bbox, x, y):
            self._held = True
            self._move_slider(x)
            return True

    def mouse_move(self, x, y):
        """
        Check if the mouse is over the slider.
        """
        self._moused_over = False
        if bbox_contains(self._bbox, x, y):
            self._moused_over = True
        if self._held:
            self._move_slider(x)
            return True
        return False

    def mouse_unclick(self, x, y):
        self._held = False
        #print("Slider mouse unclick")

    def _move_slider(self, x):
        """
        Move the slider to the new position.
        """

        rel_x = (x - self._s_left) / (self._s_right - self._s_left)  
        rel_x = np.clip(rel_x, 0, 1)
        self._slider_pos = rel_x

class Button(Tool):
    """
    Rectangular area with text label.
    left-click calls callback function.
    """

    def __init__(self, bbox, text, callback, border_indent=2):
        super().__init__(bbox, text)
        self._text = text
        self._callback = callback
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._text_bbox = {'x': (bbox['x'][0] + self._spacing_px, bbox['x'][1] - self._spacing_px),
                           'y': (bbox['y'][0] + self._spacing_px, bbox['y'][1] - self._spacing_px)}
    
        self._font_size, _ = calc_font_size([text], self._text_bbox, self._font, border_indent)
        self._colors = {c_opt: COLOR_OPTIONS[c_opt].tolist() for c_opt in COLOR_OPTIONS}
        self._calc_dims()
        self._moused_over = False
        self._held = False

    def move(self, bbox):
        """
        Move the button to the new bbox.
        """
        self._bbox = bbox
        self._calc_dims()

    def _calc_dims(self):
        """
        Determine the text size and position.
        """
        text_dims = cv2.getTextSize(self._text, self._font, self._font_size, 1)[0]
        x0, x1 = self._bbox['x']
        y0, y1 = self._bbox['y']
        self._text_pos = (x0 + (x1 - x0 - text_dims[0]) // 2, y0 + (y1 - y0 + text_dims[1]) // 2)

    def render(self, img):
        """
        Render the button.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        color = self._colors['idle']
        if self._held:
            color = self._colors['held']
        elif self._moused_over:
            color = self._colors['mouseover']
        # bbox
        # cv2.rectangle(img, p0, p1, self._colors['idle'], 1)

        # text box
        cv2.rectangle(img, (self._text_bbox['x'][0], self._text_bbox['y'][0]),
                        (self._text_bbox['x'][1], self._text_bbox['y'][1]), self._colors['idle'], 1)
        cv2.putText(img, self._text, self._text_pos, self._font, self._font_size, color)

    def mouse_click(self, x, y):
        """
        Check if the click is within the button.
        """
        #print("Button mouse click")
        if bbox_contains(self._bbox, x, y):
            self._held = True
            return True

    def mouse_move(self, x, y):
        """
        Check if the mouse is over the button.
        """
        self._moused_over = False
        if bbox_contains(self._bbox, x, y):
            self._moused_over = True
            return self._held
        else:
            if self._held:
                self._held = False
        return False
            

    def mouse_unclick(self, x, y):
        #print("Button mouse unclick")
        if self._held and bbox_contains(self._bbox, x, y):
            self._callback()
        self._held = False

    def get_value(self):
        return self._text


class RadioButtons(Tool):
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
        super().__init__(bbox, title)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._title = title
        self._bbox = bbox
        self._options = options
        self._texts = texts if texts is not None else options
        self._selected_ind = default_selection if default_selection is not None else 0  # currently selected index
        self._mouseover_ind = None  # index of item the mouse is over
        self._colors = {c_opt: COLOR_OPTIONS[c_opt].tolist() for c_opt in COLOR_OPTIONS}

        self._calc_dims()

    def move(self, bbox):
        """
        Move the button to the new bbox.
        """
        self._bbox = bbox
        self._calc_dims()

    def _calc_dims(self):
        """
        Determine the text size, the X,Y positions of each option.
        """
        self._font_size, self._mark_height = calc_font_size([self._title] + self._texts,
                                                            self._bbox,
                                                            self._font,
                                                            self._spacing_px)
        text_dims = [cv2.getTextSize(text, self._font, self._font_size, 1)  # (width, height), baseline
                     for text in self._texts]
        self._baseline = text_dims[0][1]
        title_dims = cv2.getTextSize(self._title, self._font, self._font_size, 1)[0]

        left = self._bbox['x'][0] + self._spacing_px
        top = self._bbox['y'][0] + title_dims[1] + self._spacing_px
        self._title_pos = left, top
        line_length = np.max([int(title_dims[0] * .85), 5])
        top += self._spacing_px + self._baseline
        self._line_coords = [(left, top),
                             (left+line_length, top)]
        self._text_pos = []
        # indent for indicator-dot
        self._dot_size = 2
        left += self._dot_size + self._spacing_px * 2

        # top += self._spacing_px
        self._div_lines = [top]   # y-coordinate between each selection (for determining which one the mouse is over)

        for text_dim in text_dims:
            top += text_dim[0][1] + self._spacing_px + self._baseline
            self._text_pos.append((left, top))
            self._div_lines.append(top+self._baseline)

    def render(self, img):
        """
        Render the radio buttons.
        """
        p0 = (self._bbox['x'][0], self._bbox['y'][0])
        p1 = (self._bbox['x'][1], self._bbox['y'][1])
        #cv2.rectangle(img, p0, p1, self._colors['unselected'], 1)
        cv2.putText(img, self._title, self._title_pos, self._font,
                    self._font_size, self._colors['unselected'])
        cv2.line(img, self._line_coords[0], self._line_coords[1], self._colors['unselected'])
        for i, text_pos in enumerate(self._text_pos):
            if i == self._selected_ind:
                color = self._colors['selected']
                # indicator dot
                dot_y = (self._div_lines[i+1] - self._baseline-self._mark_height)
                cv2.circle(img, (self._text_pos[i][0] - self._spacing_px, dot_y), self._dot_size, color, -1)
            elif i == self._mouseover_ind:
                color = self._colors['mouseover']
            else:
                color = self._colors['unselected']
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
        #print("RadioButtons mouse click")
        if bbox_contains(self._bbox, x, y):
            ind = self._get_item_at(y)
            if ind is not None:
                self._selected_ind = ind
        return False

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
        return False
    
    def get_value(self):
        """
        Return the selected index.
        """
        return self._options[self._selected_ind]

    def mouse_unclick(self, x, y):
        #print("RadioButtons mouse unclick")
        pass
    def get_selection(self, name=False):
        """
        Return the selected index.
        """
        return self._selected_ind if not name else self._texts[self._selected_ind]
