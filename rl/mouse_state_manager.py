"""
Determine which bounding boxes are selected/mouseovered, etc.


"""
import tkinter as tk
import tkinter.ttk as ttk
from abc import ABC, abstractmethod
from PIL import Image, ImageTk
from util import tk_color_from_rgb
from colors import COLOR_SCHEME, UI_COLORS
from layout import LAYOUT, WIN_SIZE
import numpy as np
import logging
from scipy.spatial import KDTree
import cv2


class MouseBoxManager(object):
    """
    Manages mouse interaction with bounding boxes in an image.

    Create one for each TabContentPage that needs mouse interaction:
        - For selecting states
        - For mousovering plots for color keys

    """

    def __init__(self, app):
        """
        :param app: The main application instance.
        :param env: The environment to use.
        """
        self._app = app
        self.bboxes = []
        self._box_ids = []
        self._box_centers = None
        self._box_wh = None
        self._box_tree = None
        self._size = None

        self.mouseover_id = None
        self.mouse_offset_xy = None
        self.pos_xy = None
        self.active = False

    def set_boxes(self, bboxes):
        """
        Set the bounding boxes to track for mouse events.

        :param boxes: dict of {'id': {'x': (x1, x2), 'y': (y1, y2)}}
        """
        self.bboxes = bboxes
        box_order = [box_id for box_id in bboxes]
        box_centers = []
        box_wh = []
        self._box_ids = []
        for box in box_order:
            self._box_ids.append(box)
            pos = self.bboxes[box]
            box_centers.append(((pos['x'][0] + pos['x'][1]) / 2,
                                (pos['y'][0] + pos['y'][1]) / 2))
            box_wh.append((pos['x'][1] - pos['x'][0], pos['y'][1] - pos['y'][0]))
        # Store the centers of the boxes for quick access

        self._box_inds = {box_id: i for i, box_id in enumerate(self._box_ids)}
        self._box_centers = np.array(box_centers)
        self._box_wh = np.array(box_wh)
        self._box_tree = KDTree(box_centers)

    def get_box_at(self, pos_xy, margin=0):
        """
        Get the box ID at the given coordinates, if any.

        :param (x, y): The coordinates to check.
        :return: The box ID if found, otherwise None.
        """
        if self._box_tree is None:
            return None, None

        pt = np.array(pos_xy)
        # Find the nearest box center
        _, ind = self._box_tree.query(pt)
        center_offset = self._box_centers[ind] - pt
        if np.all(np.abs(center_offset) <= (self._box_wh[ind] / 2 + margin)):
            box_id = self._box_ids[ind]
            bbox = self.bboxes[box_id]
            xy_offset = pos_xy - np.array((bbox['x'][0], bbox['y'][0]))

            return box_id, xy_offset
        return None, None

    def mouse_move(self, pos_xy):
        """
        Handle mouse move events.

        :param pos_xy: The (x, y) coordinates of the mouse.
        :return: True if mouseover state changes, False otherwise.
        """
        self.pos_xy = pos_xy
        self.active = True

        if self._box_tree is None:
            return False

        box_id, xy_offset = self.get_box_at(pos_xy)

        if box_id is not None:
            if self.mouseover_id is None or box_id != self.mouseover_id:
                self.mouseover_id = box_id
                self.mouse_offset_xy = xy_offset
                return True
        else:
            if self.mouseover_id is not None:
                self.mouseover_id = None
                return True
        return False

    def mouse_leave(self):
        """
        Handle mouse leave events.

        :return: True if mouseover state changes, False otherwise.
        """
        self.active = False
        self.pos_xy = None
        if self.mouseover_id is not None:
            self.mouseover_id = None
            return True
        return False

    def mouse_click(self, pos_xy):
        """
        Handle mouse click events.

        :param pos_xy: The (x, y) coordinates of the mouse.
        :return: relative offset w/in the box if a box was clicked, False otherwise.
        """
        self.pos_xy = pos_xy
        self.active = True
        return self.get_box_at(pos_xy)

    def mark_box(self, img, box_id, color, thickness=1):
        if box_id in self.bboxes:
            bbox = self.bboxes[box_id]
            self._draw_box(img, bbox, color, thickness=thickness)

    def render_state(self, img, selected_ids, thickness=1):
        """
        Draw boxes around the states in the image.
        """

        for box_id in selected_ids:
            if box_id in self.bboxes:
                bbox = self.bboxes[box_id]
                self._draw_box(img, bbox, UI_COLORS['selected'])

        if self.mouseover_id is not None:
            bbox = self.bboxes[self.mouseover_id]
            self._draw_box(img, bbox, UI_COLORS['mouseovered'])

    def _draw_box(self, img, bbox, color, thickness=1):
        p0 = bbox['x'][0]-1, bbox['y'][0]-1
        p1 = bbox['x'][1], bbox['y'][1]
        cv2.rectangle(img, p0, p1, color, thickness=thickness)


class MSMTester(object):
    """
    Create "tab 1" and "tab 2", each fills the frame with a single label image to be managed by the MouseBoxManager.
    Tab 1 has a grid of boxes with random colors.
    Tab 2 has a random subset of boxes, blown-up.
    Selecting a box in either tab will toggle its selection state, i.e. there is one set of selected boxes for both tabs.
    Mouseovering a box will highlight it in green.

    """

    def __init__(self, img_size, box_size):
        self._img_size = img_size  # full window
        self._tab_img_size = img_size  # inside the tab frame
        self._box_size = box_size
        self._boxes, self._colors = self._calc_boxes()

        self._tab_names = ['tab 1', 'tab 2']
        self._tab_boxes = {'tab 1': self._boxes,
                           'tab 2': self._calc_box_subset(self._boxes, 10)}

        self._init_images()
        self._msms = {tab_name: MouseBoxManager(self) for tab_name in self._tab_names}
        self._frames = None
        self._labels = None
        self._cur_tab = None
        self._root = None

        self._init_tk()

        self._selected_ids = []

    def _init_images(self):
        """
        The base images have the color in each box for the two tabs.
        """
        blank = np.zeros((self._tab_img_size[1], self._tab_img_size[0], 3), dtype=np.uint8)
        blank[:] = COLOR_SCHEME['bg']

        self._base_img = {'tab 1': blank.copy(),
                          'tab 2': blank.copy()}
        # Full set:
        for box_id, pos in self._boxes.items():
            x1, x2 = pos['x']
            y1, y2 = pos['y']
            color = self._colors[box_id]
            self._base_img['tab 1'][y1:y2, x1:x2] = color
        # Subset set:
        for box_id, pos in self._tab_boxes['tab 2'].items():
            x1, x2 = pos['x']
            y1, y2 = pos['y']
            color = self._colors[box_id]
            self._base_img['tab 2'][y1:y2, x1:x2] = color

    def toggle_selected(self, box_id):
        """
        Toggle the selection state of a box.

        :param box_id: The ID of the box to toggle.
        """
        if box_id in self._selected_ids:
            self._selected_ids.remove(box_id)
        else:
            self._selected_ids.append(box_id)
        logging.info("Toggled selection for box %s, now selected: %s" % (box_id, box_id in self._selected_ids))

    def _get_bbox_grid(self, box_size):
        bboxes = {}
        box_spacing = int(0.33 * box_size)
        x, y = box_spacing, box_spacing
        ny = 0
        while y + box_spacing + box_size < self._tab_img_size[1]:
            ny += 1
            nx = 0
            while x + box_spacing + box_size < self._tab_img_size[0]:
                nx += 1
                box_id = f"box_{x}_{y}"
                bboxes[box_id] = {'x': (x, x + box_size),
                                  'y': (y, y + box_size)}
                x += box_spacing + box_size

            y += box_spacing + box_size
            x = box_spacing
        return bboxes, (nx, ny)

    def _calc_boxes(self):
        """
        """

        bboxes, _ = self._get_bbox_grid(self._box_size)
        colors = {box_id: (np.random.randint(90, 256),
                           np.random.randint(90, 256),
                           np.random.randint(90, 256))
                  for box_id in bboxes}
        return bboxes, colors

    def _calc_box_subset(self, boxes, n):
        """
        Pick n random box_ids, make a grid with bigger boxes for the subset.
        """
        box_ids = list(boxes.keys())
        subset_ids = np.random.choice(box_ids, n, replace=False)
        subset_bboxes, (nx, ny) = self._get_bbox_grid(self._box_size * 5)
        if (nx * ny) < n:
            raise ValueError("Not enough boxes to create a subset of size %d" % n)

        subset_bboxes = {box_id: subset_bboxes[subset_id]
                         for (box_id, subset_id) in zip(subset_ids, subset_bboxes.keys())}
        return subset_bboxes

    def _init_tk(self):
        """
        Init Tkinter and create the main window with two tabs.
        """
        self._root = tk.Tk()
        self._root.title("Mouse State Manager Tester")
        self._root.geometry(f"{self._img_size[0]}x{self._img_size[1]}")

        self._notebook = tk.ttk.Notebook(self._root)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._frames = {}
        self._labels = {}

        for tab_name in self._tab_names:
            frame = tk.Frame(self._notebook, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
            self._notebook.add(frame, text=tab_name)
            self._frames[tab_name] = frame

            img = Image.new('RGB', self._img_size, color=COLOR_SCHEME['bg'])
            label = tk.Label(frame, image=ImageTk.PhotoImage(img))
            label.pack(fill=tk.BOTH, expand=True)
            self._labels[tab_name] = label

            # Set up the MouseBoxManager for this tab
            msm = self._msms[tab_name]
            msm.set_boxes(self._tab_boxes[tab_name])
            label.bind("<Motion>", self.on_mouse_move)
            label.bind("<Button-1>", self.on_mouse_click)
            label.bind("<Leave>", self.on_mouse_leave)
            label.bind("<Configure>", self.on_tab_resize)  # for all tabs

        self._notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self._cur_tab = self._tab_names[0]
        self._notebook.select(self._frames[self._cur_tab])

    def on_tab_resize(self, event):
        new_tab_size = (event.width, event.height)
        if self._tab_img_size is None or (self._tab_img_size != new_tab_size):
            logging.info("Resizing tab images to %s" % str(new_tab_size))
            self._tab_img_size = new_tab_size
            self._init_images()  # re-draw base images
            self.refresh_images()

    def on_tab_change(self, event):
        logging.info("Tab changed to %s" % self._notebook.tab(self._notebook.select(), "text"))
        self._cur_tab = self._notebook.tab(self._notebook.select(), "text")
        self.refresh_images()

    def on_mouse_leave(self, event):
        """
        Handle mouse leave events for the current tab.
        """
        msm = self._msms[self._cur_tab]
        if msm.mouse_leave():
            self.refresh_images()

    def on_mouse_move(self, event):
        msm = self._msms[self._cur_tab]
        if msm.mouse_move((event.x, event.y)):
            self.refresh_images()

    def on_mouse_click(self, event):
        msm = self._msms[self._cur_tab]
        # import ipdb; ipdb.set_trace()
        box_id, _ = msm.mouse_click((event.x, event.y))
        if box_id is not None:
            self.toggle_selected(box_id)
            self.refresh_images()

    def _render_frame(self, tab_name):
        frame = self._base_img[tab_name].copy()
        msm = self._msms[tab_name]
        # Draw the selected boxes:
        msm.render_state(frame, self._selected_ids, thickness=2)

        return frame

    def refresh_images(self):
        if self._img_size is None:
            return
        new_img = self._render_frame(self._cur_tab)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        label = self._labels[self._cur_tab]
        label.config(image=new_img)
        label.image = new_img

    def start(self):
        """
        Start the Tkinter main loop.
        """
        self._root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    img_size = (800, 600)
    box_size = 20
    tester = MSMTester(img_size, box_size)
    tester.start()
    logging.info("Mouse State Manager Tester finshed.")
