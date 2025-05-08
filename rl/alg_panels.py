"""
Algorithm selection, state load/save/reset, fullscreen, start game.

"""

from policy_eval import PolicyEvalDemoAlg
from collections import OrderedDict
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import numpy as np
from colors import COLOR_BG, COLOR_DRAW, COLOR_LINES, COLOR_TEXT
from gui_base import Panel
from util import tk_color_from_rgb, get_clobber_free_filename
from layout import LAYOUT, WIN_SIZE
import logging
import time
from threading import Thread, Event
import cv2


TITLE_INDENT = 0
ITEM_INDENT = 20
FPS = 30


class StatusControlPanel(Panel):
    """
    Status panel for displaying the current status of the algorithm and buttons for controlling it.
    """

    def __init__(self, app, alg, bbox_rel):
        """
        :param app: The application object.
        :param alg: a DemoAlg object (RL state)
        """
        self._alg = alg
        self._run_control_options = []
        super().__init__(app, bbox_rel)

    def _init_widgets(self):

        # init three frames:
        self._status_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._status_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self._run_control_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._run_control_frame.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self._button_frame = tk.Frame(self._frame, bg=self._bg_color)
        self._button_frame.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)

        self._run_control_vars = {}

        self._init_status()  # top half of the panel
        self._init_run_control()  # bottom left half of the panel
        self._init_buttons()  # bottom right half of the panel

    def _init_title(self):
        """
        Create the title label for the selection panel.
        """
        # Add dark line below the title:
        # self._title_line = tk.Frame(self._frame, height=2, width=100, bg=self._line_color)
        # self._title_line.pack(side=tk.TOP)
        self._add_spacer()

    def _init_status(self):
        """
        Goes at the top of the status panel, one line per status message.
        """
        status_title = tk.Label(self._status_frame, text="Status",
                                font=LAYOUT['fonts']['title'],
                                bg=self._bg_color, fg=self._text_color, anchor="w", justify="left")
        status_title.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

        test_status = self._alg.get_status()

        self._status_labels = []  # list of status labels, aligned on the left
        for i, (text, font) in enumerate(test_status):
            label = tk.Label(self._status_frame, text=text, bg=self._bg_color, font=font, anchor="w", justify="left")
            label.pack(side=tk.TOP, fill=tk.X, padx=ITEM_INDENT, pady=0)
            self._status_labels.append(label)

    def refresh_status(self):
        """
        Refresh the status labels with the current status of the algorithm.
        """
        test_status = self._alg.get_status()
        for i, (text, font) in enumerate(test_status):
            self._status_labels[i].config(text=text, font=font)
        # TODO: Add/shrink labels as needed.

    def _on_resize(self, event):
        return super()._on_resize(event)

    def _init_run_control(self):
        """
        Below status, left half of the panel.
        """
        breakpoint_label = tk.Label(self._run_control_frame, text="Checkpoints",
                                    bg=self._bg_color, font=LAYOUT['fonts']['title'], anchor="w", justify="left")
        breakpoint_label.pack(side=tk.TOP, fill=tk.X, padx=TITLE_INDENT, pady=4)

        self._add_spacer(frame=self._run_control_frame)
        self._run_control_frame.grid_rowconfigure(0, weight=1)  # make the bottom frame fill the remaining space
        self._run_control_frame.grid_columnconfigure(0, weight=1)
        self._reset_run_control_options()

    def get_run_control_settings(self):
        """
        :returns: {option-name:  bool} dict of the current state of the run control options.
        """
        options = {key: (var.get() == 1) for _, var, key in self._run_control_options}
        return options
    
    
    def change_run_control_settings(self, run_control):
        for _, var, key in self._run_control_options:
            var.set(1 if run_control[key] else 0)


    def _reset_run_control_options(self):
        """
        Initializing or algorithm changed, look up options, will be ordered dict of (key, string) pairs.
        The algorithm will send its updates labeled with the key.
        The strings for the options will display with check boxes.
        If there are currently the wrong number of options remove/add them. 
        """
        alg_options = self._alg.get_run_control_options()
        # remove old options:
        for option in self._run_control_options:
            option[0].destroy()
        self._run_control_options = []

        # add new options:
        for key, text in alg_options.items():
            var = tk.IntVar()
            check = tk.Checkbutton(self._run_control_frame, text=text, variable=var, bg=self._bg_color,
                                   font=LAYOUT['fonts']['default'], anchor="w", justify="left", command=self._on_control_change)
            check.pack(side=tk.TOP, fill=tk.X, padx=10, pady=0)
            self._run_control_options.append((check, var, key))
            # select all stops
            var.set(1)

    def _on_control_change(self):
        new_vals = self.get_run_control_settings()
        logging.info("Run control options changed: %s" % new_vals)
        self._alg.update_run_control(new_vals)

    def _init_buttons(self):
        """
        Below status, right half of the panel.
        """
        self._add_spacer(20, frame=self._button_frame)
        # "Clear breakpoints" button:
        self._clear_button = tk.Button(self._button_frame, text="Clear Stops",
                                       font=LAYOUT['fonts']['buttons'], bg=self._bg_color,
                                       command=self._clear_breakpoints, padx=7, pady=5)
        self._clear_button.pack(side=tk.TOP, pady=10)

        # "Go/Stop" button
        self._go_button = tk.Button(self._button_frame, text="Go",
                                    font=LAYOUT['fonts']['buttons'], bg=self._bg_color,
                                    command=self._go_stop, padx=7, pady=5)
        self._go_button.pack(side=tk.TOP, pady=10)

    def _clear_breakpoints(self):
        pass

    def _go_stop(self):
        rcs = self.get_run_control_settings()
        self._alg.update_run_control(rcs)
        self._alg.advance()


class StatePanel(Panel):
    """
    Largest, taking up the entire right half, displaying all the game states or their values, etc.
    Tabbed interface, queries algorithm for details.
    """

    def __init__(self, app, bbox_rel, alg=None):
        """
        :param app: The application object.
        :param bbox_rel: The bounding box for the panel, relative to the parent frame.
        :param alg: DemoAlg subclass
        """
        self._state_tabs = {}
        self._blank = np.zeros((10, 10, 3), dtype=np.uint8)  # blank image for the tabs
        self._blank[:] = COLOR_BG  # set the color to the background color

        super().__init__(app, bbox_rel)
        if alg is not None:
            self.set_algorithm(alg)

    def _on_resize(self, event):
        super()._on_resize(event)
        # Get a new frame for the current tab and force a redraw which will get the new size:
        #self.refresh_image(is_paused=False)

    def set_algorithm(self, alg):
        """
        User changed the algorithm, might need new tab lables, etc.
        """
        self._alg = alg
        self._set_tabs()
        self._alg.set_state_panel(self)  # probaly already set

    def _init_widgets(self):
        """
        Initialize the widgets for the state panel.
        """
        # Create a notebook for the tabs:
        self._notebook = ttk.Notebook(self._frame, style='TNotebook')

        style = ttk.Style(self._frame)
        # Override default font for tabs
        style.configure('TNotebook.Tab', font=LAYOUT['fonts']['tabs'])

        # Set callback for the tab:
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._notebook.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Set the default tab to the first one:
        self.cur_tab = None

    def _set_tabs(self):
        """
        Set the tabs for the state panel.
        :param tab_info: (see __init__)
        """
        tab_info = self._alg.get_state_image_kinds()
        self._tabs_by_text = {tab_txt: tab_name for tab_name, tab_txt in tab_info.items()}
        # Clear old tabs:
        for tab, img_label in self._state_tabs.values():
            tab.destroy()
            img_label.destroy()
        self._state_tabs = {}

        # Add new tabs:
        for tab_name, tab_str in tab_info.items():
            # Create a new frame for the tab:
            tab_frame = tk.Frame(self._notebook, bg=self._bg_color)
            tab_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Create a label for the tab to hold the image (filling the whole tab frame):
            img_label = tk.Label(tab_frame, bg=self._bg_color)
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Set the image to self._blank:
            img_copy = ImageTk.PhotoImage(Image.fromarray(self._blank.copy()))
            img_label.config(image=img_copy)
            img_label.image = img_copy
            # Add the tab to the notebook:
            self._notebook.add(tab_frame, text=tab_str)
            # Add the tab to the state tabs dictionary:
            self._state_tabs[tab_name] = (tab_frame, img_label)

        # get the current tab:
        self.cur_tab = self._tabs_by_text[self._notebook.tab(self._notebook.select(), "text")]

    def get_image_size(self):
        time.sleep(3)
        # use the w/h of the image label in the current tab:
        tab_frame, label = self._state_tabs[self.cur_tab]
        w, h = label.winfo_width(), label.winfo_height()
        if w in [1, 0] or h in [1, 0]:
            # use the frame dimensions instead:

            w, h = tab_frame.winfo_width(), tab_frame.winfo_height()
            logging.info("StatePanel: image label size is 0, using tab frame size instead.")
            if w in [1, 0] or h in [1, 0]:
                # use the default size:
                raise Exception("StatePanel: no size available for image label or tab frame, called too early?")
        return w, h

    def refresh_image(self, is_paused=False):
        """
        Ask the algorithm for a new image and put it in the current tab.
        """
        img_size=self.get_image_size()
        image = self._alg.get_state_image(size=img_size,
                                          which=self.cur_tab,
                                          is_paused=is_paused)
        # Get the current tab:
        tab_frame, img_label = self._state_tabs[self.cur_tab]
        # Create a new image from the numpy array:
        img_copy = ImageTk.PhotoImage(Image.fromarray(image))
        # Set the image to the label:
        img_label.config(image=img_copy)
        img_label.image = img_copy
        

    def _on_tab_changed(self, event):
        """
        """
        logging.info("Tab changed to %s" % self._notebook.tab(self._notebook.select(), "text"))

        # get the current tab:
        self.cur_tab = self._tabs_by_text[self._notebook.tab(self._notebook.select(), "text")]
        #self.refresh_image(is_paused=self._alg.paused)


class TestDemoAlg(PolicyEvalDemoAlg):

    def __init__(self, *args, **kwargs):
        self._state = None
        self._image_data = np.random.rand(4 * 40).reshape(40, 4)  # plot these (x,y, rad,thickness)
        self._bkg_color_rgb = COLOR_BG
        self._text_color_rgb = COLOR_TEXT
        self._line_color_rgb = COLOR_LINES

        # Update these as the algorithm runs
        self._img_size = None
        self._paused_images = {key: None for key in self.get_state_image_kinds().keys()}
        self._running_images = {key: None for key in self.get_state_image_kinds().keys()}

        super().__init__(*args, **kwargs)

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['circle-update'] = "Circle update"
        rco['frame-update'] = "Frame update"
        return rco

    def get_state_image_kinds(self):
        return OrderedDict([
            ('hollow', 'Hollow Circles'),
            ('filled', 'Filled Circles'),
        ])

    def get_state_image(self,  size, which, is_paused=False):
        if self._state_panel is None:
            print("Refresh but no state panel")
            return
        started_up = False

        while not started_up:
            try:
                img_size = self._state_panel.get_image_size()
                started_up = True
            except Exception as e:
                #  Too early to draw images
                pass
            time.sleep(0.1)
            print("Waiting for image size...")
        
        re_draw = False
        if self._img_size is None or size != self._img_size:
            # Force re-draw.
            re_draw = True
            self._img_size = size

        if is_paused:
            if re_draw or self._paused_images[which] is None:
                # Get a new image:
                img = self._get_state_image(img_size, which, is_paused=is_paused)
                self._paused_images[which] = img
            return self._paused_images[which]
        else:
            if re_draw or self._running_images[which] is None:
                # Get a new image:
                img = self._get_state_image(img_size, which, is_paused=is_paused)
                self._running_images[which] = img
            return self._running_images[which]


    def _get_state_image(self, img_size, which, is_paused=False):
        blank = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        blank[:] = self._bkg_color_rgb  # set the color to the background color
        return blank

    def draw_image_data(self, img, connect=False):
        PREC_BITS = 5
        PREC_MULT = 2**PREC_BITS
        w, h = img.shape[1], img.shape[0]
        circle_coords = self._image_data[:2] * np.array([w, h]) * PREC_MULT
        circle_coords = circle_coords.astype(int)
        circle_radii = int(self._image_data[2] * w/20) * PREC_MULT
        thicknesses = int(self._image_data[3] * 20)
        if connect:
            # draw lines from every circle center to every other circle center:
            for i in range(len(circle_coords)):
                for j in range(i + 1, len(circle_coords)):
                    color = self._line_color_rgb
                    cv2.line(img, tuple(circle_coords[i]), tuple(circle_coords[j]),
                             color, thickness=1, lineType=cv2.LINE_AA)
        for i, ((x, y), r) in enumerate(zip(circle_coords, circle_radii)):
            color = self._line_color_rgb
            cv2.circle(img, (x, y), r, color, thickness=thicknesses[i], lineType=cv2.LINE_AA, shift=PREC_BITS)
        # draw the circles:
        return img

    def _maybe_pause(self, control_point):
        """
        Pause the algorithm if the run control indicates to do so.
        :param control_point: The control point to check.
        """
        if self._run_control[control_point]:
            self.paused=True
            self._state_panel.refresh_image(is_paused=True)
            self._go_signal.wait()
            self._go_signal.clear()
            self.paused=False
        else:
            self.paused=False
            self._state_panel.refresh_image(is_paused=False)


    def start(self):
        
        def run_thread():
            """
            If running, move the circles XYZ randomly every frame (FPS times per second).
            """
            n_frames = 0

            print("Tick")

            while True:
                print("Beginnning iteration: %i" % n_frames)
                n_frames += 1
                if n_frames % 30 == 0:
                    print("TestDemoAlg: frame %i" % n_frames)
                for ind in range(self._image_data.shape[0]):
                    print("Beginning sample %i" % ind)
                    # move the circles randomly:
                    self._image_data[ind, :2] += np.random.rand(2) * 0.1 - 0.05
                    self._image_data[ind, :2] = np.clip(self._image_data[ind, :2], 0, 1)

                    self._maybe_pause('circle-update')  # check if we should pause here.

                self._maybe_pause('frame-update')  # check if we should pause here.

        self._loop_thread = Thread(target=run_thread, daemon=True)
        self._loop_thread.start()
        logging.info("TestDemoAlg: started loop thread.")

    @staticmethod
    def get_name():
        return 'test-circles'

    @staticmethod
    def get_str():
        return "Test with circles"


class TestDemoAlg2(TestDemoAlg):
    """
    Similar but squares and only two state tabs
    """

    def get_state_image_kinds(self):
        return OrderedDict([
            ('hollow', 'Hollow Squares'),
            ('filled', 'Filled Squares'),
        ])

    def draw_image_data(self, img, connect=False):
        PREC_BITS = 5
        PREC_MULT = 2**PREC_BITS
        w, h = img.shape[1], img.shape[0]
        circle_coords = self._image_data[:2] * np.array([w, h]) * PREC_MULT
        circle_coords = circle_coords.astype(int)
        circle_radii = int(self._image_data[2] * w/20) * PREC_MULT
        thicknesses = int(self._image_data[3] * 20)
        if connect:
            # draw lines from every circle center to every other circle center:
            for i in range(len(circle_coords)):
                for j in range(i + 1, len(circle_coords)):
                    color = self._line_color_rgb
                    cv2.line(img, tuple(circle_coords[i]), tuple(circle_coords[j]),
                             color, thickness=1, lineType=cv2.LINE_AA)
        for i, ((x, y), r) in enumerate(zip(circle_coords, circle_radii)):
            color = self._line_color_rgb
            x_left, y_top = x - r, y - r
            x_right, y_bottom = x + r, y + r
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color,
                          thickness=thicknesses[i], lineType=cv2.LINE_AA, shift=PREC_BITS)
        # draw the circles:
        return img

    def get_run_control_options(self):
        """
        Get the run control options for the algorithm.
        :return: A dictionary of run control options.
        """
        rco = OrderedDict()
        rco['square-update'] = "Squares update"
        rco['frame-update'] = "Frame update"
        return rco

    @staticmethod
    def get_name():
        return 'test-squares'

    @staticmethod
    def get_str():
        return "Test with squres"


class TestApp(object):
    """
    Stand-in for demo app to test the SelectionPanel.
    """

    def __init__(self):
        from layout import WIN_SIZE
        from selection_panel import SelectionPanel
        from game_base import Mark
        from reinforcement_base import Environment
        from baseline_players import HeuristicPlayer

        self._bkg_color_rgb = COLOR_BG
        self._text_color_rgb = COLOR_TEXT
        self._line_color_rgb = COLOR_LINES
        self.win_size = WIN_SIZE
        self._fullscreen = False
        self._running = False
        self.root = tk.Tk()
        self.root.geometry(f"{self.win_size[0]}x{self.win_size[1]}")
        self.root.title("Selection Panel Test")
        self._env = Environment(opponent_policy=HeuristicPlayer(mark=Mark.O, n_rules=2))
        

        self._alg = TestDemoAlg(self, self._env)
        self.status_control_panel = StatusControlPanel(
            self, self._alg, bbox_rel=LAYOUT['frames']['control'])
        self._algorithms = [TestDemoAlg, TestDemoAlg2]
        self.selection_panel = SelectionPanel(self, self._algorithms, bbox_rel=LAYOUT['frames']['selection'])
        # test state tabs:
        self._state_img_size = None

        self.state_panel = StatePanel(self, bbox_rel=LAYOUT['frames']['state-tabs'])

        self._set_alg()

    def _set_alg(self):
        # selection panel
        first_alg_name = self._alg.get_name()
        self.selection_panel.set_selection(first_alg_name)

        # state panel
        self.state_panel.set_algorithm(self._alg)

    def run(self):
        # Start the run thread:
        self._alg.start()
        self.root.mainloop()

    def save_state(self):
        logging.info("Save state button pressed.")

    def load_state(self):
        logging.info("Load state button pressed.")

    def reset_state(self):
        logging.info("Reset state button pressed.")

    def toggle_fullscreen(self):
        logging.info("Toggle fullscreen button pressed.")

    def toggle_fullscreen(self):

        self._fullscreen = not self._fullscreen
        global geom
        if self._fullscreen:
            geom = self.root.geometry()
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.overrideredirect(True)
            self.root.geometry('%dx%d+0+0' % (w, h))

        else:
            self.root.overrideredirect(False)
            self.root.geometry(geom)

    def set_opponent(self, n_rules):
        logging.info(f"Setting opponent to {n_rules} rules.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = TestApp()
    app.run()
