"""
Class for window/toolbar management

The main app shows two game-tree images side by side:

   * Representation of v(s), small boxes w/a 2d embedding like the game
     graph, color indicating value.

   * Representation of delta-V(s), V_{t+1)(s), etc.  accumulating updates
     to be applied at the end of the epoch in batch mode, or the current
     update in online mode, etc.

Beneath these is a toolbar setting the different options for the RL algorithm.

   * Speed-setting: determines when learning pauses and waits for the user's
     signal to proceed.  Speed control has four mutually exclusive settings:
      * Stop after each state updates.
      * Stop after every state updates (an epoch).
      * Stop after state updating converges, between policy iterations.
      * Run continuously (until policy stability).

    (NOTE that the speed control settings will depend on the RL algorithm being used.
    The class for the algorithm should provide the GUI class with a list of settings,
    the action buttons for each setting.  User interactions w/the gui will call the
    algorithm's callback with the current setting & button pressed etc.)

   * Action buttons:  "Go/Stop" or "Step" button (name changes depending on speed setting 
     & current state)

   * Other toggles:

      * highlight updating states, (box around states visited since last frame)

      * Show updating states.  Instead of color boxes, show the game state images
        of the state being updateed and the states in which the update information
        originates and the edges between states.  Put these images in the same locations
        as their color box representations, but slightly larger.

      * SHow policy changes:  Every policy update, show a new window with all/some of
        the policy changes.  Each policy change is displayed as a game state image with
        the old move in red and the new move in green.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import numpy as np
from value_panel import get_box_placer, get_state_icons
from layer_optimizer import SimpleTreeOptimizer
from game_base import Result, Mark
from colors import COLOR_BG, COLOR_LINES, RED, GREEN, MPL_BLUE_RGB, MPL_GREEN_RGB, MPL_ORANGE_RGB
import tkinter as tk
from PIL import Image, ImageTk
from copy import deepcopy
import matplotlib.pyplot as plt
from enum import IntEnum
from color_scaler import ColorScaler

from threading import Lock, get_ident


def tk_color_from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
       https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


STATUS_W = 0.175  # Horizontal placement of first column.
FUNC_W = (1.0 - STATUS_W) / 2.0
T_CUTOFF = .7  # Vertical top of tournament frame.
BOTTOM_ROW = .8  # Vertical placement of bottom row.

global geom
geom = ""


class ViewModes(IntEnum):
    states = 0
    values = 1
    updates = 2


class RLDemoWindow(object):
    """
    Manage images and interactivity, don't do learning.
    """
    # fractions of window width & height for the different panels.
    # Use Tkinter place mode to set each of these panels' frames.
    LAYOUT = {'frames': {'status': {'x_rel': (0, STATUS_W),
                                    'y_rel': (0, 0.25)},
                         'tools': {'x_rel': (0, STATUS_W),
                                   'y_rel': (0.25, T_CUTOFF)},
                         'tournament': {'x_rel': (0, STATUS_W),
                                        'y_rel': (T_CUTOFF, 1.)},
                         'values': {'x_rel': (STATUS_W,  STATUS_W + 2 * FUNC_W),
                                    'y_rel': (0, BOTTOM_ROW)},
                         'step_viz': {'x_rel': (STATUS_W, 1),
                                      'y_rel': (BOTTOM_ROW, 1.)}},
              'margin_px': 5,
              'fonts': {'pannel_title': ('Helvetica', 16),
                        'title': ('Helvetica', 14, 'bold'),
                        'default': ('Helvetica', 12),
                        'menu': ('Helvetica', 14, 'underline'),
                        'buttons': ('Helvetica', 12, 'bold'),
                        'flag': ('Helvetica', 13, 'bold')}}

    FRAME_TITLES = {'status': 'Status',
                    'tools': 'Tools',
                    'tournament': 'TTT Tournament',
                    'values': 'Value Function',
                    'step_viz': 'Step Visualization'}

    def __init__(self, size,  demo_app, speed_options, player_mark=Mark.X):
        """
        :param demo_app:  instance of PolicyImprovementDemo that created this window.
        :param speed_options:  dict with one key-value pair per speed setting (see above),
          key is the name of the setting, value is a list of states the action button can be in, e.g.:
             if clicking once starts the algorithm running until the end of the epoch/iteration,
             the list will have one element corresponding to this action.  If clicking once pauses,
             clicking again resumes, there will be two elements in the list, cycled each time the button
             is clicked.  Each element of the list is a dict with ['text','callback'] where "text" is
             the text of the button and "callback" is the function to call when the button is clicked.
        :param player_mark:  The player for the agent.  The opponent is the other player.
        """
        self._size = size
        self._app = demo_app
        self._speed_options = speed_options
        self._player = player_mark
        self.color_bg = tk_color_from_rgb(COLOR_BG)
        self.color_lines = tk_color_from_rgb(COLOR_LINES)
        self._img_label = None
        self._pending_updates = []
        self._color_scalers = {'values': None, 'updates': None}
        self._view = 'states'  # one of ['states', 'values', 'updates']
        self._fullscreen = False

        self._layout_cache = {}  # {(w,h): {'box_placer': BoxPlacer, 'box_sizes': box_size_list}}

        self._init_tk()
        self._init_frames()
        self._init_status_panel()
        self._init_tools_panel()
        self._init_step_viz()
        # self._init_tournament_panel()

        self._state_images = None
        self._resize_lock = Lock()

        # self.make_images()
        self._base_images = {'states': None,  # representation of game state
                             'values': None,  # colors indicating value
                             'updates': None}  # (same)

        self._disp_images = {'states': None,  # annotate version of base_images
                             'values': None,
                             'updates': None}

        # Calls resize which creates state images:
        self._root.bind("<Configure>", lambda event: self._resize(event))

    def toggle_fullscreen(self):

        self._fullscreen = not self._fullscreen
        global geom
        if self._fullscreen:
            geom = self._root.geometry()
            w = self._root.winfo_screenwidth()
            h = self._root.winfo_screenheight()
            self._root.overrideredirect(True)
            self._root.geometry('%dx%d+0+0' % (w, h))

        else:
            self._root.overrideredirect(False)
            self._root.geometry(geom)

    def _init_step_viz(self):
        """
        For now just add a label that says "Step visualization"
        """
        self._step_viz_frame = self._frames['step_viz']
        # Create a label to hold the step vis image:
        self._step_viz_label = tk.Label(self._step_viz_frame, bg=self.color_bg, font=self.LAYOUT['fonts']['default'])
        self._step_viz_label.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, expand=True)

    def update_step_viz_image(self, image=None):
        """
        Update the step visualization image.
        :param image:  The image to update.
        """
        if image is not None:
            self._step_viz_label['image'] = ImageTk.PhotoImage(Image.fromarray(image))
            self._step_viz_label.image = image
            self._frame_labels['step_viz'].pack_forget()
            self._frame_lines['step_viz'].pack_forget()

        else:
            self._step_viz_label['image'] = None
            self._step_viz_label.image = None
            self._frame_labels['step_viz'].pack(side=tk.TOP, fill=tk.X)
            self._frame_lines['step_viz'].pack(side=tk.TOP)

    def get_step_viz_frame_size(self):
        """
        Get the size of the step visualization frame.
        :return:  The width and height of the step visualization frame.
        """
        width = self._frames['step_viz'].winfo_width()
        height = self._frames['step_viz'].winfo_height()
        return width, height

    def get_image_frame_size(self):
        width = self._frames['values'].winfo_width() - 8
        height = self._frames['values'].winfo_height()
        label_height = self._frame_labels['values'].winfo_height()
        return width, height - label_height

    def make_images(self):
        # for both (values & updates) for both views, create the base numpy images
        #  that will be modified as the algorithm progresses and sent to the canvas as new PhotoImages.
        logging.info("Creating images for states, values, and updates")
        if self._state_images is None:
            all_states = self._app.updatable_states + self._app.terminal_states
            self._state_images = get_state_icons(self._states_by_layer, box_sizes=self._box_sizes, player=self._player)
        # set size from frame dimensions
        frame_size = self.get_image_frame_size()

        values, new_values = self._app.get_values()
        self._color_scalers = {'values': ColorScaler(values), 'updates': ColorScaler(new_values)}
        state_img_blank = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        state_img_blank[:] = COLOR_BG
        val_img_blank = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        val_img_blank[:] = (100, 100, 100)
        self._base_images = {'values': self.box_placer.draw(colors=self._color_scalers['values'].color_LUT, dest=val_img_blank.copy()),
                             'states': self.box_placer.draw(images=self._state_images, dest=state_img_blank.copy()),
                             'updates': self.box_placer.draw(colors=self._color_scalers['updates'].color_LUT, dest=val_img_blank.copy())}
        self._disp_images = deepcopy(self._base_images)

    def _recalc_box_positions(self):
        """
        Create label to hold the main image.
        Determine the layout of the boxes in the image.
        """
        logging.info("Recalculating box positions")
        if self._img_label is not None:
            self._img_label.destroy()
        self._img_label = tk.Label(self._frames['values'], bg=self.color_bg, font=self.LAYOUT['fonts']['default'])
        # top center
        self._img_label.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, expand=True)
        # The image to put in the label will be this size:
        frame_size = self.get_image_frame_size()
        logging.info("Image in states/values frame will be %s" % str(frame_size))

        frame_size = self.get_image_frame_size()

        #  BOX LAYOUT:
        layout_key = (frame_size[0], frame_size[1])
        if layout_key in self._layout_cache:
            self.box_placer = self._layout_cache[layout_key]['box_placer']
            self._box_sizes = self._layout_cache[layout_key]['box_sizes']
            return
        all_states = self._app.updatable_states + self._app.terminal_states
        self.box_placer, self._box_sizes, self._states_by_layer = get_box_placer(
            frame_size, all_states, player=self._player)
        # Need to change to {'id': s, 'state':s} for each state in each layer.
        # states_by_layer = [[{'state': state} for state in layer] for layer in states_by_layer]
        terminal_lut = {state: state.check_endstate() for state in all_states}
        tree_opt = SimpleTreeOptimizer(image_size=frame_size,
                                       states_by_layer=self._states_by_layer,
                                       state_positions=self.box_placer.box_positions,
                                       terminal=terminal_lut)
        new_positions = tree_opt.get_new_positions()
        self.box_placer.box_positions = new_positions

        self._layout_cache[layout_key] = {'box_placer': self.box_placer,
                                          'box_sizes': self._box_sizes}

    def _init_status_panel(self):
        """
        Create labels for the status panel.
        These are left-justified in the status frame.
        Display the key and value like "Key:  Value" for each status item.
        """
        self._status_labels = {}
        status = self._app.get_status()

        # Add a spacer label at the top of the status panel
        self._add_spacer(self._frames['status'], height=5)
        for stat_name, stat_val in status.items():
            if stat_name not in ['title', 'flag']:
                status_str = "%s:  %s" % (stat_name, stat_val)
                font = self.LAYOUT['fonts']['default']
            elif stat_name == 'title':
                status_str = stat_val
                font = self.LAYOUT['fonts']['title']
            else:
                status_str = stat_val
                font = self.LAYOUT['fonts']['flag']
            label = tk.Label(self._frames['status'], text=status_str,
                             bg=self.color_bg, font=font, anchor=tk.W, justify=tk.LEFT)
            label.pack(side=tk.TOP, fill=tk.X)
            self._status_labels[stat_name] = label

    def _init_frames(self):
        """
        Create the frames for the different panels in the window.
        """
        self._frames = {}
        # Get window width & height to calculate padding
        x_pad_rel = self.LAYOUT['margin_px'] / self._size[0]
        y_pad_rel = self.LAYOUT['margin_px'] / self._size[1]

        self._frame_labels = {}
        self._frame_lines = {}

        for name, layout in self.LAYOUT['frames'].items():

            x_rel = layout['x_rel']
            y_rel = layout['y_rel']
            font = self.LAYOUT['fonts']['pannel_title']
            frame = tk.Frame(self._root, bg=self.color_bg)
            frame.place(relx=x_rel[0]+x_pad_rel, rely=y_rel[0]+y_pad_rel, relwidth=x_rel[1] - x_rel[0] - 2*x_pad_rel,
                        relheight=y_rel[1] - y_rel[0] - 2*y_pad_rel)

            label = tk.Label(frame, text=self.FRAME_TITLES[name], bg=self.color_bg, font=font)
            label.pack(side=tk.TOP, fill=tk.X)
            self._frame_labels[name] = label

            # Add dark line under label
            if name not in ['values', 'updates']:
                self._frame_lines[name] = tk.Frame(frame, height=2, width=100, bg=self.color_lines)
                self._frame_lines[name].pack(side=tk.TOP)

            self._frames[name] = frame

    def _add_spacer(self, frame, height=5):
        """
        Add a spacer label to the given frame.
        :param frame:  The frame to add the spacer to.
        :param height:  Height of the spacer in pixels.
        """
        label = tk.Label(frame, text="", bg=self.color_bg, font=('Helvetica', height))
        label.pack(side=tk.TOP, fill=tk.X, pady=0)

    def _init_tools_panel(self):
        """
        Create the toolbar with the speed settings and action buttons.
        Left frame has speed options
        Right frame has buttons.
        """
        # Add spacer label at the top of the tools panel
        self._add_spacer(self._frames['tools'], height=5)
        self.cur_speed_option = 'state-update'
        self.cur_speed_state = 0

        # Create frames for both sides of the tools panel.
        self._frames['tools_left'] = tk.Frame(self._frames['tools'], bg=self.color_bg)
        self._frames['tools_right'] = tk.Frame(self._frames['tools'], bg=self.color_bg)
        self._frames['tools_left'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._frames['tools_right'].pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Add "Speed options" label, left justified at the top:
        label = tk.Label(self._frames['tools_left'], text="Speed options",
                         bg=self.color_bg, font=self.LAYOUT['fonts']['menu'])
        label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=0)

        # Create radio buttons for each speed setting.
        self._radio_buttons = {}
        for i, (mode, states) in enumerate(self._speed_options.items()):
            # Create a radio button for each speed setting.
            rb = tk.Radiobutton(self._frames['tools_left'], text=mode, variable=self.cur_speed_option, value=mode,
                                command=lambda m=mode: self._set_speed_mode(m), bg=self.color_bg,
                                font=self.LAYOUT['fonts']['default'])
            rb.pack(side=tk.TOP, anchor=tk.W, padx=35, pady=0)
            self._radio_buttons[mode] = rb
        # set current radio button to the current speed option.
        self._radio_buttons[self.cur_speed_option].select()
        self._add_spacer(self._frames['tools_left'], height=1)

        # Create the "view options" (states, values, updates) under the speed options:
        label = tk.Label(self._frames['tools_left'], text="View options",
                         bg=self.color_bg, font=self.LAYOUT['fonts']['menu'])
        label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=0)
        # Create radio buttons for each view option.
        for i, mode in enumerate(['states', 'values', 'updates']):
            # Create a radio button for each view setting.
            rb = tk.Radiobutton(self._frames['tools_left'], text=mode, variable=self._view, value=i,
                                command=lambda m=mode: self._set_view_mode(m), bg=self.color_bg,
                                font=self.LAYOUT['fonts']['default'])
            rb.pack(side=tk.TOP, anchor=tk.W, padx=35, pady=0)
            self._radio_buttons[mode] = rb
        # set the current view option to 'states'
        self._radio_buttons['states'].select()

        # self._add_spacer(self._frames['tools_left'], height=1)
        # Create action button, set text to current speed opton's 'text' value.
        self._action_button = tk.Button(self._frames['tools_right'], text=self._speed_options[self.cur_speed_option][self.cur_speed_state]['text'],
                                        command=lambda: self._action(), font=self.LAYOUT['fonts']['buttons'], anchor=tk.CENTER)
        self._action_button.pack(side=tk.TOP, padx=5, pady=5)

        # TODO gamma slider

        # Create tournament start/stop button.
        # self._tournament_button = tk.Button(self._frames['tools_right'], text="Start Tournament" if not self._app.running_tournament else "Stop Tournament",
        #                                    command=lambda: self._toggle_tournament(), font=self.LAYOUT['fonts']['buttons'], anchor=tk.CENTER)
        # self._tournament_button.pack(side=tk.TOP, padx=5, pady=5)

        # Create reset button:
        self._reset_button = tk.Button(self._frames['tools_right'], text="Reset",
                                       command=lambda: self._reset(), font=self.LAYOUT['fonts']['buttons'], anchor=tk.CENTER)
        self._reset_button.pack(side=tk.TOP, padx=5, pady=5)

        # Fullscreen button
        self._fullscreen_button = tk.Button(self._frames['tools_right'], text="Fullscreen",
                                            command=self.toggle_fullscreen, font=self.LAYOUT['fonts']['buttons'], anchor=tk.CENTER)
        self._fullscreen_button.pack(side=tk.TOP, padx=5, pady=5)

    def _set_view_mode(self, mode):
        """
        Change which image is being shown.
        # TODO:  Change title label text.
        """
        logging.info("Setting view mode:  %s" % mode)
        self._view = mode
        self.refresh_images()

    def _toggle_tournament(self):
        """
        Callback for tournament button.
        Toggles the tournament run status and updates the button text.
        """
        self._app.toggle_tournament()
        # self._tournament_button['text'] = "Stop Tournament" if self._app.running_tournament else "Start Tournament"

    def _action(self):
        """
        Callback for action button.
        1. look up the current speed option and state.  
        2. If current speed option has multiple states, cycle through them.
        3. Call the callback function for the current state.
        """
        # Call the callback function for the current state (Before updating state).
        self._speed_options[self.cur_speed_option][self.cur_speed_state]['callback']()

        n_speed_states = len(self._speed_options[self.cur_speed_option])
        new_speed_state = (self.cur_speed_state + 1) % n_speed_states
        logging.info("Updating speed state:  %s in mode %s" % (self.cur_speed_state, self.cur_speed_option))
        self.cur_speed_state = new_speed_state
        # UPdate the button text to the new state.
        self._action_button['text'] = self._speed_options[self.cur_speed_option][self.cur_speed_state]['text']

    def _init_tk(self):
        """
        Initialize the tkinter window and the canvas for drawing.
        """
        self._root = tk.Tk()

        # self._root.attributes('-fullscreen', True)
        # self._root.state('normal') # Optional, but may be required on some systems

        self._root.title("Tic Tac Toe RL Demo")
        # set window to 1200x800 pixels
        self._root.geometry(f"{self._size[0]}x{self._size[1]}")
        self._root.configure(bg=self.color_lines)

    def _set_speed_mode(self, mode):
        """
        Callback for speed setting radio buttons.
        :param mode:  The mode to set.  One of the keys in self._speed_options.
        """
        self.cur_speed_option = mode
        self.cur_speed_state = 0

        # Update the action button text to the first state in the new speed option.
        self._action_button['text'] = self._speed_options[mode][self.cur_speed_state]['text']

    def refresh_text_labels(self):
        """
        Refresh contents of all frames.
        """
        # get fresh status from app & populate labels:
        status = self._app.get_status()
        for stat_name, stat_val in status.items():
            if stat_name not in ['title', 'flag']:
                status_str = "%s:  %s" % (stat_name, stat_val)
                self._status_labels[stat_name]['text'] = status_str
            else:
                self._status_labels[stat_name]['text'] = stat_val

        # Update the action button text to the current speed option's 'text' value.
        self._action_button['text'] = self._speed_options[self.cur_speed_option][self.cur_speed_state]['text']
        # Update the tournament button text to the current tournament status.
        # self._tournament_button['text'] = "Stop Tournament" if self._app.running_tournament else "Start Tournament"

    def refresh_images(self):
        """
        Update screen with appropriate image depending on app state.
        Create a PhotoImage and replace whatever is in the label with it.
        """
        print("Refreshing with %s images, %s view." %
              ("base" if self._app.running_continuous else "annotated", self._view))
        images = self._base_images if self._app.running_continuous else self._disp_images
        image = images[self._view]
        img = ImageTk.PhotoImage(Image.fromarray(image))
        self._img_label.config(image=img)
        self._img_label.image = img

    def annotate_frame(self, vis_update):
        """
        Annotate the current base images.
        """
        self._disp_images = deepcopy(self._base_images)
        vis_update.annotate_images(self._disp_images)
        self.refresh_images()

    def _reset(self):
        logging.info("Resetting demo window...")
        self._app.reset()

        self.cur_speed_state = 0
        # self._recalc_box_positions()
        self.make_images()
        self.refresh_images()

    def _resize(self, event):
        if event.widget == self._frames['values']:
            new_size = (event.width, event.height)
            with self._resize_lock:
                # Check if the size has changed:
                if new_size == self._size:
                    return
                logging.info("Resizing state image: %s  (thread %s)" % (new_size, get_ident()))
                self._size = new_size

                self._recalc_box_positions()
                self.make_images()
                self.refresh_images()
                # TODO: Apply latest PEStep

    def get_box_positions(self):
        return self.box_placer.box_positions

    def start(self):
        """
        Start the demo window.
        """
        self._root.mainloop()


'''

class DemoWindowTester(object):
    """
    fill-in for PolicyImprovmentDemo for testing the GUI.
    """

    def __init__(self, size=(1920, 1080)):
        self._size = size
        self._iter = 37
        self._epoch = 42
        self._running = False  # for continuous mode
        self.running_tournament = False
        from baseline_players import HeuristicPlayer
        self._player = Mark.X
        self._opponent = Mark.O
        self._max_iter = 1000
        self._seed_p = HeuristicPlayer(mark=self._player, n_rules=2, p_give_up=0.0)
        self._opponent_p = HeuristicPlayer(mark=self._opponent, n_rules=6, p_give_up=0.0)
        self._gamma = .9  # discount factor for future rewards.

        logging.info("Starting DemoWindowTester with size:  %s" % (self._size,))

        # P.I. initialization:\
        from reinforcement_base import Environment
        self._env = Environment(self._opponent_p, self._player)
        self.children = self._env.get_children()

        self._pi = self._seed_p
        self.updatable_states = self._env.get_nonterminal_states()
        self.terminal_states = self._env.get_terminal_states()
        speed_options = self.get_speed_options()

        def _get_val(term_state):
            term = term_state.check_endstate()
            if term == Result.X_WIN:  # check self._player here
                return 1.0
            elif term == Result.O_WIN:
                return -1.0
            elif term == Result.DRAW:
                return -.9
            else:
                raise ValueError("Unknown terminal state: %s" % str(term_state))

        # Fake values:
        self._v = {state: _get_val(state) for state in self.terminal_states}
        self._v.update({state: 0.0 for state in self.updatable_states})
        self._v_new = {state: self._v[state]+np.random.randn(1)*.1 for state in self._v}

        self._gui = RLDemoWindow(size, self, speed_options=speed_options, player_mark=Mark.X)

    def get_values(self):
        # return random function.
        return self._v, self._v_new

    def reset(self):
        self._iter = 0
        self._epoch = 0
        self._running = False
        if self.running_tournament:
            self.toggle_tournament()
        self._gui.refresh_text_labels()

    def get_speed_options(self):
        options = OrderedDict()
        options['state-update'] = [{'text': 'Step State', 'callback': lambda: self._step(1)}]
        options['epoch-update'] = [{'text': 'Step Epoch', 'callback': lambda: self._step(0)}]
        options['pi-round'] = [{'text': 'Step Policy', 'callback': lambda: self._step(-1)}]
        options['continuous'] = [{'text': 'Start running...', 'callback': lambda: self._start()},

                                 {'text': '(pause)', 'callback': lambda: self._stop()}]

        return options

    def _step(self, n_steps):
        print(f"Stepping {n_steps} steps")

    def toggle_tournament(self):
        self.running_tournament = not self.running_tournament
        print("Changed tournament run-status:  ", self.running_tournament)

    def _start(self):
        print("Starting continuous...")
        self._running = True

    def _stop(self):
        print("Stopping continuous...")
        self._running = False

    def run(self):
        """
        Run the demo window.
        """
        self._gui._root.mainloop()
        # self._gui._root.destroy()

    def get_status(self):

        status = OrderedDict()
        status['title'] = "Policy Evaluation / Improvement"
        status['phase'] = "Policy Evaluation"
        status['iteration'] = self._iter
        status['epoch'] = self._epoch
        status['states processed'] = "%i of %i" % (self._iter, 8533)
        status['delta-v(s)'] = "nmax delta %.3e (<> %.3e)." % (1.3, 1e-6)
        return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tester = DemoWindowTester()
    tester.run()
'''
