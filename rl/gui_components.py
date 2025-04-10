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
from collections import OrderedDict
import logging
import numpy as np
from drawing import GameStateArtist
from tic_tac_toe import Game, get_game_tree_cached
from node_placement import FixedCellBoxOrganizer
from game_base import Result, Mark
from colors import COLOR_BG, COLOR_LINES, RED, GREEN, MPL_BLUE_RGB, MPL_GREEN_RGB, MPL_ORANGE_RGB
import tkinter as tk

def tk_color_from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
       https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


STATUS_W = 0.25
FUNC_W = (1.0 - STATUS_W) / 2.0 


class RLDemoWindow(object):
    """
    Manage images and interactivity, don't do learning.
    """
    # fractions of window width & height for the different panels.
    # Use Tkinter place mode to set each of these panels' frames.
    LAYOUT = {'frames': {'status': {'x_rel': (0, STATUS_W),
                                    'y_rel': (0, 0.25)},
                         'tools': {'x_rel': (0, STATUS_W),
                                   'y_rel': (0.25, .5)},
                         'tournament': {'x_rel': (0, .25),
                                        'y_rel': (0.5, 1.)},
                         'values': {'x_rel': (STATUS_W, STATUS_W + FUNC_W),
                                    'y_rel': (0, 0.666)},
                         'updates': {'x_rel': (STATUS_W + FUNC_W, STATUS_W + 2 * FUNC_W),
                                     'y_rel': (0, 0.666)},
                         'step_viz': {'x_rel': (0.25, 1),
                                      'y_rel': (.666, 1.)}},
              'margin_px': 5,
              'fonts': {'pannel_title': ('Helvetica', 16),
                        'title': ('Helvetica', 14, 'bold'),
                        'default': ('Helvetica', 12)}}

    FRAME_TITLES = {'status': 'Status',
                    'tools': 'Tools',
                    'tournament': 'TTT Tournament',
                    'values': 'Value Function',
                    'updates': 'Delta V(s)',
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
        self._color_bg = tk_color_from_rgb(COLOR_BG)
        self._color_lines = tk_color_from_rgb(COLOR_LINES)
        # Create the game tree and the artist for drawing it.
        self._tree = get_game_tree_cached(player=player_mark)
        self._artist = GameStateArtist(space_size=1, bar_w_frac=0.0)  # no padding for value representations.

        #self._bboxes = self._calc_bboxes()
        #import pprint
        #pprint.pprint(self._bboxes)
        self._init_tk()
        self._init_frames()
        self._init_status()
        #self._init_tools()
        #self._init_values_and_updates()
        #self._init_tournament()
        #self._init_step_viz()
        
    def _init_step_viz(self):
        """
        For now just add a label that says "Step visualization"
        """
        self._step_viz_frame = self._frames['step_viz']
    
    
    def _init_values_and_updates(self):
        """
        Create labels for the panel titles "V_t(s)" and "V_{t+1}(s)" at the top center.
        Create Canvases for drawing the two value function visualizations that extend to the bottom of the window.
        """

        # Create canvases for drawing the value functions
        self._canvas_values = tk.Canvas(self._frames['values'], bg=self._color_lines)
        self._canvas_values.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._canvas_updates = tk.Canvas(self._frames['updates'], bg=self._colo_color_linesr_bg)
        self._canvas_updates.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _init_status(self):
        """
        Create labels for the status panel.
        These are left-justified in the status frame.
        Display the key and value like "Key:  Value" for each status item.
        """
        self._status_labels = {}
        status = self._app.get_status()
        import pprint
        pprint.pprint(status)
        # Add a spacer label at the top of the status panel
        label = tk.Label(self._frames['status'], text="", bg=self._color_bg, font=self.LAYOUT['fonts']['pannel_title'])
        label.pack(side=tk.TOP, fill=tk.X)
        for stat_name, stat_val in status.items():
            if stat_name != 'title' :
                status_str = "%s:  %s" % (stat_name, stat_val)
                font = self.LAYOUT['fonts']['default']
            else:
                status_str = stat_val
                font = self.LAYOUT['fonts']['title']
            label = tk.Label(self._frames['status'], text=status_str, bg=self._color_bg, font=font, anchor=tk.W, justify=tk.LEFT)
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

        print("Relative pads: ", x_pad_rel, y_pad_rel)


        for name, layout in self.LAYOUT['frames'].items():
            
            x_rel = layout['x_rel']
            y_rel = layout['y_rel']
            font = self.LAYOUT['fonts']['pannel_title']
            frame = tk.Frame(self._root, bg=self._color_bg)
            frame.place(relx=x_rel[0]+x_pad_rel, rely=y_rel[0]+y_pad_rel, relwidth=x_rel[1] - x_rel[0] - 2*x_pad_rel,
                        relheight=y_rel[1] - y_rel[0]- 2*y_pad_rel)
            label = tk.Label(frame, text=self.FRAME_TITLES[name], bg=self._color_bg, font=font)
            label.pack(side=tk.TOP, fill=tk.X)
            # Add dark line under label
            tk.Frame(frame, height=2, width=100, bg=self._color_lines).pack(side=tk.TOP)
            
            self._frames[name] = frame 

    def _init_tools(self):
        """
        Create the toolbar with the speed settings and action buttons.
        """
        self._frames['tools'].pack(side=tk.BOTTOM, fill=tk.X)

        # Create radio buttons for speed settings
        self._speed_var = tk.StringVar(value=list(self._speed_options.keys())[0])
        for mode in self._speed_options.keys():
            rb = tk.Radiobutton(self._frames['tools'], text=mode, variable=self._speed_var,
                                value=mode, command=lambda m=mode: self._set_speed_mode(m))
            rb.pack(side=tk.LEFT)

        self._cur_speed_option = list(self._speed_options.keys())[0]  # default to first speed option
        self._cur_speed_state = 0  # default to first state in the current speed option (which may only have one)

        # Create action button, set text to current speed opton's 'text' value.
        self._action_button = tk.Button(self._frames['tools'], text=self._speed_options[self._cur_speed_option][self._cur_speed_state]['text'],
                                        command=lambda: self._action())

    def _action(self):
        """
        Callback for action button.
        1. look up the current speed option and state.  
        2. If current speed option has multiple states, cycle through them.
        3. Call the callback function for the current state.
        """
        n_speed_states = len(self._speed_options[self._cur_speed_option])
        new_speed_state = (self._cur_speed_state + 1) % n_speed_states
        print("Updating speed state:  %s in mode %s", (self._cur_speed_state, self._cur_speed_option))
        self._cur_speed_state = new_speed_state
        # UPdate the button text to the new state.
        self._action_button['text'] = self._speed_options[self._cur_speed_option][self._cur_speed_state]['text']
        # Call the callback function for the current state.
        self._speed_options[self._cur_speed_option][self._cur_speed_state]['callback']()

    def _init_tk(self):
        """
        Initialize the tkinter window and the canvas for drawing.
        """
        self._root = tk.Tk()
        self._root.title("Tic Tac Toe RL Demo")
        # set window to 1200x800 pixels
        self._root.geometry(f"{self._size[0]}x{self._size[1]}")
        self._root.configure(bg=self._color_lines)

    def _set_speed_mode(self, mode):
        """
        Callback for speed setting radio buttons.
        :param mode:  The mode to set.  One of the keys in self._speed_options.
        """
        self._cur_speed_option = mode
        self._cur_speed_state = 0
        # Update the action button text to the first state in the new speed option.

    def _calc_bboxes(self):
        """
        Get absolute (pixels) of each panel in the window.
        :return:  dict of bboxes for each panel.
        """
        bboxes = {}
        for name, layout in self.LAYOUT['frames'].items():
            x_rel = layout['x_rel']
            y_rel = layout['y_rel']
            bboxes[name] = {'x_min': int(x_rel[0] * self._size[0] + self.LAYOUT['margin_px']),
                            'x_max': int(x_rel[1] * self._size[0] - self.LAYOUT['margin_px']),
                            'y_min': int(y_rel[0] * self._size[1] + self.LAYOUT['margin_px']),
                            'y_max': int(y_rel[1] * self._size[1] - self.LAYOUT['margin_px'])}
        return bboxes

    def refresh_status(self):
        """
        Refresh contents of all frames.
        """
        status = self._app.get_status()
        self._status_labels['title']['text'] = status['title']
        self._status_labels['phase']['text'] = status['phase']
        self._status_labels['iteration']['text'] = "Iteration: %d" % status['iteration']
        self._status_labels['status_str']['text'] = status['status_str']
        # Update the action button text to the current speed option's 'text' value.
        self._action_button['text'] = self._speed_options[self._cur_speed_option][self._cur_speed_state]['text']
    

class DemoWindowTester(object):
    """
    fill-in for PolicyImprovmentDemo for testing the GUI.
    """

    def __init__(self, size=(1900, 950)):
        self._size = size        
        self._iter = 37
        self._epoch = 42
        self._running = False  # for continuous mode
        speed_options = self.get_speed_options()
        self._gui = RLDemoWindow( size, self,speed_options=speed_options, player_mark=Mark.X)

    def get_speed_options(self):
        options = {'state-update': [{'text': 'Step State', 'callback': lambda: self._step(1)}],
                   'epoch-update': [{'text': 'Step Epoch', 'callback': lambda: self._step(0)}],
                   'pi-iteration': [{'text': 'Step Policy', 'callback': lambda: self._step(-1)}],
                   'continuous':  [{'text': 'Start running...', 'callback': lambda: self._start_stop(True)},
                                   {'text': '(pause)', 'callback': lambda: self._start_stop(False)}]}
        return options

    def _step(self, n_steps):
        print(f"Stepping {n_steps} steps")

    def _start_stop(self, running_state):
        self._running = running_state
        print("New state: ", "Running" if running_state else "Paused")

    def run(self):
        """
        Run the demo window.
        """
        self._gui._root.mainloop()
        self._gui._root.destroy()

    def get_status(self):

        status = OrderedDict()
        status['title'] = "Policy Evaluation / Improvement"
        status['phase'] = "Policy Evaluation"
        status['iteration'] = self._iter
        status['epoch'] = self._epoch
        status['states processed'] = "%i of %i" % (self._iter, 8533)
        status['delta-v(s)']="nmax delta %.3e (<> %.3e)." % (1.3, 1e-6)
        return status

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tester = DemoWindowTester()
    tester.run()
