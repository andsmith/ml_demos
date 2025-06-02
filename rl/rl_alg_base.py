"""
Base class for algorithm-specific panel manager for the RL Demo App.
Sublcasses manage 3 panels:
    - status and control, 
    - step visualization, 
    - state/value/update visualization.
"""
import time
from abc import ABC, abstractmethod
import pickle
from threading import Event, Thread, get_ident
import logging
from layout import LAYOUT
import numpy as np

# from loop_timing.loop_profiler import LoopPerfTimer as LPT


class DemoAlg(ABC):
    """
    Base class for algorithm-specific panel manager for the RL Demo App.
    Subclasses manage 3 panels:
        - status and control, 
        - step visualization, 
        - state/value/update visualization.
    """

    def __init__(self, app):
        """
        :param app: The main app object.
        :param advance_event: Event to signal when the algorithm should advance.
        """
        if self.is_stub():
            raise RuntimeError("This is a stub class. It should not be instantiated directly.")

        self.app = app
        self._go_signal = None  # Used to signal the algorithm to advance/continue.
        self._run_control = {option: True for option in self.get_run_control_options()}  # start all options on
        self.paused = True
        self._shutdown = False
        self._learn_thread = None

        self._viz_img_size = None  # Size of the visualization image, set when the step-visualization panel is resized.
        self._state_img_size = None  # Size of the state image, set when the state-tabs panel is resized.   

        # The current state being processed by the algorithm, set before pausing so visualization can show it.
        self.state = None

        # The current control point for the algorithm.
        self.current_ctrl_pt = self.get_init_control()  
        self._tabs = self._make_tabs()


    def _calc_key_placement(self,key_sizes, key_spacing=None):
        # Full State Tab just uses the state key, on the left in the key area, so sum up the space for the other keys
        
        key_spacing = LAYOUT['keys']['h_pad_px'] if key_spacing is None else key_spacing
        total_key_height = np.max([k_size['height'] for k_size in key_sizes.values()]) 
        total_key_width = np.sum([k_size['width'] for k_size in key_sizes.values()]) + \
            key_spacing * (len(key_sizes) - 1)
        
        x = - total_key_width
        x_offsets = {}

        for key_name, k_size in key_sizes.items():
            x_offsets[key_name] = x
            key_sizes[key_name] = (k_size['width'], total_key_height)
            x += k_size['width'] + key_spacing
            
        total_key_size = (total_key_width, total_key_height)

        return x_offsets, key_sizes, total_key_size

    def advance(self):
        """
        Run 1 step (depending on the run control options), or continuously, etc.
        """
        self._go_signal.set()  # Signal the algorithm to continue running.

    def update_run_control(self, new_rcs):
        """
        Update the run control options.
        :param run_control: The run control options dict(option: bool) (see alg_panels.StatusControlPanel)
        """
        self._run_control = new_rcs

    def _maybe_pause(self, control_point):
        """
        Pause the algorithm if the run control indicates to do so.
        :param control_point: The control point to check.
        """

        self.current_ctrl_pt = control_point

        do_pause = self._run_control[control_point]

        if 'stops' in self._run_control and self._run_control['stops']:
            if self.state is not None and self.state in self.app.selected_states:
                do_pause = True
                logging.info("Stopping at user-set stop state: %s" % self.state)

        if do_pause:
            self.paused = True
            self.app.tick(is_paused=True, control_point=control_point)
            logging.info("Algorithm paused at control point: %s" % control_point)
            self._go_signal.wait()
            self._go_signal.clear()
        else:
            self.paused = False
            self.app.tick(is_paused=False, control_point=control_point)

        return self._shutdown  # was this changed while paused?

    @staticmethod
    def is_stub():
        return True  # Override and return False in subclasses to indicate that this can be run in the demo app.

    @staticmethod
    @abstractmethod
    def get_name():
        """
        Return a short string to be used as a key to refer to this algorithm.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_run_control_options():
        """
        Return a list of options to be used in the run-control panel.
        NOTE:  A special option 'stops' is used to indicate that the algorithm should pause when processing this state.
        :returns: list of (option-key, option-string tuples).  Keys are for 
        references w/the gui, strings are for displaying in checkboxes.
        """
        pass

    @abstractmethod
    def _make_tabs(self, ):
        """
        :return: ordered dict of tab_name: TabContentBage pairs.
        """
        pass

    def get_init_control(self):
        """
        Return the initial control point for the algorithm.
        :return: The initial control point.
        """
        options = self.get_run_control_options()
        names = [name for name in options]
        return names[0]

    @staticmethod
    @abstractmethod
    def get_str():
        """
        Return a string to be used to display this algorithm in the selection frame.
        """
        pass

    @abstractmethod
    def load_state(state_file):
        """
        Load the algorithm state from a file.
        :param app: The application object.
        :param state_file: The state file to load.
        :return: An instance of the algorithm.
        """
        # determine the subclass, and call it's load function.

    @abstractmethod
    def save_state(self, state_file, clobber=False):
        """
        Save the current state of the algorithm to a file.
        :param app: The application object.
        :param state_file: The file to save the state to.
        """
        pass

    @abstractmethod
    def _learn_loop(self):
        """
        This function runs the algorithm in a separate thread.

        At every point it can be paused (the control points), call self._maybe_pause('control_point_name').
        Stop when self._shutdown is set to True or if _maybe_pause returns True.
        """
        pass

    def start(self, advance_event):
        self._go_signal = advance_event
        self._go_signal.clear()
        self._learn_thread = Thread(target=self._learn_loop, daemon=True)
        self._learn_thread.start()
        logging.info("Algorithm thread started.")

    def stop(self):
        if self._go_signal is not None:
            self._go_signal.set()  # Signal the algorithm to stop.

        self._shutdown = True
        if self._learn_thread is not None:
            logging.info("Waiting for algorithm thread to stop...")
            for _ in range(3):
                self._learn_thread.join(timeout=1)
                if self._go_signal is not None:
                    self._go_signal.set()  # Signal the algorithm to stop.
            time.sleep(1)

            self._learn_thread.join()

            logging.info("Algorithm thread stopped.")
        else:
            logging.info("Algorithm thread was not started.")
        self._go_signal = None

    @abstractmethod
    def reset_state(self):
        """
        Reset the algorithm to its initial state.
        """
        pass

    @abstractmethod
    def get_status(self):
        """
        Get the current status of the algorithm (for display in the status panel).
        :return: list of (text, font) tuples to be displayed in the status panel.
        """
        pass

    @abstractmethod
    def get_viz_image(self, size, control_point, is_paused):
        """
        Get the visualization image for the given tab.
        :param size: (width, height) tuple for the image size.
        :param control_point: String ('state', 'epoch', etc. what to visualize).
        :param is_paused: Whether the algorithm is paused or not.
        :return: The visualization image for the given tab.
        """
        pass

    def check_file_type(self, file):
        """
        Read the first pickled item, make sure it matches the algorithm name.
        """
        alg_name = pickle.load(file)
        if alg_name != self.get_name():
            raise ValueError(f"Load state name mismatch: expected {self.get_name()}, got {alg_name}")

    def mark_file_type(self, file):
        """
        Write the algorithm name to the file.
        """
        print("Marking file type: ", self.get_name())
        pickle.dump(self.get_name(), file)

    def resize(self, panel, new_size):
        """
        :param panel:  either 'state-tabs'  or  'step-visualization'
        :param new_size:  (width, height) of the panel in pixels.
        """
        if panel == 'step-visualization':
            if self._viz_img_size is None or self._viz_img_size != new_size:
                self._viz_img_size = new_size
                logging.info(f"Resized step-visualization panel to {new_size} pixels.")
        elif panel == 'state-tabs':
            self._state_img_size = new_size
            for tab in self._tabs:
                self._tabs[tab].resize(new_size)
        else:
            raise ValueError(f"Unknown panel type: {panel}. Expected 'state-tabs' or 'step-visualization'.")

        