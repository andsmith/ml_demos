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

    def __init__(self, app, env):
        """
        :param app: The main app object.
        :param advance_event: Event to signal when the algorithm should advance.
        """
        self.app = app

        self._env = env
        self._content = self._make_content_manager()  # Get the state/viz image manager for this algorithm.

        self._go_signal = None  # Used to signal the algorithm to advance/continue.
        self._run_control = {option: True for option in self.get_run_control_options()}  # start all options on

        self.paused = True
        self._shutdown = False
        self._learn_thread = None

        # The current state being processed by the algorithm, set before pausing so visualization can show it.
        self.state = None

        self.current_ctrl_pt = self.get_init_control()  # The current control point for the algorithm.
        if self.is_stub():
            raise RuntimeError("This is a stub class. It should not be instantiated directly.")

        self._stop_states = []  # user-set breakpoints, states where the algorithm should pause.

    def get_content_manager(self):
        return self._content
    
    @staticmethod
    @abstractmethod
    def _get_key_info(self):
        """
        Return layout of the key area (upper right) for content tabs.
        :return:  dict('key1_name': {
                            'size': {
                                'width': w, 'height': h}},
                        ... })
        """
        pass

    def _calc_key_placement(self):
        # Full State Tab just uses the state key, on the left in the key area, so sum up the space for the other keys
        keys = self._get_key_info()

        key_spacing = LAYOUT['key_h_pad']
        total_key_height = np.max([k_size['height'] for k_size in keys['sizes'].values()]) + \
            key_spacing * (len(keys['sizes']) - 1)  # no space on

        total_key_width = np.sum([k_size['width'] for k_size in keys['sizes'].values()]) + \
            key_spacing * (len(keys['sizes']) - 1)
        
        x = - total_key_width
        x_offsets = {}
        key_sizes = {}
        for key_name, k_size in keys['sizes'].items():
            x_offsets[key_name] = x
            key_sizes[key_name] = (k_size['width'], total_key_height)
            x += k_size['width'] + key_spacing
            
        total_key_size = (total_key_width, total_key_height)
        return x_offsets, key_sizes, total_key_size

    def toggle_stop_state(self, state):
        if state in self._stop_states:
            self._stop_states.remove(state)
            logging.info("Removed stop state: %s" % state)
        else:
            self._stop_states.append(state)
            logging.info("Added stop state: %s" % state)

    def clear_stop_states(self):
        self._stop_states = []
        logging.info("Cleared all stop states.")
        if self._img_mgr is not None:
            self._img_mgr.clear_selected()

    def get_stop_states(self):
        return self._stop_states

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
            if self.state is not None and self.state in self._stop_states:
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

    @staticmethod
    @abstractmethod
    def get_tab_content():
        """
        ordered dict of {tab_name: TabContentPage object} for the state image panel.
        :return: A dictionary of tab names and their display names.
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
    def get_state_image(self, size, tab_name, is_paused):
        """
        Get the state image for the given tab.
        :param size: (width, height) tuple for the image size.
        :param tab_name: The name of the tab.
        :param is_paused: Whether the algorithm is paused or not.
        :return: The state image for the given tab.
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
