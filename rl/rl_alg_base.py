"""
Base class for algorithm-specific panel manager for the RL Demo App.
Sublcasses manage 3 panels:
    - status and control, 
    - step visualization, 
    - state/value/update visualization.
"""

from abc import ABC, abstractmethod
import pickle
from threading import Event
import logging


class DemoAlg(ABC):
    """
    Base class for algorithm-specific panel manager for the RL Demo App.
    Subclasses manage 3 panels:
        - status and control, 
        - step visualization, 
        - state/value/update visualization.
    """

    def __init__(self, app, state_panel):
        """
        :param app: The main app object.
        """
        self._go_signal = Event()  # Used to signal the algorithm to advance/continue.
        self._go_signal.clear()  # Initially, the algorithm is not running.
        self._run_control = {option: True for option in self.get_run_control_options()}  # start all options on
        self._state_panel = state_panel  # The panel for displaying the state of the algorithm.
        self.paused=True  # Initially, the algorithm is paused.

        if self.is_stub():
            raise RuntimeError("This is a stub class. It should not be instantiated directly.")
        self.app = app
        
    def set_state_panel(self, state_panel):
        self._state_panel = state_panel  # The panel for displaying the state of the algorithm.

    @abstractmethod
    def start(self):
        """
        Start the algorithm running in a separate thread and return.
        At every point it can be paused (the control points), call self._maybe_pause(control_point)
        """
        pass

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
        self._run_control=new_rcs

    @abstractmethod
    def get_state_image(self, size, which, is_paused=False):
        """
        Draw a new image and send it to the state panel. 
        Get the current tab from self._state_panel.cur_tab.
        Re-draw the image (if state changes, etc) and call self._state_panel.set_image() with it.

        Will be caused by _maybe_pause to draw something while waiting,
        or by the gui if it wants a new image.

        :param control_point: The control point at which to refresh the image.
        :param is_paused: Whether the algorithm is paused or not.
        :param force_redraw: Whether to force a redraw of the image or not (will be true if window size changes, etc).
        """
        pass

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
        :returns: list of (option-key, option-string tuples).  Keys are for 
        references w/the gui, strings are for displaying in checkboxes.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_state_image_kinds():
        """
        What are the tab names for the state-visualization panel?
        e.g. "State", "Value", "Update", etc.

        :returns: OrderedDict of (option-key, option-string tuples).  Option-keys are for 
        references w/the gui, option-strings are for displaying as the tab labels.
        """
        pass

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


    def reset_state(self):
        self._reset_state()
        if self._state_panel is not None:
            for option in self.get_run_control_options():
                self._run_control = {option: True for option in self.get_run_control_options()} 
                self._state_panel.change_run_control_settings(self._run_control)  # Update the run control options in the state panel.

        

    @abstractmethod
    def _reset_state(self):
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
