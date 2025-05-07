"""
Base class for algorithm-specific panel manager for the RL Demo App.
Sublcasses manage 3 panels:
    - status and control, 
    - step visualization, 
    - state/value/update visualization.
"""

from abc import ABC, abstractmethod
import pickle


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
        """
        if self.is_stub():
            raise RuntimeError("This is a stub class. It should not be instantiated directly.")
        self.app = app

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
