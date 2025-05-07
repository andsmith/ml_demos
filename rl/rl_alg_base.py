"""
Base class for algorithm-specific panel manager for the RL Demo App.
Sublcasses manage 3 panels:
    - status and control, 
    - step visualization, 
    - state/value/update visualization.
"""

from abc import ABC, abstractmethod, abstractstaticmethod


class DemoAlg(ABC):
    """
    Base class for algorithm-specific panel manager for the RL Demo App.
    Subclasses manage 3 panels:
        - status and control, 
        - step visualization, 
        - state/value/update visualization.
    """

    def __init__(self, app):
        self.app = app
        self.is_stub=True  # will be grayed out in algorithm selection frame
        
        self._init_frames()  # initialize the algorithm-specific panels
        
            
    @staticmethod
    @abstractmethod
    def get_name():
        """
        Return a short string to be used as a key to refer to this algorithm.
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
    def _init_frames(self):
        """
        Initialize the algorithm-specific panels.
        """
        pass