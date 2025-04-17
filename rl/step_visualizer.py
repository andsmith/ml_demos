from abc import ABC, abstractmethod
import numpy as np
import logging
import cv2
from tic_tac_toe import Game, Mark, Result


class PEStep(ABC):
    """
    Represent the step (state update, epoch, PI round, or continuously running).  There should be 1 subclass per speed setting.

    Demo will generate these, send them to the GUI, which will apply them as frames are needed and then forget them.

    Include all information needed to update the GUI & methods to alter the state/value/update images & 
    render the step visualization.  GUI will call these when it needs updates.
    """

    def __init__(self, demo, gui):
        self._demo = demo
        self._gui = gui

    @abstractmethod
    def get_annotated_images(self, images):
        """
        Calculations ran, now we need to display something with the results.  These changes disappear.
        (For speed modes that pause between steps.)

        :param images:  dict with {'state','values','updates'} keys, each an image
            in the "Value Function" panel.
        :returns: set of images to display in the GUI.
        """
        pass

    @abstractmethod
    def update_images(self, images):
        """
        Modify the images so they reflect the current state of the algorithm.  These changes are permanent.
        (for continuously running speed modes)

        :param images:  dict with {'state','values','updates'} keys, each an image
            in the "Value Function" panel.
        """
        pass

    @abstractmethod
    def draw_step_vis(self, img_size):
        """
        Draw the step visualization in the GUI.

        :param img_size:  size of the image to draw.
        """
        pass


class StateUpdateStep(PEStep):
    """
    Update for a specific state.
    """

    def __init__(self, demo, gui, state, actions, next_states, rewards, old_value, new_value):
        super().__init__(demo, gui)
        self._state = state  # the state being updated
        self._actions = actions  # possible actions for the state
        self._next_states = next_states  # distribution of next states for each action
        self._rewards = rewards  # reward for each action
        self._old_value = old_value  # old value for the state
        self._new_value = new_value  # new value for the state
        self.delta = new_value - old_value

    def annotate_images(self, images):
        """
        Draw a green box around the state being updated.
        For the N possible actions, pick N colors and draw boxes in those colors around 
        each of the next states for each action.

        Do this for all three images.        
        """
        pass

    def update_images(self, images):
        """
        Draw the new color in the updates image.
        Leave the other images untouched.
        """

        # update the state image with the new values:
        self._gui.update_state_image(images['state'])
        # update the value function image with the new values:
        self._gui.update_value_image(images['values'])
        # update the updates image with the new values:
        self._gui.update_updates_image(images['updates'])

    def make_step_vis(self, img_size):
        """
        Draw the state being updated in the top left, the intermediate states resulting from 
        each action distributed in a row under it, and the distribution of next states
        under each intermediate state.

        Under the RL states, print the value function. 
        Under the intermediate states, print the reward values.
        """
        pass  # TODO:  implement this


class EpochStep(PEStep):
    def __init__(self, demo, gui, state_updates):
        super().__init__(demo, gui)
        self._state_updates = state_updates
        self._epoch = demo._epoch  # the epoch being updated
