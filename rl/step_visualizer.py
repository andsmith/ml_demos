from abc import ABC, abstractmethod
import numpy as np
import logging
import cv2
from tic_tac_toe import Game, Mark, Result
from colors import get_n_colors, shade_color, NEON_GREEN

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
    def annotate_images(self, images):
        """
        Calculations ran, now we need to display something with the results.  These images disappear.
        (For speed modes that pause between steps.)

        :param images:  dict with {'state','values','updates'} keys, each an image
            in the "Value Function" panel.
        :returns: set of images to display in the GUI.
        """
        pass

    @abstractmethod
    def update_images(self, images):
        """
        Modify the images so they reflect the current state of the algorithm.  These images are permanent.
        (for continuously running speed modes)

        :param images:  dict with {'state','values','updates'} keys, each an image
            in the "Value Function" panel.
        """
        pass

    @abstractmethod
    def draw_step_viz(self, img_size):
        """
        Draw the step visualization in the GUI.

        :param img_size:  size of the image to draw.
        """
        pass


class StateUpdateStep(PEStep):
    """
    Update for a specific state.
    """

    def __init__(self, demo, gui, state, actions, next_states, rewards, old_value, new_value, bg_color=(0,0,0)):
        super().__init__(demo, gui)
        self._state = state  # the state being updated
        self._actions = actions  # possible actions for the state
        self._next_states = next_states  # distribution of next states for each action
        self._rewards = rewards  # reward for each action
        self._old_value = old_value  # old value for the state
        self._new_value = new_value  # new value for the state
        self.delta = new_value - old_value
        self._bg_color = bg_color  # background color for the images

    def annotate_images(self, images):
        """
        1. Draw a thick green box around the state being updated.
        
        2. For the N possible actions, pick N colors.

        3. For each next state, draw a box around it in a shaded color corresponding to the action leading to it.
           each of the next states for each action.

        Do this for all three images.        
        """
        n_actions = len(self._actions)
        n_next_states = [len(self._next_states[a_ind]) for a_ind in range(n_actions)]
        colors = get_n_colors(n_actions)

        shades = [shade_color(c, n_next_states[c_i]) for c_i,c in enumerate(colors)]

        def add_box(state, color, thickness):
            #print("Adding state:\n%s\n" % str(state))
            for img in images.values():
                self._gui.box_placer.draw_box(img, state, color=color, thickness=thickness)

        add_box(self._state, NEON_GREEN, 1)
        for a_ind, action in enumerate(self._actions):
            for s_ind, (next_state, prob) in enumerate(self._next_states[a_ind]):
                add_box(next_state, shades[a_ind][s_ind], 1)

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

    def draw_step_viz(self):
        """
        Draw the state being updated in the top left, the intermediate states resulting from 
        each action distributed in a row under it, and the distribution of next states
        under each intermediate state.

        Under the RL states, print the value function. 
        Under the intermediate states, print the reward values.
        """
        w,h = self._gui.get_step_viz_frame_size()
        logging.info("Creating step visualization with size %i x %i" % (w, h))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = self._bg_color

        #In the top right, draw the updating state in a green box.
        

        # divide the strip into |actions| columns, show the action in the top


        # show each following state under each action in the appropriate color.

        # annotate with value functions, etc.

        return img


class EpochStep(PEStep):
    def __init__(self, demo, gui, state_updates):
        super().__init__(demo, gui)
        self._state_updates = state_updates
        self._epoch = demo._epoch  # the epoch being updated


class PIStep(PEStep):
    def __init__(self, demo, gui, phase, update_info):
        super().__init__(demo, gui)
        self._phase = phase
        self._info = update_info

    def annotate_images(self, images):
        """
        Draw a green box around the state being updated.
        For the N possible actions, pick N colors and draw boxes in those colors around 
        each of the next states for each action.

        Do this for all three images.        
        """
        pass  # TODO:  implement this


class ContinuousStep(PEStep):
    def __init__(self, demo, gui, state_updates):
        super().__init__(demo, gui)
        self._state_updates = state_updates
        self._epoch = demo._epoch  # the epoch being updated

    def annotate_images(self, images):
        """
        Draw a green box around the state being updated.
        For the N possible actions, pick N colors and draw boxes in those colors around 
        each of the next states for each action.

        Do this for all three images.        
        """
        pass  # TODO:  implement this