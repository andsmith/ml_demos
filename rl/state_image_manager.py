import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_BG, SKY_BLUE
from layout import LAYOUT, WIN_SIZE
from game_base import Mark, Result
from game_util import get_box_placer, get_state_icons, sort_states_into_layers
from layer_optimizer import SimpleTreeOptimizer
import numpy as np
from tic_tac_toe import Game
import time
import logging
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from drawing import GameStateArtist
# BOX_SIZES = [22, 12, 12, 12, 12, 12]  # good for single value function
SPACE_SIZES = [7, 2, 2, 2, 2, 3]


class StateImageManager(ABC):
    def __init__(self, env, tabs):
        self._size = None
        self.tabs = tabs
        self._env = env
        self.terminal_states = env.get_terminal_states()
        self.updatable_states = env.get_nonterminal_states()
        self.all_states = self.updatable_states + self.terminal_states
        self.states_by_layer = sort_states_into_layers(self.all_states)
        logging.info(f"StateImageManager initialized with {len(self.all_states)} states")
        self.clear()

    def clear(self):
        # re-generate these JIT.
        self._base_images = {tab: None for tab in self.tabs}  # state icons / values
        self._disp_images = {tab: None for tab in self.tabs}  # base images + annotatations (selected/mouseovered boxes)

    def set_size(self, new_size):
        logging.info(f"StateImageManager resized to {new_size}")
        self.clear()
        self._size = new_size
        self._calc_dims()

    @abstractmethod
    def _calc_dims(self):
        """
        Recalculate placement of boxes, etc.
        """
        pass

    @abstractmethod
    def get_tab_img(self, tab, annotated=True):
        pass


class ModelBasedSIM(StateImageManager):
    """
    State Image Manager for model-based RL algorithms.
    All states are initially visible, values are updated, etc.
    """

    def __init__(self, env, tabs, value_ranges, colormap_names, bg_colors):
        """
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)

          NOTE:  If one of the tabs is named "states" the state icons will be drawn in that image.
                 All other tabs will be associated with a state:value mapping.

        :param value_ranges:  dict(tab_name: (min, max)) for each tab (defaults to (-1, 1))
        :param colormap_names:  dict(tab_name: colormap name) for each tab (defaults to 'viridis')
        :param bg_colors:  dict(tab_name: color) for each tab (defaults COLOR_BG for states, SKY_BLUE for values/updates)
        """
        super().__init__(env, tabs)
        self._cmaps = {tab: plt.get_cmap(colormap_names[tab]) for tab in tabs if tab != 'states'}
        self._ranges = value_ranges
        self._state_icons, self._box_sizes = self._get_state_icons()
        self._values = {tab: {state: np.nan for state in self.all_states} for tab in tabs if tab != 'states'}
        self._bg_colors = bg_colors

    def get_color(self, tab, value):
        """
        Get the color for a value in a tab.
        :param tab:  The tab to get the color for.
        :param value:  The value to get the color for.
        :return:  The color for the value in the tab.
        """
        if tab not in self._cmaps:
            raise ValueError(f"Tab {tab} not found in {self._cmaps.keys()}")
        cmap = self._cmaps[tab]
        min_val, max_val = self._ranges[tab]
        norm_value = (value - min_val) / (max_val - min_val)

        color = cmap(norm_value)

        return int(color[0]*255), int(color[1]*255), int(color[2]*255)

    def set_state_val(self, state, tab, value):
        """
        Set the value for a state in a tab.
        :param state:  The state to set the value for.
        :param tab:  The tab to set the value for.
        :param value:  The value to set.
        """
        if tab not in self._values:
            raise ValueError(f"Tab {tab} not found in {self._values.keys()}")
        self._values[tab][state] = value
        if self._disp_images[tab] is not None:
            color = self.get_color(tab, value)
            self._box_placer.draw_box(image=self._disp_images[tab],
                                      state_id=state,
                                      color=color,
                                      thickness=0)  # filled box

    def _calc_dims(self):
        """
        Calculate state layout.
        """
        self._box_placer, _, _ = get_box_placer(self._size,
                                                self.all_states,
                                                box_sizes=self._box_sizes,
                                                layer_vpad_px=1,
                                                layer_bar_w=1,
                                                player=self._env.player)
        terminal_lut = {state: state.check_endstate() for state in self.all_states}
        tree_opt = SimpleTreeOptimizer(image_size=self._size,
                                       states_by_layer=self.states_by_layer,
                                       state_positions=self._box_placer.box_positions,
                                       terminal=terminal_lut)
        new_positions = tree_opt.get_new_positions()
        self._box_placer.box_positions = new_positions

    def _get_state_icons(self):
        artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in SPACE_SIZES]
        images = {state['id']: artists[layer_num].get_image(
            state['id']) for layer_num, state_list in enumerate(self.states_by_layer) for state in state_list}
        box_sizes = [artists[layer_no].get_image(Game()).shape[0] for layer_no in range(len(artists))]
        return images, box_sizes

    def _check_images_valid(self):
        """
        Make sure all images that are not None are the right size.
        """
        for tab in self.tabs:
            if self._base_images[tab] is not None:
                if self._base_images[tab].shape[:2] != self._size[::-1]:
                    logging.info(f"Base image for {tab} is not the right size:  {self._base_images[tab].shape[:2]} != {self._size[::-1]}")
                    return False
            if self._disp_images[tab] is not None:
                if self._disp_images[tab].shape[:2] != self._size[::-1]:
                    logging.info(f"Display image for {tab} is not the right size:  {self._disp_images[tab].shape[:2]} != {self._size[::-1]}")
                    return False
        return True

    def get_tab_img(self, tab, annotated=True):
        print("Getting tab image for", tab)
        if not self._check_images_valid():
            logging.info("Invalidating images, will regenerate.")
            self.clear()
        if annotated:
            if self._disp_images[tab] is None:
                self._disp_images[tab] = self.draw_annotated(tab)
            return self._disp_images[tab]
        else:
            if self._base_images[tab] is None:
                self._base_images[tab] = self.draw_base(tab)
            return self._base_images[tab]

    def draw_base(self, tab):
        logging.info(f"Regenerating base image for {tab}")
        img = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        img[:] = COLOR_BG

        if tab == 'states':
            img = self._box_placer.draw(images=self._state_icons, colors=None, show_bars=True, dest=img)
        else:
            state_colors = {state: self.get_color(tab, self._values[tab][state]) for state in self.all_states}
            img = self._box_placer.draw(images=None, colors=state_colors, show_bars=True, dest=img)
        return img


class PolicyEvalSIM(ModelBasedSIM):
    def __init__(self, env):
        tabs = ['states', 'values', 'updates']
        colormap_names = {'values': 'gray', 'updates': 'coolwarm'}
        value_ranges = {'values': (-1, 1), 'updates': (-1, 1)}
        bg_colors = {'states': COLOR_BG, 'values': SKY_BLUE, 'updates': SKY_BLUE}
        super().__init__(env, tabs, value_ranges, colormap_names, bg_colors)


class SimTester(object):
    """
    Simple TK app with 3 tabs, 1 frame in each (the state image).
    """

    def __init__(self, size, env):
        # args for Policy Eval tabs:

        self.current_tab_ind = 0
        self._init_tk(size)
        self._init_images(env)
        self._init_tabs()

    def _on_resize(self, event):
        # check if is a resize event
        new_size = (event.width, event.height)
        if new_size == self._size:
            return
        logging.info(f"Window resized to {new_size}")
        self._sim.set_size(new_size)
        self._size = new_size
        self.refresh_image()

    def _init_tk(self, size):
        self._size = size
        self._root = tk.Tk()
        self._root.title("State Image Manager")
        self._root.geometry(f"{size[0]}x{size[1]}")
        self._root.resizable(True, True)
        self._root.bind("<Configure>", self._on_resize)

    def _init_tabs(self):
        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._tab_images = {}
        for name in self._tabs:
            tab_frame = tk.Frame(self._notebook)
            tab_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            img_label = tk.Label(tab_frame)
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self._notebook.add(tab_frame, text=name)
            self._tab_images[name] = (tab_frame, img_label)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        current_tab = self._tabs[self.current_tab_ind]
        self._notebook.select(self._tab_images[current_tab][0])

    def _init_images(self, env):
        self._sim = PolicyEvalSIM(env)
        for state in self._sim.all_states:
            self._sim.set_state_val(state, 'values', np.random.rand()*2 - 1)
            self._sim.set_state_val(state, 'updates', np.random.rand()*2 - 1)
        self._tabs = self._sim.tabs

    def _on_tab_changed(self, event):
        new_tab = self._notebook.tab(self._notebook.select(), "text")
        new_tab_ind = self._tabs.index(new_tab)
        if new_tab_ind != self.current_tab_ind:
            self.current_tab_ind = new_tab_ind
            logging.info(f"Tab changed to {new_tab}")
            self.refresh_image()
        else:
            logging.warning(f"Tab {new_tab} already selected!")

    def refresh_image(self):
        print("Refreshing image at size:", self._size)
        current_tab = self._tabs[self.current_tab_ind]
        img = self._sim.get_tab_img(current_tab, annotated=False)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        label = self._tab_images[current_tab][1]
        label.config(image=img)
        label.image = img
        label.update_idletasks()

    def start(self):
        self._root.mainloop()


def test_state_image_manager(init_size=(1100, 980)):
    """
    Create a tk window, let the SIM know when it is resized, etc.
    """
    from baseline_players import HeuristicPlayer
    from reinforcement_base import Environment
    win_name = "State Image Manager"
    AGENT_MARK = Mark.X
    OPPONENT_MARK = Mark.O
    opp_policy = HeuristicPlayer(mark=OPPONENT_MARK, n_rules=2)
    env = Environment(opp_policy, AGENT_MARK)

    window = SimTester(init_size, env)
    window.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_state_image_manager()
