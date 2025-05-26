import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_BG, SKY_BLUE, NEON_GREEN, NEON_RED
from layout import LAYOUT, WIN_SIZE
from game_base import Mark, Result, TERMINAL_REWARDS
from game_util import get_box_placer, get_state_icons, sort_states_into_layers
from layer_optimizer import SimpleTreeOptimizer
import numpy as np
from tic_tac_toe import Game
import time
import logging
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from drawing import GameStateArtist
from scipy.spatial import KDTree
# BOX_SIZES = [22, 12, 12, 12, 12, 12]  # good for single value function
SPACE_SIZES = [7, 2, 2, 2, 2, 3]


class StateImageManager(ABC):
    """
    Base class for images that go into the state panel tabs.
    There is a tiny icon for each possible RL state, can used directly to make the image, or
       a colored box can be drawn in its place to represent a value, etc.

    This class also allows mouse interaction, selecting/unselecting/mouseovering states for
    breakpoints in the algorithm loop.
    """

    def __init__(self, app, env, tabs):
        self._app = app
        self._size = None
        self.tabs = tabs
        self._env = env
        self._cur_state = None  # highlight in images

        self._box_placer = None
        self._box_centers = None
        self._box_tree = None

        self.terminal_states = env.get_terminal_states()
        self.updatable_states = env.get_nonterminal_states()
        self.all_states = self.updatable_states + self.terminal_states
        self.states_by_layer = sort_states_into_layers(self.all_states)
        self.layers_per_state = {state['id']: i for i, layer in enumerate(self.states_by_layer) for state in layer}
        self._state_icons, self._box_sizes = self._get_state_icons()

        self.mouseovered = None
        self.selected = []

        # Base images for states/values, updated as algorithm runs.
        self._base_images = {tab: None for tab in tabs}
        # Which states are selected/moused over, etc., drawn over base iamges
        self._marked_images = {tab: None for tab in tabs}
        # Images for displaying states with annotations of the learnining algorithm's step, drawn over marked images.
        self._disp_images = {tab: None for tab in tabs}

        self._size_cache = {}

        logging.info(f"StateImageManager initialized with {len(self.all_states)} states")
        self.clear_images()

    def clear_images(self, tabs=None, marked_only=False):
        """
        Clear the images for the tabs.
        :param tabs:  list of tabs to clear, or None to clear all.
        :param marked_only:  If True, only clear the marked/display images (e.g. if a mouse interaction occurs).
        """
        # print("Clearing state images for tabs: %s (Disp only:  %s)" % (tabs, disp_only))
        tabs_to_clear = self.tabs if tabs is None else tabs
        tabs_to_clear = (tabs_to_clear,) if isinstance(tabs_to_clear, str) else tabs_to_clear

        for tab in tabs_to_clear:
            if not marked_only:
                self._base_images[tab] = None

            self._marked_images[tab] = None
            self._disp_images[tab] = None

    def set_size(self, new_size):
        if new_size != self._size or new_size not in self._size_cache:
            logging.info(f"StateImageManager resized to {new_size}")
            self.clear_images()
            self._size = new_size
            if new_size in self._size_cache:
                self._box_placer, self._box_centers, self._box_tree = self._size_cache[new_size]
            else:
                self._box_placer, self._box_centers, self._box_tree = self._calc_dims()
                self._size_cache[new_size] = (self._box_placer, self._box_centers, self._box_tree)

    def _get_state_at(self, x, y):
        pos = np.array([x, y])
        closest_ind = self._box_tree.query(pos)[1]
        closest_state = self.all_states[closest_ind]
        layer = self.layers_per_state[closest_state]
        state_size = self._box_sizes[layer]
        closest_center = np.array(self._box_centers[closest_ind])
        dist = np.linalg.norm(pos - closest_center)
        if dist < state_size / 2:
            return closest_state
        return None

    def mouse_move(self, x, y, tab):
        """
        Update mouseovered state.
        If it changes, invalidate display images.
        """
        new_mo_state = self._get_state_at(x, y)
        if not (self.mouseovered is new_mo_state):
            self.mouseovered = new_mo_state
            self.clear_images(marked_only=True)
            return True
        return False

    def mouse_click(self, x, y, tab):
        new_click_state = self._get_state_at(x, y)
        if new_click_state is None:
            return False
        self._toggle_selected(new_click_state)
        return True

    def _toggle_selected(self, state):
        """
        Toggle the selected state.
        :param state:  The state to toggle.
        """
        if state in self.selected:
            self.selected.remove(state)
        else:
            self.selected.append(state)
        self.clear_images(marked_only=True)
        self._app.toggle_stop_state(state)  # internal change, need to tell the app

    def clear_selected(self):
        """
        externally cleared (not from user clicks)
        """
        n_selected = len(self.selected)
        self.selected = []
        if n_selected > 0:
            logging.info(f"Cleared {n_selected} selected states.")
            self.clear_images(marked_only=True)

    def _calc_dims(self):
        """
        Calculate state layout.
        """
        box_placer, _, _ = get_box_placer(self._size,
                                          self.all_states,
                                          box_sizes=self._box_sizes,
                                          layer_vpad_px=1,
                                          layer_bar_w=1,
                                          player=self._env.player)
        terminal_lut = {state: state.check_endstate() for state in self.all_states}
        tree_opt = SimpleTreeOptimizer(image_size=self._size,
                                       states_by_layer=self.states_by_layer,
                                       state_positions=box_placer.box_positions,
                                       terminal=terminal_lut)
        new_positions = tree_opt.get_new_positions()
        box_placer.box_positions = new_positions

        # Make a KD tree to see which box the mouse is nearest.
        box_centers = []
        for state in self.all_states:
            pos = box_placer.box_positions[state]
            box_centers.append(((pos['x'][0] + pos['x'][1]) / 2,
                                (pos['y'][0] + pos['y'][1]) / 2))
        box_centers = np.array(box_centers)
        box_tree = KDTree(box_centers)
        return box_placer, box_centers, box_tree

    def _get_state_icons(self):
        artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in SPACE_SIZES]
        images = {state['id']: artists[layer_num].get_image(
            state['id']) for layer_num, state_list in enumerate(self.states_by_layer) for state in state_list}
        box_sizes = [artists[layer_no].get_image(Game()).shape[0] for layer_no in range(len(artists))]
        return images, box_sizes

    def set_current_state(self, state=None):
        old_cur_state = self._cur_state
        self._cur_state = state

        if old_cur_state is None and state is None:
            return
        
        if (state is None and old_cur_state is not None) or\
            (state is not None and old_cur_state is None) or \
                (old_cur_state != state):
            self.clear_images(marked_only=True)

    def get_tab_img(self, tab, annotated=True):
        if annotated:
            if self._disp_images[tab] is None:
                self._disp_images[tab] = self.draw_annotated(tab)
            return self._disp_images[tab]
        else:
            if self._marked_images[tab] is None:
                self._marked_images[tab] = self.draw_marked(tab)
            return self._marked_images[tab]

    @abstractmethod
    def draw_base(self, tab):
        """
        Draw the base image for a tab.
        :param tab:  The tab to draw.
        :return:  The base image for the tab.
        """
        pass

    @abstractmethod
    def draw_marked(self, tab):
        """
        Draw the images with the selected/mouseovered states marked.
        :param tab:  The tab to draw.
        :return:  The annotated image for the tab.
        """
        pass

    @abstractmethod
    def draw_annotated(self, tab):
        """
        Draw the images with the selected/mouseovered states marked, and any additional annotations.
        :param tab:  The tab to draw.
        :return:  The annotated image for the tab.
        """
        pass


class ValueFunctionSIM(StateImageManager):
    """
    State Image Manager for representing value funcitons / updates.
    Assigns colors to values, linearly interpolating the range.
    Does not plot anything for unassigned states.

    All states are initially visible, values are updated, etc.
    """

    def __init__(self, app, env, tabs, value_ranges, colormap_names, bg_colors):
        """
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)

          NOTE:  If one of the tabs is named "states" the state icons will be drawn in that image.
                 All other tabs will be associated with a state:value mapping.

        :param value_ranges:  dict(tab_name: (min, max)) for each tab (defaults to (-1, 1))
        :param colormap_names:  dict(tab_name: colormap name) for each tab (defaults to 'viridis')
        :param bg_colors:  dict(tab_name: color) for each tab (defaults COLOR_BG for states, SKY_BLUE for values/updates)
        """
        super().__init__(app, env, tabs)
        self._cmaps = {tab: plt.get_cmap(colormap_names[tab]) for tab in tabs if tab != 'states'}
        self._ranges = value_ranges
        self._bg_colors = bg_colors
        self._values = {}
        self.reset_values()

    def set_range(self, tab, value_range):
        """
        Set the value range for a tab.
        :param tab:  The tab to set the range for.
        :param value_range:  The value range to set.
        """
        if tab not in self._ranges:
            raise ValueError(f"Tab {tab} not found in {self._ranges.keys()}")
        self._ranges[tab] = value_range
        self.clear_images(tabs=[tab])

    def reset_values(self, tabs=None):
        # logging.info(f"Value Function Image Manager  -  Resetting values for tabs: {tabs}")
        # print("Disp image keys before reset:", self._disp_images.keys())

        tabs_to_reset = self.tabs if tabs is None else tabs
        tabs_to_reset = (tabs_to_reset,) if isinstance(tabs_to_reset, str) else tabs_to_reset

        for tab in tabs_to_reset:
            if tab == 'states':
                continue
            self._values[tab] = {}

        self.clear_images(tabs=tabs, marked_only=False)

        # print("Disp image keys after reset:", self._disp_images.keys())

    def get_color(self, tab, value):
        """
        Get the color for a value in a tab.
        :param tab:  The tab to get the color for.
        :param value:  The value to get the color for.
        :return:  The color for the value in the tab.
        """
        if value is None:
            return None
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
        Update the base image if it exists.

        :param state:  The state to set the value for.
        :param tab:  The tab to set the value for.
        :param value:  The value to set.
        """
        if tab == 'states':
            raise ValueError("Cannot set values for 'states' tab, use 'values' or 'updates' instead.")
        self._values[tab][state] = value
        if self._base_images[tab] is not None and self._box_placer is not None:
            color = self.get_color(tab, value)
            self._box_placer.draw_box(image=self._base_images[tab],
                                      state_id=state,
                                      color=color,
                                      thickness=0)  # filled box
            # invalidate display image since the base image has changed
            self._disp_images[tab] = None
            self._marked_images[tab] = None

    def draw_base(self, tab):
        logging.info(f"Regenerating base image for {tab}")
        img = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        img[:] = self._bg_colors[tab]

        if tab == 'states':
            img = self._box_placer.draw(images=self._state_icons, colors=None, show_bars=True, dest=img)
        else:
            state_colors = {state: self.get_color(tab, self._values[tab].get(state, None)) for state in self.all_states}
            img = self._box_placer.draw(images=None, colors=state_colors, show_bars=False, dest=img)
        return img

    def draw_marked(self, tab):
        if self._base_images[tab] is None:
            self._base_images[tab] = self.draw_base(tab)
        img = self._base_images[tab].copy()

        for state in self.selected:
            self._box_placer.draw_box(image=img,
                                      state_id=state,
                                      color=NEON_RED,
                                      thickness=1)

        if self.mouseovered is not None:
            self._box_placer.draw_box(image=img,
                                      state_id=self.mouseovered,
                                      color=NEON_GREEN,
                                      thickness=1)

        if self._cur_state is not None:
            self._box_placer.draw_box(image=img,
                                      state_id=self._cur_state,
                                      color=NEON_GREEN,
                                      thickness=2)
        return img

    def draw_annotated(self, tab):
        # Stub, override for subclass
        if self._marked_images[tab] is None:
            self._marked_images[tab] = self.draw_marked(tab)
        img = self._marked_images[tab].copy()
        return img


class PolicyEvalSIM(ValueFunctionSIM):
    def __init__(self, app, env):

        tabs = ['states', 'values', 'updates']
        colormap_names = {'values': 'gray', 'updates': 'coolwarm'}
        value_ranges = {'values': (-1, 1), 'updates': (-1, 1)}
        bg_colors = {'states': COLOR_BG, 'values': SKY_BLUE, 'updates': SKY_BLUE}
        super().__init__(app, env, tabs, value_ranges, colormap_names, bg_colors)

    def get_state_update_order(self):
        if self._box_placer is None:
            logging.info("Warning: Box placer not initialized, cannot get update order.")
            return self.updatable_states
        update_order = sorted(self.updatable_states, key=lambda s: self._box_placer.box_positions[s]['x'][0])
        return update_order


class SimTester(object):
    """
    Simple TK app with 3 tabs, 1 frame in each (the state image).
    """

    def __init__(self, size, env):
        # args for Policy Eval tabs:
        self._start_time = time.perf_counter()
        self.current_tab_ind = 0
        self._init_tk(size)
        self._init_values(env)
        self._init_tabs()

    def toggle_selected(self, state):
        print(f"Toggling selected state:\n{state}\n")

    def _init_values(self, env):

        self._sim = PolicyEvalSIM(self, env)
        self._tabs = self._sim.tabs
        vals = []

        top_states = [state for state in self._sim.updatable_states if state.n_free() > 7]

        for state in self._sim.terminal_states:
            value = TERMINAL_REWARDS[state.check_endstate()]
            self._sim.set_state_val(state, 'values', value)
            vals.append(value)

        for state in self._sim.updatable_states:
            self._sim.set_state_val(state, 'updates', 0.0)
            self._sim.set_state_val(state, 'values', 0.0)

        for i, state in enumerate(top_states):
            self._sim.set_state_val(state, 'updates', (i-4.5)/4.5)
            self._sim.set_state_val(state, 'values', (i-4.5)/4.5)

    def _init_tk(self, size):
        self._size = size
        self._root = tk.Tk()
        self._root.title("State Image Manager")
        self._root.geometry(f"{size[0]}x{size[1]}")
        self._root.resizable(True, True)
        self._root.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        # check if is a resize event
        new_size = (event.width, event.height)
        if new_size == self._size:
            return
        logging.info(f"Window resized to {new_size}")
        self._sim.set_size(new_size)
        self._size = new_size
        self.refresh_image()

    def _mouse_click(self, event, tab):
        if self._sim.mouse_click(event.x, event.y, tab):
            self.refresh_image()

    def _mouse_move(self, event, tab):
        if self._sim.mouse_move(event.x, event.y, tab):
            self.refresh_image()

    def _init_tabs(self):
        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._tab_images = {}
        for name in self._tabs:
            tab_frame = tk.Frame(self._notebook)
            tab_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            img_label = tk.Label(tab_frame)
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            img_label.bind("<Button-1>", lambda event: self._mouse_click(event, name))
            img_label.bind("<Motion>", lambda event: self._mouse_move(event, name))

            self._notebook.add(tab_frame, text=name)
            self._tab_images[name] = (tab_frame, img_label)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        current_tab = self._tabs[self.current_tab_ind]
        self._notebook.select(self._tab_images[current_tab][0])

    def _on_tab_changed(self, event):
        new_tab = self._notebook.tab(self._notebook.select(), "text")
        new_tab_ind = self._tabs.index(new_tab)
        if new_tab_ind != self.current_tab_ind:
            self.current_tab_ind = new_tab_ind
            logging.info(f"Tab changed to {new_tab}")
            self.refresh_image()

    def refresh_image(self):

        current_tab = self._tabs[self.current_tab_ind]
        self._sim.set_size(self._size)
        img = self._sim.get_tab_img(current_tab, annotated=True)
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
