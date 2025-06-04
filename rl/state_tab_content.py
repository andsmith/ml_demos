import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_SCHEME
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
from color_key import ColorKey, ProbabilityColorKey
from state_key import StateKey
from gameplay import ResultSet, Match
from tab_content import TabContentPage
from state_embedding import StateEmbedding


class FullStateContentPage(TabContentPage):
    """
    FullStateContentPage is a base class for content tabs that display the full state embedding.
    """
    def __init__(self, alg, embedding,keys,bg_color=None):
        """
        :param app:  The application instance, used to notify about state changes.
        :param env:  The environment to use.
        :param bg_color:  The background color to use for the image. Defaults to COLOR_BG.
        :param embedding:  The StateEmbedding instance to use for drawing states.
        """
        super().__init__(alg, keys)
        self._app = alg.app
        self._env = alg.app.env
        all_states = self._env.get_terminal_states() + self._env.get_nonterminal_states()
        self._displayed_states = {state: None for state in all_states}
        self._bg_color = bg_color if bg_color is not None else COLOR_SCHEME['bg']
        self._embed = embedding
        self._layout = LAYOUT['state_embedding']
        self._state_icons = self._get_state_icons()
        self._base_image = None  # Base image for the content page, used for mouseover and marking.
        self._marked_image = None
        self._cur_state = None  # The state the algorithm is currently on, if set.
        self._cur_box = None  # The box for the current state, if set.

    def _get_state_icons(self):
        images = {state['id']: self._embed.artists[layer_num].get_image(
            state['id']) for layer_num, state_list in enumerate(self._embed.states_by_layer) for state in state_list}
        return images
    
    def resize(self, new_size):
        # embedding should already be changed, just send its boxes to the mouse manager.
        super().resize(new_size)
        box_placer = self._embed.box_placer
        if box_placer is not None:
            self._mouse_manager.set_boxes(box_placer.box_positions)


    def _draw_base(self):
        size = self._embed.size
        logging.info(f"Regenerating base image.")
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = self._bg_color
        img = self._embed.box_placer.draw(images=self._state_icons, colors=None, show_bars=False, dest=img)
        return img

    def set_current_state(self, state=None):
        old_cur_state = self._cur_state
        self._cur_state = state

        if old_cur_state is None and state is None:
            return

        if (state is None and old_cur_state is not None) or\
            (state is not None and old_cur_state is None) or \
                (old_cur_state != state):
            self.clear_images(marked_only=True)

    def _get_key_value(self, key_name):
        """
        Subclasses with additional/different keys should override this.
        """
        if key_name=='state':
            return self._mouse_manager.mouseover_id
        elif key_name=='embedding':
            pass # static key
        else:
            raise ValueError(f"Unknown key name: {key_name}. Only 'state' is supported in FullStateContentPage.")
        

class ValueFunctionContentPage(FullStateContentPage):
    def __init__(self, alg, embedding, keys, bg_color=None):
        """
        :param app:  The application instance.
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)
        :param value_ranges:  dict(tab_name: (min, max)) for each tab (defaults to (-1, 1))
        :param colormap_names:  dict(tab_name: colormap name) for each tab (defaults to 'viridis')
        :param bg_colors:  dict(tab_name: color) for each tab (defaults COLOR_BG for states, SKY_BLUE for values/updates)
        :param key_sizes:  dict with 'color' and 'state' keys, each with a tuple (width, height)
        """
        bg_color = bg_color if bg_color is not None else COLOR_SCHEME['func_bg']
        super().__init__(alg=alg, embedding=embedding, keys=keys, bg_color=bg_color)
        self._values = {}
        self._cmap = plt.get_cmap('viridis')  # Default colormap for value functions

    def set_values(self, values):   
        self._values = values
        logging.info(f"Setting values for {len(values)} states.")
        self.clear_images(marked_only=False)

    def _get_key_value(self, key_name):
        if key_name=='values':
            state= self._mouse_manager.mouseover_id
            if state is None or state not in self._values:
                return None
            return self._values[state]
        elif key_name=='state':
            return super()._get_key_value(key_name)
        else:
            raise ValueError(f"Unknown key name: {key_name}. Only 'state' is supported in FullStateContentPage.")


'''
class ValueFunctionContentPage(FullStateContentPage):
    """
    State Image Manager for representing value funcitons / updates.
    Assigns colors to values, linearly interpolating the range.
    Does not plot anything for unassigned states.

    All states are initially visible, values are updated, etc.
    """

    def __init__(self, app, env, alg):
        """
        :param env:  The environment to use.
        :param tabs:  list of strings, names of tabs (for dict references, not display)

          NOTE:  If one of the tabs is named "states" the state icons will be drawn in that image.
                 All other tabs will be associated with a state:value mapping.

        :param value_ranges:  dict(tab_name: (min, max)) for each tab (defaults to (-1, 1))
        :param colormap_names:  dict(tab_name: colormap name) for each tab (defaults to 'viridis')
        :param bg_colors:  dict(tab_name: color) for each tab (defaults COLOR_BG for states, SKY_BLUE for values/updates)
        :param key_sizes:  dict with 'color' and 'state' keys, each with a tuple (width, height)

        """

        super().__init__(app, env)
        # colormap for value function representation:
        self._cmap = {tab: plt.get_cmap(colormap_names[tab]) for tab in tabs if tab != 'states'}
        ck_width, ck_height = key_sizes['color']
        self._ranges = value_ranges
        self._bg_colors = bg_colors

        # create the color keys
        self._color_keys = {tab: ColorKey(size=(ck_width, ck_height),
                                          cmap=self._cmaps[tab],
                                          range=self._ranges[tab])
                            for tab in self.tabs if tab not in ['results', 'states']}
        self._color_keys['results'] = ProbabilityColorKey(size=(ck_width, ck_height))

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
            img = self._box_placer.draw(images=self._state_icons, colors=None, show_bars=False, dest=img)
        else:
            state_colors = {state: self.get_color(tab, self._values[tab].get(state, None)) for state in self.all_states}
            img = self._box_placer.draw(images=None, colors=state_colors, show_bars=False, dest=img)
            # Add a vertical line between the state key and the color key:
            # v_line_x = self._size[0] - self._key_sizes['color'][0]
            # v_line_bottom = self._key_sizes['color'][1]
            # img[0:v_line_bottom, v_line_x:v_line_x + 1] = COLOR_LINES

        return img

    def draw_marked(self, tab):

        # Mark on the base image, regenerate if needed.
        if self._base_images[tab] is None:
            self._base_images[tab] = self.draw_base(tab)
        img = self._base_images[tab].copy()

        # Indicate selected states with a box.
        for state in self.selected:
            self._box_placer.draw_box(image=img,
                                      state_id=state,
                                      color=NEON_RED,
                                      thickness=1)
        indicated_value = None

        # Indicate mouseovered state with a box.
        if self.mouseovered is not None:
            self._box_placer.draw_box(image=img,
                                      state_id=self.mouseovered,
                                      color=NEON_GREEN,
                                      thickness=1)
            if tab in self._color_keys:
                indicated_value = self._values[tab].get(self.mouseovered, None)

        # Draw the color key if applicable.
        if tab in self._color_keys:
            self._color_keys[tab].draw(img, line_color=COLOR_LINES, text_color=COLOR_TEXT,
                                       indicate_value=indicated_value)

        # Draw the state key if not the state tab
        if self.mouseovered is not None:
            self._state_key.draw(img, self.mouseovered, pos=self._state_key_pos)

        # Draw the state the algorithm is currently on, if set.
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


class PolicyEvalSIM(FullStateContentPage):
    def __init__(self, app, env):

        tabs = ['states', 'values', 'updates']
        colormap_names = {'values': 'gray', 'updates': 'coolwarm'}
        value_ranges = {'values': (-1, 1), 'updates': (-1, 1)}
        bg_colors = {'states': COLOR_BG, 'values': SKY_BLUE, 'updates': SKY_BLUE}

        key_sizes = {'color': (LAYOUT['color_key']['width'], LAYOUT['color_key']['height']),
                     'state': (LAYOUT['state_key']['width'], LAYOUT['state_key']['height'])}

        # Initialize the base class with the app, env, tabs, value_ranges, colormap_names, bg_colors, and key_sizes
        super().__init__(app, env, tabs, value_ranges, colormap_names, bg_colors, key_sizes)

    def get_state_update_order(self):
        if self.box_placer is None:
            logging.info("Warning: Box placer not initialized, cannot get update order.")
            return self.updatable_states
        update_order = sorted(self.updatable_states, key=lambda s: self.box_placer.box_positions[s]['x'][0])
        return update_order


class SimTester(object):
    """
    Simple TK app with 3 tabs, 1 frame in each (the state image).
    """

    def __init__(self, size, env,player_policy):
        # args for Policy Eval tabs:
        self._start_time = time.perf_counter()
        self.current_tab_ind = 0
        self._sim = PolicyEvalSIM(self, env)

        self._init_tk(size)
        self._init_values(env)
        self._init_tabs()
        self._player_mark = env.player
        self._player_policy=player_policy
        self._opp_policy = env.pi_opp
        self._results = ResultSet(self._player_mark)
        self._play(env)

    def _play(self,env):
        for _ in range(100):
            match = Match(self._player_policy, self._opp_policy)
            self._sim.add_result_trace(match.play_and_trace(order=0, verbose=False))
        logging.info(f"Played %i matches." % (self._results.get_summary()['games']))


    def toggle_stop_state(self, state):
        logging.info(f"Toggling selected state:\n{state}\n")

    def _init_values(self, env):

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
    player_policy = HeuristicPlayer(mark=AGENT_MARK, n_rules=1)
    env = Environment(opp_policy, AGENT_MARK)

    window = SimTester(init_size, env, player_policy=player_policy)
    window.start()


'''

def test():
    p = FullStateContentPage(alg=None, embedding=StateEmbedding(Game, size=(500, 500)), keys=StateKey())


if __name__ == "__main__":
    test()
