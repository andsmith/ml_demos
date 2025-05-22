import cv2

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageDraw
from colors import COLOR_BG, COLOR_TEXT, COLOR_DRAW, COLOR_LINES
from layout import LAYOUT, WIN_SIZE
from game_base import Mark, Result
from game_util import get_box_placer, get_state_icons, sort_states_into_layers
from layer_optimizer import SimpleTreeOptimizer
import numpy as np
from tic_tac_toe import Game
import time
import logging
from abc import ABC, abstractmethod
from drawing import GameStateArtist
#BOX_SIZES = [22, 12, 12, 12, 12, 12]  # good for single value function
SPACE_SIZES = [7, 2,2,2,2,3]

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

    def set_size(self, new_size):
        logging.info(f"StateImageManager resized to {new_size}")
        self._size = new_size
        self._calc_dims()

    @abstractmethod
    def _calc_dims(self):
        """
        Recalculate placement of boxes, etc.
        """
        pass

    @abstractmethod
    def draw(self):
        pass


class ModelBasedSIM(StateImageManager):
    """
    State Image Manager for model-based reinforcement learning algorithms.
    All states are initially visible, values are updated, etc.
    """

    def __init__(self, env, tabs):
        super().__init__(env, tabs)

        self._state_icons, self._box_sizes = self._get_state_icons()
        # self._box_placer = get_box_placer(self._size[0], self._size[1], 3, 3)

    def _calc_dims(self):
        """
        Find the bounding box of each state
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

        #space_sizes = [GameStateArtist.get_space_size(box_size, bar_w_frac=0.0) for box_size in BOX_SIZES]
        artists = [GameStateArtist(space_size=s, bar_w_frac=0.0) for s in SPACE_SIZES]
        images = {state['id']: artists[layer_num].get_image(state['id']) for layer_num, state_list in enumerate(self.states_by_layer) for state in state_list}
        # get the size of images in each layer:
        test_icons = []
        box_sizes = [artists[layer_no].get_image(Game()).shape[0] for layer_no in range(len(artists))]
        print("Space sizes:", SPACE_SIZES)
        print("Box sizes:", box_sizes)
        return images, box_sizes

    def draw(self, tab):
        img = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        img[:] = COLOR_BG

        if tab == 'states':
            #    def draw(self, images=None, colors=None, dest=None, show_bars=False):
            #import ipdb; ipdb.set_trace()

            img=self._box_placer.draw( images=self._state_icons, colors=None, show_bars=True, dest=img)

        return img


class SimTester(object):
    """
    Simple TK app with 3 tabs, 1 frame in each (the state image).
    """

    def __init__(self, size, env):
        self._tabs = ['states', 'values', 'updates']
        self.current_tab_ind = 0
        self._init_tk(size)
        self._init_tabs()
        self._init_images(env)

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
        self._notebook.bind("<Button-1>", self._on_tab_changed)

    def _init_images(self, env):
        self._sim = ModelBasedSIM(env, self._tabs)

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
        img = self._sim.draw(current_tab)
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
