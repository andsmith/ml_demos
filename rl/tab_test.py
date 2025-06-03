import tkinter as tk
from tkinter import ttk
from state_tab_content import FullStateContentPage
from policy_eval import PolicyEvalDemoAlg
from colors import COLOR_SCHEME
import numpy as np
import logging
from PIL import Image, ImageTk
from util import tk_color_from_rgb
from baseline_players import HeuristicPlayer
from reinforcement_base import Environment
from tic_tac_toe import Mark,Game
class FakeApp(object):
    """
    stand-in for rl_demo.RLDemoApp

    """
    def __init__(self):
        self.player = Mark.X
        
        self.opp_policy = HeuristicPlayer(mark=Mark.O, n_rules=2)
        self.env = Environment(self.opp_policy,self.player)
        terminals, nonterminals = self.env.get_terminal_states(), self.env.get_nonterminal_states()
        self.all_states = {state:None for state in terminals + nonterminals}
        self.selected = []

    def toggle_selected_state(self, state_id):
        if state_id not in self.all_states:
            raise ValueError(f"State {state_id} not found in all_states.")
        if state_id in self.selected:
            self.selected.remove(state_id)
        else:
            self.selected.append(state_id)
        print(f"Toggled selection for state {state_id}, now selected: {state_id in self.selected}, Total selected: {len(self.selected)}")   


class TabTester(object):
    """

    stand-in for alg_panels.TabPanel

    Create "tab 1" and "tab 2"
    Tab 1 has a grid of boxes with random colors.
    Tab 2 has a random subset of boxes, blown-up.
    Selecting a box in either tab will toggle its selection state, i.e. there is one set of selected boxes for both tabs.
    Mouseovering a box will highlight it in green.

    """

    def __init__(self, app, alg, img_size):
        self._img_size = img_size
        self._app = app
        self._alg = alg
        self._alg.state = Game.from_strs(["   ", " O ", "   "])  # Set a default state for the algorithm
        self._tab_img_size = None
        self._init_tk()
        self._change_algorithm(alg) 

    def _change_algorithm(self, alg):        
        tab_content = alg.get_tabs()
        if tab_content is not None:
            self.set_tabs(tab_content)

    def _init_tk(self):
        """
        Init Tkinter and create the main window with two tabs.
        """
        self._root = tk.Tk()
        self._root.title("Mouse State Manager Tester")
        self._root.geometry(f"{self._img_size[0]}x{self._img_size[1]}")

        self._notebook = tk.ttk.Notebook(self._root)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        self._notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def _set_cur_tab(self):
        tab_text = self._notebook.tab(self._notebook.select(), "text")
        self._cur_tab = self._tab_names_from_texts[tab_text]
        return self._cur_tab


    def set_tabs(self, tab_content):

        self._tabs =tab_content

        self._frames = {}
        self._labels = {}
        self._tab_names_from_texts = {}

        for tab_name in tab_content:
            tab_text = tab_content[tab_name]['disp_text']
            frame = tk.Frame(self._notebook, bg=tk_color_from_rgb(COLOR_SCHEME['bg']))
            self._notebook.add(frame, text=tab_text)
            self._frames[tab_name] = frame
            self._tab_names_from_texts[tab_text] = tab_name

            img = Image.new('RGB', self._img_size, color=COLOR_SCHEME['bg'])
            label = tk.Label(frame, image=ImageTk.PhotoImage(img))
            label.pack(fill=tk.BOTH, expand=True)
            self._labels[tab_name] = label
            
            label.bind("<Motion>", self.on_mouse_move)
            label.bind("<Button-1>", self.on_mouse_click)
            label.bind("<Leave>", self.on_mouse_leave)
            label.bind("<Configure>", self.on_tab_resize)  # for all tabs

        self._cur_tab =  list(self._tabs.keys())[0]  # Start with the first tab
        self._notebook.select(self._frames[self._cur_tab])

    def on_resize(self, event):
        """
        Handle the resize event for the main window.
        """
        pass

    def on_tab_resize(self, event):
        new_tab_size = (event.width, event.height)
        if self._tab_img_size is None or   (self._tab_img_size != new_tab_size):
            logging.info("Resizing tab images to %s" % str(new_tab_size))
            self._tab_img_size = new_tab_size
            self._alg.resize('state-tabs', new_tab_size)
            self.refresh_images(clear=True)
            
    def on_tab_change(self, event):
        self._set_cur_tab()
        logging.info("Tab changed to: %s" % self._cur_tab)
        self.refresh_images()

    def on_mouse_leave(self, event):
        """
        Handle mouse leave events for the current tab.
        """
        tab = self._tabs[self._set_cur_tab()]['tab_content']
        if tab.mouse_leave():
            self.refresh_images()

    def on_mouse_move(self, event):
        tab = self._tabs[self._set_cur_tab()]['tab_content']
        if tab.mouse_move((event.x, event.y)):
            self.refresh_images()

    def on_mouse_click(self, event):
        tab = self._tabs[self._set_cur_tab()]['tab_content']
        logging.info("Mouse clicked at (%d, %d) in tab: %s" % (event.x, event.y, self._cur_tab))
        box_id = tab.mouse_click((event.x, event.y))
        if box_id is not None:
            self.refresh_images()

    def _render_frame(self, tab_name):
        tab = self._tabs[self._set_cur_tab()]['tab_content']
        frame=tab.get_tab_frame(self._tab_img_size, annotated=True)

        return frame

    def refresh_images(self,clear=False):
        if self._img_size is None:
            return
        tab = self._tabs[self._set_cur_tab()]['tab_content']
        if clear:
            tab.clear_images(marked_only=True)
        new_img = self._render_frame(self._cur_tab)
        new_img = ImageTk.PhotoImage(image=Image.fromarray(new_img))
        label = self._labels[self._cur_tab]
        label.config(image=new_img)
        label.image = new_img

    def start(self):
        """
        Start the Tkinter main loop.
        """
        self._root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    img_size = (1200, 970)
    box_size = 20
    app = FakeApp()  # Stand-in for the RLDemoApp
    pi_seed = HeuristicPlayer(mark=Mark.X, n_rules=1)

    alg = PolicyEvalDemoAlg(app,pi_seed)
    
    tester = TabTester(app,alg, img_size)
    tester.start()
    logging.info("Mouse State Manager Tester finshed.")
