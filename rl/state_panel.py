"""
Visualize the value function of a policy as it's updated.

Should be a tabbed interface with different views of the reinforcement state.

"""
from gui_base import Panel
import tkinter as tk
from colors import COLOR_BG, COLOR_TEXT, COLOR_DRAW, COLOR_LINES
from layout import LAYOUT, WIN_SIZE
from game_base import Mark, Result

class StatePanel(Panel):
    """
    Panel for displaying the current state of the game.
    """
    def __init__(self, app, bbox_rel):
        super().__init__(app, bbox_rel)
        self._state = None  # The current state of the game.
        self._state_artists = []  # The artists for the current state.
        self._state_labels = []  # The labels for the current state.
        self._state_mark = Mark.X  # The mark of the player for the current state.