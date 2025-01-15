from colors import COLORS
import cv2
import numpy as np
from enum import IntEnum
from util import vsplit_bbox
"""
Windows are laid out rougly:
|---------------------------------------------------|
|  Adding clusters ("UI")         |  spectrum       |
|  (click to add points)          |-----------------|
|                                 |  eigenvectors   |
|---------------------------------|-----------------|
|  sim_matrix  | graph_stats      |  clustering     |
|---------------------------------------------------|
| toolbar                                           |
|---------------------------------------------------|
"""


class Windows(IntEnum):
    """
    Enum for the windows in the layout.
    """
    ui = 0
    spectrum = 1
    eigenvectors = 2
    sim_matrix = 3
    graph_stats = 4
    clustering = 5
    toolbar = 6


WINDOW_LAYOUT = {"windows": {Windows.ui: {'x': (0, .667),  # scale from unit square to window size
                                          'y': (0, .5)},
                             Windows.toolbar: {'x': (0, .667),
                                               'y': (.75, 1)},
                             Windows.spectrum: {'x': (.667, 1),
                                                'y': (0, .333)},
                             Windows.eigenvectors: {'x': (.667, 1),
                                                    'y': (.333, .667)},
                             Windows.sim_matrix: {'x': (0, .25),
                                                  'y': (.5, .75)},  # will be made square regardless of window size
                             Windows.graph_stats: {'x': (.25, .666),
                                                   'y': (.5, .75)},
                             Windows.clustering: {'x': (.667, 1),
                                                  'y': (.667, 1)}},
                 'colors': {'bkg': COLORS['white'],
                            'border': COLORS['gray'],
                            'active_border': COLORS['black'],
                            'font': COLORS['black']},

                 'dims': {'margin_px': 5,
                          'pt_size': 2,
                          'mouseover_rad_px': 20},
                 'font': cv2.FONT_HERSHEY_SIMPLEX,
                 'font_size': .5,
                 'font_color': (0, 0, 0),
                 'font_thickness': 1}


"""
Toolbar layout roughly 4 columns:
|-----------------------------------------------------------|
| C-Kind:     Sim_graph:     [sim-params]:  Algorithm:      |
|   1gauss      1epsilon     |---+-----|      1unnormalized |
|   2ellipse    2K-nn                         2normalized   |
|   3anulus     3full                         3k-means      |
|   4moons                                                  |
|             K=5 (N clust)  N_pts = 100                    |
||run|clear|  |---+-----|    |-------+--------------------| |
|-----------------------------------------------------------|


Where [sim-par] is a custom toolbar bbox area for the similarity parameter(s).  See SIM_PARAM below.
"""


class Tools(IntEnum):
    """
    Enum for the tools in the toolbar.
    """
    kind_radio = 0
    sim_graph_radio = 1
    k_slider = 2
    n_pts_slider = 3
    run_button = 4
    clear_button = 5
    alg_radio = 6

    nn_slider = 7
    nn_toggle = 10
    epsilon_slider = 8
    sigma_slider = 9


# coords are in unit square, will be scaled to toolbar area of window (as defined in LAYOUT['windows']['tools'])
TOOLBAR_LAYOUT = {Tools.kind_radio: {'x': (0, .25),  # scale from unit square to window size
                                     'y': (0, .75)},
                  Tools.sim_graph_radio: {'x': (.25, .5),
                                          'y': (0, .75)},
                  Tools.k_slider: {'x': (.25, .5),
                                   'y': (.75, 1)},
                  Tools.n_pts_slider: {'x': (.5, 1),
                                       'y': (.75, 1)},

                  Tools.epsilon_slider:  None,  # fill these in below
                  Tools.sigma_slider: None,
                  Tools.nn_slider: None,
                  Tools.nn_toggle: None,
                  Tools.run_button: {'x': (0, .125),
                                     'y': (.75, 1)},
                  Tools.clear_button: {'x': (.125, .25),
                                       'y': (.75, 1)},

                  Tools.alg_radio: {'x': (.75, 1),
                                    'y': (0, .75)}, }

sim_param_area = {'x': (.5, .70),
                  'y': (0, .75)}

three_boxes = vsplit_bbox(sim_param_area, [1.5,.5,1,1])
TOOLBAR_LAYOUT[Tools.epsilon_slider] = three_boxes[0]
TOOLBAR_LAYOUT[Tools.sigma_slider] = three_boxes[0]
TOOLBAR_LAYOUT[Tools.nn_slider] = three_boxes[0]
TOOLBAR_LAYOUT[Tools.nn_toggle] = three_boxes[2]


PLOT_LAYOUT = {'axis_spacing': 5,
               'font': cv2.FONT_HERSHEY_SIMPLEX,
               'title_color': COLORS['gray'],
               'axis_color': COLORS['black'],
               'tick_color':COLORS['gray']
               }