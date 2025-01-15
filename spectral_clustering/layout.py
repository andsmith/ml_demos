from colors import COLORS
import cv2
import numpy as np
from enum import IntEnum
from util import hsplit_bbox
"""
Windows are laid out rougly:
+---------------------+----------+----------+
|  "UI window"        |          |          |
|  (click to add and  | spectrum |e-vectors |
|   modify clusters)  |          |          |
|                     +----------+----------|
+------------+--------+          |          |
|  tollbar   |  sim   | randproj | clusters |
|            | matrix |          |          |
+------------+--------+----------+----------+
"""


class Windows(IntEnum):
    """
    Enum for the windows in the layout.
    """
    ui = 0
    spectrum = 1
    eigenvectors = 2
    sim_matrix = 3
    rand_proj = 4
    clustering = 5
    toolbar = 6


h_div = 0.4  # relative x position of the  vertical dividing line between UI and spectrum
h_mid = (h_div+1.)/2.  # x position between spectrum and eigenvectors

WINDOW_LAYOUT = {"windows": {Windows.ui: {'x': (0, h_div),  # scale from unit square to window size
                                          'y': (0, .75)},
                             Windows.toolbar: {'x': (0, .45),
                                               'y': (.75, 1)},
                             Windows.spectrum: {'x': (h_div, h_mid),  # move a bit in for a slider
                                                'y': (0, .5)},
                             Windows.eigenvectors: {'x': (h_mid, 1),
                                                    'y': (0, .5)},
                             Windows.sim_matrix: {'x': (.45, h_div),
                                                  'y': (.75, 1)},  # will be made square regardless of window size
                             Windows.rand_proj: {'x': (h_div, h_mid),
                                                 'y': (.5, 1)},
                             Windows.clustering: {'x': (h_mid, 1),
                                                  'y': (.5, 1)}},
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


OTHER_TOOL_LAYOUT = {'spectrum_slider_w_frac': .14,  # portion of the window for the slider
                     }
"""
Toolbar layout roughly 3 columns:
|-------------------------------------------|
| C-Kind:     Sim_graph:    Algorithm:      |
|   1gauss      1epsilon      1unnormalized |
|   2ellipse    2K-nn         2normalized   |
|   3anulus     3full         3k-means      |
|   4moons                                  |
|             [sim-param]   K=5 (N clust)   | 
              |---+-----|   |---+---------| |
|                           N points = 300  |       
| |run|  |clear|            |-------+-----| |
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


button_indent = 0.02
# coords are in unit square, will be scaled to toolbar area of window (as defined in LAYOUT['windows']['tools'])
TOOLBAR_LAYOUT = {Tools.kind_radio: {'x': (0, .33),  # scale from unit square to window size
                                     'y': (0, .66)},
                  Tools.sim_graph_radio: {'x': (.33, .66),
                                          'y': (0, .66)},
                  Tools.alg_radio: {'x': (.66, 1),
                                    'y': (0, .66)},

                  Tools.k_slider: {'x': (.5, 1),
                                   'y': (.6, .80)},
                  Tools.n_pts_slider: {'x': (.5, 1),
                                       'y': (.8, 1)},

                  Tools.epsilon_slider:  None,  # fill these in below
                  Tools.sigma_slider: None,
                  Tools.nn_slider: None,
                  Tools.nn_toggle: None,

                  Tools.run_button: {'x': (button_indent, button_indent+.16),
                                     'y': (.83, 1)},
                  Tools.clear_button: {'x': (button_indent+.24-.02, button_indent+.40+.02),
                                       'y': (.83, 1)},
                  }


sim_param_area = {'x': (button_indent, .45),
                  'y': (.6, .8)}
left, right = hsplit_bbox(sim_param_area, [2.5,1.4], integer=False)

# three_boxes = vsplit_bbox(sim_param_area, [1.5, .5, 1, 1])
TOOLBAR_LAYOUT[Tools.epsilon_slider] = sim_param_area  # three_boxes[0]
TOOLBAR_LAYOUT[Tools.sigma_slider] = sim_param_area  # three_boxes[0]
TOOLBAR_LAYOUT[Tools.nn_slider] = left  # three_boxes[0]
TOOLBAR_LAYOUT[Tools.nn_toggle] = right  # three_boxes[2]


PLOT_LAYOUT = {'axis_spacing': 5,
               'font': cv2.FONT_HERSHEY_SIMPLEX,
               'title_color': COLORS['gray'],
               'axis_color': COLORS['black'],
               'tick_color': COLORS['gray']
               }
