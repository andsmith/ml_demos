from colors import COLORS
import cv2


LAYOUT = {"windows": {'ui': {'x': (0, .666),  # scale from unit square to window size
                             'y': (0, .666)},
                      'tools': {'x': (0, .666),
                                'y': (.667, 1)},
                      'eigenvectors': {'x': (.667, 1),
                                       'y': (.333, .667)},
                      'clusters': {'x': (.667, 1),
                                   'y': (.667, 1)},
                      'spectrum': {'x': (.667, 1),
                                   'y': (0, .333)}},
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

TOOLS = ['kind_radio',
         'alg_radio',
         'n_nearest_slider',
         'n_pts_slider',
         'run_button',
         'clear_button',]

"""
Toolbar layout roughly:
|----------------------------------------|
| kind:     Algorithm:       K:          |
| 1gauss     1unnormalized   |------+--| |
| 2ellipse   2normalized                 |
| 3annulus   3kmeans         N-nearest:  |
| 4????                      |---+-----| |
|            n_pts                      |
| run clear  |---+--------------------|  |
|---------------------------------------|
"""
# coords are in unit square, will be scaled to toolbar area of window (as defined in LAYOUT['windows']['tools'])
TOOLBAR_LAYOUT = {'kind_radio': {'x': (0, .33), 'y': (0, .67)},
                  'alg_radio': {'x': (.33, .67), 'y': (0, .67)},
                  'k_slider': {'x': (.67, 1), 'y': (0, .33)},
                  'n_nearest_slider': {'x': (.67, 1), 'y': (.33, .67)},
                  'n_pts_slider': {'x': (.33, 1), 'y': (.67, 1)},
                  'run_button': {'x': (0, .33/2), 'y': (.67, 1)},
                  'clear_button': {'x': (.33/2, .33), 'y': (.67, 1)}}
