

from colors import COLOR_SCHEME
import cv2
MID_X = 0.5  # "state-tabs" frame (sate/value/update images) to the right of this point
MID_Y = 0.5  # "step-visualization" frame below this point
LOW_Y = 1.0
HALF_MID_X = 0.15

SHIFT_BITS = 6
SHIFT_MUL = 2 ** SHIFT_BITS  # shift to the left by this many bits to get the value
LAYOUT = {'frames': {'control': {'x_rel': (0.0, HALF_MID_X),
                                 'y_rel': (0.0, MID_Y)},

                     'selection': {'x_rel': (0.0, HALF_MID_X),
                                   'y_rel': (MID_Y, LOW_Y)},
                     'step-visualization': {'x_rel': (HALF_MID_X, MID_X),
                                            'y_rel': (0.0, 1.0)},
                     'state-tabs': {'x_rel': (MID_X, 1.0),
                                    'y_rel': (0.0, 1.0)}
                     },
          'state_embedding': {  # a kind of TabContentPage showing all RL states
                'space_sizes': [12, 2, 2, 2, 2, 3]
          },  # sizes of the spaces between states in pixels
          'margin_rel': .0025,  # margin between frames in relative coordinates
        'img_margin_rel': .01,  # for tab and viz images
          'fonts': {'panel_title': ('Helvetica', 16, 'underline'),
                    'title': ('Helvetica', 16, 'underline'),
                    'default': ('Helvetica', 12),
                    'status': ('Helvetica', 13),
                    'status_bold': ('Helvetica', 13, 'bold'),
                    'menu': ('Helvetica', 12, ),
                    'buttons': ('Helvetica', 11, 'bold'),
                    'big_button': ('Helvetica', 14, 'bold'),
                    'tabs': ('Helvetica', 12, 'bold'),
                    'flag': ('Helvetica', 13, 'bold')},
                    
        'cv2_fonts': {'main_titles': {'font': cv2.FONT_HERSHEY_COMPLEX,'scale':1.0},
                      'sub_titles': {'font': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.9},
                      'state_captions': {'font': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.65, 'v_spacing':.4}},

           'results_viz': {

              'summary': {
                  'font': cv2.FONT_HERSHEY_COMPLEX,
                  'size': {'h': 130, },  # size of the summary area in pixels'
                  # indent for the text in the summary area (x,y) in pixels
                  'text_indent': (20, 10),
                  # fit text in spaces this X smaller than vertical room allows.
                  'text_spacing_frac': .7,
                  'bar_w_frac': 0.5,  # fraction of font scale for bar graph
                  'graph_width_frac': 0.4,  # fraction of the summary area width for the bar graph
                  'graph_indent_frac': 0.1},
               'match_area': {
                  'trace_size': (90, 380),  # w,h in pixels
                  'trace_pad_frac': (.0, 0.05),  # between traces, frac of trace width, height
                  'group_pad_frac': .01,  # between groups, frac of img_width
                  'group_bar_thickness_frac': 0.01},  # fraction of image height for the bar thickness, fraction of image_width
               'trace_params': {
                  'header_font_frac': 0.45,  # fraction of image side length for the header text
                  'return_font_frac': 0.4,  # same for the return text
                  'col_title_frac': 0.4,  # fraction of image side length for the column titles
                  'txt_spacing_frac': 0.2,
                  # fraction of image side length to use as padding between images and text, etc.
                  "pad_frac": 0.2,
                  'font': cv2.FONT_HERSHEY_SIMPLEX,
                  'colors': COLOR_SCHEME},
               'color_key': {'width': 200, 'height': 80}},

          'keys': {'h_pad_px': 15}}

TITLE_INDENT = 5

WIN_SIZE = (1920, 990)
