

MID_X = 0.38  # "state-tabs" frame (sate/value/update images) to the right of this point
MID_Y = 0.5  # "step-visualization" frame below this point
LOW_Y = 1.0
HALF_MID_X = 0.15
import cv2
LAYOUT = {'frames': {'control': {'x_rel': (0.0, HALF_MID_X),
                                 'y_rel': (0.0, MID_Y)},

                     'selection': {'x_rel': (0.0, HALF_MID_X),
                                   'y_rel': (MID_Y, LOW_Y)},
                     'step-visualization': {'x_rel': (HALF_MID_X, MID_X),
                                            'y_rel': (0.0, 1.0)},
                     'state-tabs': {'x_rel': (MID_X, 1.0),
                                    'y_rel': (0.0, 1.0)}
                     },
          'color_key': {'height': 60, 'width': 250},
          
          'state_key': {'height': 60, 'width': 70},  # should be same height as color_key

          'margin_rel': .0025,  # margin between frames in relative coordinates

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

          'results_viz': {'summary': {'font': cv2.FONT_HERSHEY_COMPLEX,
                                      'size': {'h': 130, 'text_w': 500},  # size of the summary area in pixels'
                                      # indent for the text in the summary area (x,y) in pixels
                                      'text_indent': (20, 10),
                                      # fit text in spaces this X smaller than vertical room allows.
                                      'text_spacing_frac': .7,
                                      'bar_w_frac': 0.5,  # fraction of font scale for bar graph
                                      'graph_width_frac': 0.4,  # fraction of the summary area width for the bar graph
                                      'graph_indent_frac': 0.1},
                          'matches': {'state_space_size': 20,  # determines state image size
                                      'indent_frac': 0.5,
                                      'outer_indent_frac': 0.025,  # between image boundary and result box
                                      'inner_indent_frac': 0.033,  # between result box and trace images
                                      'box_thickness_frac': 0.015,  # fraction of the image size for the box around the trace images
                                      }}}
TITLE_INDENT = 5


WIN_SIZE = (1920, 990)
