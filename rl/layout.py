

MID_X = 0.38  # "state-tabs" frame (sate/value/update images) to the right of this point
MID_Y = 0.5  # "step-visualization" frame below this point
LOW_Y = 1.0
HALF_MID_X = 0.15

LAYOUT = {'frames': {'control': {'x_rel': (0.0, HALF_MID_X),
                                   'y_rel': (0.0, MID_Y)},

                     'selection': {'x_rel': (0.0, HALF_MID_X),
                                 'y_rel': (MID_Y, LOW_Y)},
                     'step-visualization': {'x_rel': (HALF_MID_X, MID_X),
                                            'y_rel': (0.0, 1.0)},
                     'state-tabs': {'x_rel': (MID_X, 1.0),
                                    'y_rel': (0.0, 1.0)}
                     },

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
                    'flag': ('Helvetica', 13, 'bold')}}
TITLE_INDENT = 5


WIN_SIZE = (1920, 990)
