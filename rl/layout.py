

MID_X = 0.35  # "state-tabs" frame (sate/value/update images) to the right of this point
MID_Y = 0.4  # "step-visualization" frame below this point
HALF_MID_X = 0.18

LAYOUT = {'frames': {'selection': {'x_rel': (0.0, HALF_MID_X),
                                   'y_rel': (0.0, MID_Y)},

                     'control': {'x_rel': (HALF_MID_X, MID_X),
                                 'y_rel': (0.0, MID_Y)},
                     'step-visualization': {'x_rel': (0.0, MID_X),
                                            'y_rel': (MID_Y, 1.0)},
                     'state-tabs': {'x_rel': (MID_X, 1.0),
                                    'y_rel': (0.0, 1.0)}
                     },

          'margin_px': 5,

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


WIN_SIZE = (1920, 990)
