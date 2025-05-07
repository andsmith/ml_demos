

MID_X = 0.4  # "state-tabs" frame (sate/value/update images) to the right of this point
MID_Y = 0.4  # "step-visualization" frame below this point
HALF_MID_X = 0.19

LAYOUT = {'frames': {'selection': {'x_rel': (0.0, HALF_MID_X),
                                   'y_rel': (0.0, MID_Y)},

                     'control': {'x_rel': (HALF_MID_X, MID_X),
                                 'y_rel': (0.0, MID_Y)},
                     'step-visualization': {'x_rel': (0.0, MID_X),
                                            'y_rel': (MID_Y, 1.0)},
                     'state-tabs': {'x_rel': (HALF_MID_X, 1.0),
                                    'y_rel': (0.0, 1.0)}
                     },

          'margin_px': 5,

          'fonts': {'panel_title': ('Helvetica', 16),
                    'title': ('Helvetica', 14, 'bold'),
                    'default': ('Helvetica', 12),
                    'menu': ('Helvetica', 12, ),
                    'buttons': ('Helvetica', 11, 'bold'),
                    'flag': ('Helvetica', 13, 'bold')}}


FRAME_TITLES = {'selection': 'Select Algorithm',
                'control': 'Status / Controls', }  # Algorithms set other frames' titles.

WIN_SIZE = (1920, 990)
