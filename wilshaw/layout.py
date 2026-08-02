"""
Layout constants for the Willshaw app: window size, relative panel bounding
boxes, fonts and the colour scheme.  Kept in one place so every module refers
to the same values (mirrors the ../rl/layout.py convention).
"""

# Default window size (width, height) in pixels.
WIN_SIZE = (1600, 950)

# Relative bounding boxes (fractions of the window).  Each is (start, end).
# A short status/title bar across the top, a narrow control panel down the
# left, and the tabbed main view filling the rest.
LAYOUT = {
    'status':  {'x_rel': (0.0, 1.0),  'y_rel': (0.0, 0.075)},
    'control': {'x_rel': (0.0, 0.16), 'y_rel': (0.075, 1.0)},
    'main':    {'x_rel': (0.16, 1.0), 'y_rel': (0.075, 1.0)},
    'margin_rel': 0.004,
}

# Named fonts.  Tabs are deliberately large and bold ("large, noticeable font").
FONTS = {
    'title':       ('Helvetica', 20, 'bold'),
    'status':      ('Helvetica', 13),
    'status_small': ('Helvetica', 10),
    'tabs':        ('Helvetica', 16, 'bold'),
    'panel_title': ('Helvetica', 12, 'bold', 'underline'),
    'section':     ('Helvetica', 11, 'bold'),
    'default':     ('Helvetica', 10),
    'buttons':     ('Helvetica', 10, 'bold'),
    'small':       ('Helvetica', 9),
    'value':       ('Consolas', 10),
}

# Colour scheme -- RGB tuples in 0..255.  A light, high-contrast theme so the
# black/white Willshaw matrix and the colour-coded vectors read clearly.
COLOR_SCHEME = {
    'bg':          (236, 237, 240),   # window background
    'panel_bg':    (216, 219, 226),   # control panel / frames
    'view_bg':     (250, 250, 252),   # drawing canvas background
    'text':        (24, 24, 30),
    'text_dim':    (110, 110, 120),
    'lines':       (150, 152, 160),

    # Domain colours (used when drawing patterns / states).
    'off':         (208, 210, 214),   # inactive unit (the "0" gray)
    'on':          (18, 18, 22),       # correct active unit (the "1" black)
    'spurious':    (150, 12, 12),      # excitation noise -- dark red
    'missing':     (150, 220, 150),    # inhibition noise -- light green
    'highlight':   (56, 118, 230),     # highlighted / active-cue rows -- blue

    # Matrix + excitation.
    'matrix_on':   (20, 20, 24),        # W_ij == 1
    'matrix_off':  (252, 252, 254),     # W_ij == 0
    'ref':         (20, 20, 24),        # reference pattern

    # Accents.
    'optimal':     (220, 120, 0),       # optimal-threshold marker
    'good':        (30, 140, 60),
    'bad':         (170, 30, 30),
    'neon':        (57, 255, 20),       # "about to be used" KB arrow highlight

    # Per-symbol colours (module_art.py multi-symbol wedge/split coding).
    'sym_colors': [
        (56, 118, 230),    # symbol A - blue
        (230, 90, 20),     # symbol B - orange
        (40, 160, 90),     # symbol C - green
        (170, 40, 170),    # symbol D - purple
    ],
}
