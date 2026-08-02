"""
Low-level drawing primitives for the Willshaw app.  Every function returns a
uint8 RGB numpy array so results can be blitted straight into a tk.Label (see
gui_base.blit).  Colours come from layout.COLOR_SCHEME.

Two representations are used throughout:
  * a *state* is a sorted int array of active unit indices,
  * "cell colours" is a (D, 3) uint8 array giving a colour per unit, which the
    strip helpers stretch into a horizontal or vertical band of pixels.
"""

import cv2
import numpy as np

from layout import COLOR_SCHEME as CS


# --------------------------------------------------------------------------- #
#  Cell-colour builders
# --------------------------------------------------------------------------- #
def binary_colors(state_idx, D, cs=CS):
    """(D, 3) colours for a plain binary vector: active -> 'on', else 'off'."""
    colors = np.tile(np.array(cs['off'], np.uint8), (D, 1))
    if len(state_idx):
        colors[np.asarray(state_idx, np.int64)] = cs['on']
    return colors


def provenance_colors(state_idx, target_idx, D, cs=CS):
    """
    (D, 3) colours classifying every unit of ``state_idx`` against the true
    pattern ``target_idx``:

        correct active (in both)      -> 'on'        (black)
        spurious (state, not target)  -> 'spurious'  (dark red)
        missing  (target, not state)  -> 'missing'   (light green)
        inactive (neither)            -> 'off'       (light gray)
    """
    colors = np.tile(np.array(cs['off'], np.uint8), (D, 1))
    smask = np.zeros(D, bool)
    tmask = np.zeros(D, bool)
    if len(state_idx):
        smask[np.asarray(state_idx, np.int64)] = True
    if len(target_idx):
        tmask[np.asarray(target_idx, np.int64)] = True
    colors[smask & tmask] = cs['on']
    colors[smask & ~tmask] = cs['spurious']
    colors[~smask & tmask] = cs['missing']
    return colors


def greyscale_colors(h, vmax, cs=CS):
    """(D, 3) greyscale colours for an excitation vector: 0 -> white, vmax -> black."""
    h = np.asarray(h, np.float32)
    t = np.clip(h / max(float(vmax), 1.0), 0.0, 1.0)
    g = (255.0 * (1.0 - t)).astype(np.uint8)
    return np.stack([g, g, g], axis=1)


# --------------------------------------------------------------------------- #
#  Strips  (stretch a (n, 3) colour array into a band of pixels)
# --------------------------------------------------------------------------- #
def strip(cell_colors, width, height, horizontal=True, sep=False, cs=CS):
    """
    Stretch a (n, 3) colour array into a (height, width, 3) band.

    :param horizontal: True -> cells run left-to-right; False -> top-to-bottom
    :param sep: if True and cells are wide enough, draw thin separators
    """
    cell_colors = np.asarray(cell_colors, np.uint8)
    n = len(cell_colors)
    if n == 0:
        img = np.empty((height, width, 3), np.uint8)
        img[:] = cs['view_bg']
        return img
    if horizontal:
        base = cell_colors.reshape(1, n, 3)
    else:
        base = cell_colors.reshape(n, 1, 3)
    img = cv2.resize(base, (width, height), interpolation=cv2.INTER_NEAREST)
    if sep:
        line = np.array(cs['lines'], np.uint8)
        cell_px = (width / n) if horizontal else (height / n)
        if cell_px >= 4:
            for k in range(1, n):
                if horizontal:
                    x = int(round(k * width / n))
                    if 0 <= x < width:
                        img[:, x] = line
                else:
                    y = int(round(k * height / n))
                    if 0 <= y < height:
                        img[y, :] = line
    return img


# --------------------------------------------------------------------------- #
#  Willshaw matrix heatmap
# --------------------------------------------------------------------------- #
def heatmap_W(W, width, height, highlight_rows=None, cs=CS):
    """
    Render the bool Willshaw matrix as a black/white heatmap (1 -> black),
    optionally tinting the given rows blue (to indicate active-cue rows).

    :param W: (D, D) bool matrix
    :param highlight_rows: iterable of row indices to tint, or None
    :return: (height, width, 3) uint8 RGB
    """
    on = np.array(cs['matrix_on'], np.uint8)
    off = np.array(cs['matrix_off'], np.uint8)
    small = np.where(W[:, :, None], on, off).astype(np.uint8)
    if highlight_rows is not None and len(highlight_rows):
        rows = np.asarray(highlight_rows, np.int64)
        blue = np.array(cs['highlight'], np.float32)
        small[rows] = (0.5 * small[rows].astype(np.float32)
                       + 0.5 * blue).astype(np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #
def blank(width, height, color=None, cs=CS):
    """A solid (height, width, 3) image of the given colour (default view_bg)."""
    img = np.empty((max(height, 1), max(width, 1), 3), np.uint8)
    img[:] = cs['view_bg'] if color is None else color
    return img


def put_label(img, text, org, scale=0.4, color=None, cs=CS, thickness=1):
    """Draw a short text label onto ``img`` (in place) and return it."""
    color = cs['text'] if color is None else color
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                tuple(int(c) for c in color), thickness, cv2.LINE_AA)
    return img


def _test_draw_utils():
    D = 40
    idx = np.array([2, 5, 9, 20, 33])
    tgt = np.array([2, 5, 9, 10, 11])
    assert binary_colors(idx, D).shape == (D, 3)
    assert provenance_colors(idx, tgt, D).shape == (D, 3)
    s = strip(binary_colors(idx, D), 200, 20, horizontal=True, sep=True)
    assert s.shape == (20, 200, 3)
    W = np.zeros((D, D), bool)
    W[np.ix_(idx, idx)] = True
    hm = heatmap_W(W, 160, 160, highlight_rows=idx)
    assert hm.shape == (160, 160, 3)
    print("draw_utils OK")


if __name__ == '__main__':
    _test_draw_utils()
