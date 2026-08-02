"""
Drawing primitives for a "confabulation module" -- a square packed edge to
edge with tiny circles, one per neuron (closest packing + a bit of noise).
Stand-alone for now; intended to become the per-module widget drawn on
willshaw_app.py.  Follows draw_utils.py's conventions: every drawing
function returns a uint8 RGB numpy array, colours come from
layout.COLOR_SCHEME, cv2 does the actual drawing.

Packing (positions/radius) and colouring (per-neuron fill) are kept as two
separate functions so a caller can compute the circle layout once per
(D, size) and cache it, redrawing only colours on each update -- circle
positions must not re-jitter every frame.

Colouring is one unified primitive: each neuron carries a membership level
in [0, 1] per "channel" (a channel is either a single black/grey channel
for general excitation/activation, or one channel per symbol).  A neuron
with zero present channels is blank; one present channel fills the whole
circle; two present channels split it into a semicircle each; three or
more split it into that many pie wedges.  Mono black/grey display is just
the one-channel case of this same wedge renderer.
"""

import cv2
import numpy as np

from layout import COLOR_SCHEME as CS


# --------------------------------------------------------------------------- #
#  Circle packing
# --------------------------------------------------------------------------- #
def _hex_grid(r, size, margin):
    """Hex-packed candidate centers of radius ``r`` filling a size x size
    square, row-major top to bottom, left to right within each row."""
    if r <= 0:
        return []
    dx = 2.0 * r
    dy = r * np.sqrt(3.0)
    lo, hi = margin + r, size - margin - r
    if hi < lo:
        return []
    pts = []
    y = lo
    row = 0
    while y <= hi + 1e-6:
        x0 = lo + (r if row % 2 else 0.0)
        x = x0
        while x <= hi + 1e-6:
            pts.append((x, y))
            x += dx
        y += dy
        row += 1
    return pts


def pack_circles(D, size, jitter=0.12, margin=2.0, rng=None):
    """
    Hexagonal (closest) circle packing of ``D`` circles filling a
    ``size`` x ``size`` square, with small per-circle positional noise.

    :param D: number of circles (neurons)
    :param size: side length of the square canvas, in pixels
    :param jitter: per-circle positional noise, as a fraction of the radius
    :param margin: empty border in pixels
    :param rng: numpy Generator (default: a fresh ``default_rng()``)
    :return: (centers, r) -- centers is a (D, 2) float array of (x, y)
        pixel coordinates, r is the (pre-jitter) circle radius
    """
    D = int(D)
    if D <= 0:
        return np.empty((0, 2), np.float64), 0.0
    rng = np.random.default_rng() if rng is None else rng

    lo, hi = 0.1, max(size / 2.0, 0.2)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(_hex_grid(mid, size, margin)) >= D:
            lo = mid
        else:
            hi = mid
    r = lo
    pts = _hex_grid(r, size, margin)
    centers = np.asarray(pts[:D], dtype=np.float64)
    if jitter > 0 and len(centers):
        centers += rng.uniform(-jitter * r, jitter * r, size=centers.shape)
    return centers, r


# --------------------------------------------------------------------------- #
#  Membership builders
# --------------------------------------------------------------------------- #
def mono_activation(active_idx, D):
    """(D, 1) binary membership array: 1.0 at ``active_idx``, else 0.  Pair
    with ``colors=[cs['on']]``, ``mode='binary'`` for the default
    "solid black if active, blank if inactive" mono display."""
    m = np.zeros((D, 1), np.float32)
    if len(active_idx):
        m[np.asarray(active_idx, np.int64), 0] = 1.0
    return m


def mono_excitation(h, vmax=None):
    """(D, 1) membership array normalized to [0, 1] from an excitation
    vector (same ``h / max(vmax, 1)`` convention as
    draw_utils.greyscale_colors).  Pair with ``colors=[cs['on']]``,
    ``mode='level'`` for the grayscale mono display."""
    h = np.asarray(h, np.float32)
    vmax = float(h.max()) if vmax is None else float(vmax)
    t = np.clip(h / max(vmax, 1.0), 0.0, 1.0)
    return t[:, None]


def symbol_membership(symbol_idx_list, D, levels=None):
    """
    Build a (D, K) membership array from K symbols.

    :param symbol_idx_list: list of K sparse active-index arrays (as
        produced by e.g. willshaw_core.make_patterns rows)
    :param D: dimensionality
    :param levels: optional parallel list of K (D,) arrays already
        normalized to [0, 1], used as each neuron's per-symbol value
        instead of flat 1.0 membership -- lets draw_module shade each
        symbol's own excitation level while still wedge-splitting neurons
        shared between symbols.
    :return: (D, K) float32 array
    """
    K = len(symbol_idx_list)
    m = np.zeros((D, K), np.float32)
    for k, idx in enumerate(symbol_idx_list):
        if levels is not None:
            m[:, k] = np.clip(np.asarray(levels[k], np.float32), 0.0, 1.0)
        elif len(idx):
            m[np.asarray(idx, np.int64), k] = 1.0
    return m


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def draw_module(size, centers, r, membership, colors, mode='level',
                 active_thresh=1e-6, outline=True, outline_color=None,
                 cs=CS):
    """
    Render one module square: every circle in ``centers`` filled per its
    row of ``membership``, split into equal pie wedges (a full circle at
    K=1, a semicircle split at K=2, true pie wedges at K>2) over whichever
    channels are present at that neuron.

    :param size: side length of the square canvas, in pixels
    :param centers: (D, 2) circle-center array from pack_circles
    :param r: circle radius from pack_circles
    :param membership: (D, K) float array in [0, 1]; membership[i, k] is
        neuron i's level for channel k (0 => not part of that channel)
    :param colors: (K, 3) uint8/int array, one colour per channel (K=1,
        colors=[cs['on']] for the mono black/grey display)
    :param mode: 'binary' -> present channels fill their wedge at full
        colour strength; 'level' -> wedges are alpha-blended from the
        background colour up to full strength by membership[i, k]
    :param active_thresh: membership values at/under this are "not
        present" for that channel
    :param outline: draw a thin outline circle on every neuron
    :param outline_color: outline colour (default cs['on'], the project's
        "black" domain colour)
    :param cs: colour scheme dict (default layout.COLOR_SCHEME)
    :return: (size, size, 3) uint8 RGB image
    """
    membership = np.clip(np.asarray(membership, np.float32), 0.0, 1.0)
    colors = np.asarray(colors, np.float32)
    D, K = membership.shape
    assert colors.shape == (K, 3), (colors.shape, K)
    outline_color = tuple(int(c) for c in (
        cs['on'] if outline_color is None else outline_color))
    bg = np.array(cs['view_bg'], np.float32)
    ri = max(1, int(round(r)))

    img = np.empty((size, size, 3), np.uint8)
    img[:] = cs['view_bg']

    for i in range(D):
        present = np.where(membership[i] > active_thresh)[0]
        if len(present):
            center = (int(round(centers[i, 0])), int(round(centers[i, 1])))
            step = 360.0 / len(present)
            start = -90.0  # start at 12 o'clock so K=2 splits left/right
            for k in present:
                level = float(membership[i, k]) if mode == 'level' else 1.0
                fill = bg + level * (colors[k] - bg)
                fill = tuple(int(c) for c in np.clip(fill, 0, 255))
                cv2.ellipse(img, center, (ri, ri), 0, start, start + step,
                            fill, -1, cv2.LINE_AA)
                start += step
        if outline:
            center = (int(round(centers[i, 0])), int(round(centers[i, 1])))
            cv2.circle(img, center, ri, outline_color, 1, cv2.LINE_AA)

    return img


# --------------------------------------------------------------------------- #
#  Stand-alone tests
# --------------------------------------------------------------------------- #
def _test_module_art():
    for D, size in [(1, 40), (37, 120), (400, 200)]:
        centers, r = pack_circles(D, size, rng=np.random.default_rng(0))
        assert centers.shape == (D, 2)
        assert r > 0
        assert np.all(centers >= -r) and np.all(centers <= size + r)

    D, size = 50, 160
    centers, r = pack_circles(D, size, jitter=0.0, rng=np.random.default_rng(1))

    # all-zero membership -> background + anti-aliased outlines only, no
    # fill colour (every pixel is a blend of just those two colours).
    blank_m = np.zeros((D, 1), np.float32)
    img = draw_module(size, centers, r, blank_m, [CS['on']], mode='binary')
    assert img.shape == (size, size, 3) and img.dtype == np.uint8
    bg = np.array(CS['view_bg'], np.int16)
    outline = np.array(CS['on'], np.int16)
    lo, hi = np.minimum(bg, outline) - 1, np.maximum(bg, outline) + 1
    flat = img.reshape(-1, 3).astype(np.int16)
    assert np.all((flat >= lo) & (flat <= hi))

    # mono activation: an active neuron's center pixel is (near-)black.
    act_m = mono_activation([0], D)
    img = draw_module(size, centers, r, act_m, [CS['on']], mode='binary')
    cx, cy = int(round(centers[0, 0])), int(round(centers[0, 1]))
    assert tuple(int(v) for v in img[cy, cx]) == tuple(CS['on'])

    # two-symbol split: a shared neuron splits left/right by colour.
    sym_m = np.zeros((D, 2), np.float32)
    sym_m[0] = [1.0, 1.0]
    colors = [CS['sym_colors'][0], CS['sym_colors'][1]]
    img = draw_module(size, centers, r, sym_m, colors, mode='binary')
    ri = max(1, int(round(r)))
    row = img[cy, max(0, cx - ri):cx + ri + 1]
    left_px = tuple(int(v) for v in row[max(0, len(row) // 4)])
    right_px = tuple(int(v) for v in row[min(len(row) - 1, 3 * len(row) // 4)])
    assert left_px == tuple(colors[1]), left_px    # 2nd wedge (90..270) -> left half
    assert right_px == tuple(colors[0]), right_px  # 1st wedge (-90..90) -> right half

    print("module_art OK")


def _preview_window():
    import tkinter as tk
    from gui_base import blit, tk_color_from_rgb

    size = 220
    rng = np.random.default_rng(42)
    D = 260

    centers, r = pack_circles(D, size, jitter=0.15, rng=rng)

    # 1) mono activation (~5% active)
    active_idx = rng.choice(D, size=max(1, D // 20), replace=False)
    img_act = draw_module(size, centers, r, mono_activation(active_idx, D),
                          [CS['on']], mode='binary')

    # 2) mono excitation (random continuous levels)
    h = rng.uniform(0, 1, size=D) ** 2
    img_exc = draw_module(size, centers, r, mono_excitation(h, vmax=1.0),
                          [CS['on']], mode='level')

    # 3) two symbols with an overlapping region
    symA = rng.choice(D, size=D // 3, replace=False)
    symB = rng.choice(D, size=D // 3, replace=False)
    m2 = symbol_membership([symA, symB], D)
    img_sym2 = draw_module(size, centers, r, m2, CS['sym_colors'][:2],
                           mode='binary')

    # 4) four symbols -> pie wedges where they overlap
    syms = [rng.choice(D, size=D // 4, replace=False) for _ in range(4)]
    m4 = symbol_membership(syms, D)
    img_sym4 = draw_module(size, centers, r, m4, CS['sym_colors'][:4],
                           mode='binary')

    root = tk.Tk()
    root.title("module_art preview")
    root.configure(bg=tk_color_from_rgb(CS['bg']))
    row = tk.Frame(root, bg=tk_color_from_rgb(CS['bg']))
    row.pack(padx=8, pady=8)
    for title, img in [("mono activation", img_act),
                        ("mono excitation", img_exc),
                        ("2 symbols (split)", img_sym2),
                        ("4 symbols (wedges)", img_sym4)]:
        col = tk.Frame(row, bg=tk_color_from_rgb(CS['bg']))
        col.pack(side=tk.LEFT, padx=6)
        tk.Label(col, text=title, bg=tk_color_from_rgb(CS['bg']),
                 fg=tk_color_from_rgb(CS['text'])).pack(side=tk.TOP)
        lbl = tk.Label(col, bg=tk_color_from_rgb(CS['view_bg']))
        lbl.pack(side=tk.TOP)
        blit(lbl, img)
    root.mainloop()


if __name__ == '__main__':
    _test_module_art()
    _preview_window()
