"""
Rendering for the confabulation demo app: a horizontal row of word
modules -- locked-down words drawn as sparse module-art activations
(``module_art.py``), still-inferring words as a graded "simulated symbol
distribution" (a superposition of the ranked candidates' sparse codes,
weighted by probability) plus a ranked candidate table below -- with
curved knowledge-base "arches" above the row showing which lateral links
are in play (see ``nl_demo_tab.md``).

Pure rendering: returns uint8 RGB numpy arrays like ``draw_utils.py`` /
``module_art.py``, no Tk here. Consumes duck-typed unit objects exposing
``.label``, ``.locked_word`` (str or None) and ``.candidates`` (list of
``(word, weight)`` or None) -- ``confabulation_stepper.UnitState``
satisfies this directly, with no import dependency needed in either
direction.
"""

import cv2
import numpy as np

import draw_utils as du
import module_art as ma
from layout import COLOR_SCHEME as CS

ARCH_ZONE_H = 70     # pixels reserved at the top of the row for KB arches
LABEL_H = 16          # unit label ("5", "next", ...) row height
TABLE_ROWS = 10       # ranked-candidate rows shown per inferring unit
TABLE_ROW_H = 13
PADDING = 8
_PACK_SEED = 0        # fixed so the module circle layout never re-jitters between draws


# --------------------------------------------------------------------------- #
#  Row of modules
# --------------------------------------------------------------------------- #
def _candidate_membership(candidates, word_codes, vocab_index, D):
    """
    (D, 1) graded membership for a still-inferring unit: each candidate
    word's sparse code contributes its probability weight to every neuron
    it activates, so neurons shared by several likely candidates read
    brighter -- a "simulated symbol distribution" over the module.

    Normalizes to the *unit's own* peak (not via ``module_art
    .mono_excitation``, whose ``h / max(vmax, 1)`` convention assumes
    excitation counts >= 1 -- candidate weights are probabilities that are
    almost always < 1, so that clamp would silently dim every neuron down
    to ``vmax`` instead of the intended peak-at-1.0 brightness).
    """
    level = np.zeros(D, dtype=np.float32)
    for word, weight in candidates:
        wi = vocab_index.get(word)
        if wi is not None:
            level[word_codes[wi]] += weight
    peak = level.max()
    if peak > 0:
        level = level / peak
    return level[:, None]


def draw_demo_row(width, height, units, word_codes, vocab_index, D, cs=CS):
    """
    Draw one horizontal row: one square module per unit, arch space
    reserved (but not drawn -- see :func:`draw_kb_arches`) at the top.

    :param units: list of duck-typed unit objects/dicts (see module
        docstring)
    :param word_codes: (V, S) int array of cosmetic sparse codes (row mu
        is ``vocab_index``'s word mu's code), from
        ``confabulation_stepper.build_word_codes``
    :param vocab_index: ``{token: row index into word_codes}``
    :param D: dimensionality the codes were built at
    :param cs: colour scheme dict
    :return: ``(img, unit_x_centers)`` -- img is ``(height, width, 3)``
        uint8; ``unit_x_centers`` is each unit's horizontal pixel center,
        parallel to ``units`` (feed to :func:`draw_kb_arches`)
    """
    img = du.blank(width, height, color=cs['view_bg'])
    n = len(units)
    if n == 0:
        return img, []

    col_w = width / n
    info_h = TABLE_ROWS * TABLE_ROW_H + 6
    module_size = int(max(24, min(col_w - 2 * PADDING,
                                   height - ARCH_ZONE_H - LABEL_H - info_h - 2 * PADDING)))
    centers, r = ma.pack_circles(D, module_size, rng=np.random.default_rng(_PACK_SEED))

    x_centers = []
    for i, u in enumerate(units):
        cx = int((i + 0.5) * col_w)
        x_centers.append(cx)
        x0 = cx - module_size // 2
        y0 = ARCH_ZONE_H + LABEL_H

        label = getattr(u, 'label', i)
        du.put_label(img, str(label), (max(0, x0), ARCH_ZONE_H + LABEL_H - 4), 0.42, cs['text'])

        locked = getattr(u, 'locked_word', None)
        # Prefer the live in-progress preview (updated after every single
        # lateral step) over the last-committed candidates (only updated at
        # reduce time), so the module visibly changes after each Step click
        # during the lateral phase, not just at the lateral->reduce boundary.
        candidates = getattr(u, 'display_candidates', None) or getattr(u, 'candidates', None)

        if locked is not None and locked in vocab_index:
            membership = ma.mono_activation(word_codes[vocab_index[locked]], D)
            draw_mode = 'binary'
        elif candidates:
            # "Simulated symbol distribution": superpose the still-inferring
            # unit's ranked candidates' sparse codes, each weighted by its
            # current probability, so the module shows a graded activation
            # rather than sitting blank until it locks (nl_demo_tab.md:
            # "show a simulated symbol distribution").
            membership = _candidate_membership(candidates, word_codes, vocab_index, D)
            draw_mode = 'level'
        else:
            membership = ma.mono_activation([], D)
            draw_mode = 'binary'
        # At high D the circles are only a couple of pixels across, and a
        # 1px outline on *every* neuron (active or not) visually swamps the
        # sparse fill that actually encodes activation -- the whole module
        # reads as a uniform dark stipple regardless of which neurons are
        # on. Drop outlines once circles get that small so only the active
        # neurons draw anything, which is what actually stays legible.
        mod_img = ma.draw_module(module_size, centers, r, membership, [cs['on']],
                                  mode=draw_mode, outline=(r >= 3.0), cs=cs)
        img[y0:y0 + module_size, x0:x0 + module_size] = mod_img

        y_info = y0 + module_size + 4
        if locked is not None:
            du.put_label(img, locked, (max(0, x0), y_info + 16), 0.55, cs['good'], thickness=2)
        elif candidates:
            for row_i, (word, weight) in enumerate(candidates[:TABLE_ROWS]):
                ty = y_info + (row_i + 1) * TABLE_ROW_H
                du.put_label(img, f"{weight:.5f}  {word}", (max(0, x0), ty), 0.35, cs['text'])
        else:
            du.put_label(img, "(pending)", (max(0, x0), y_info + 16), 0.4, cs['text_dim'])

    return img, x_centers


# --------------------------------------------------------------------------- #
#  KB arches
# --------------------------------------------------------------------------- #
def draw_kb_arches(img, unit_labels, unit_x_centers, links, cs=CS, arch_zone_h=ARCH_ZONE_H):
    """
    Draw curved arcs in the top ``arch_zone_h`` pixels of ``img``, one per
    KB link currently in play, connecting source/destination unit
    x-centers; arc height scales with horizontal distance.

    :param unit_labels: list of unit labels, parallel to ``unit_x_centers``
    :param unit_x_centers: as returned by :func:`draw_demo_row`
    :param links: list of ``(src_label, dst_label, kb_key, state)`` as
        from ``confabulation_stepper.StepEngine.arch_links()`` -- ``state``
        is ``'applied'``, ``'next'`` or ``'pending'``
    :return: ``img`` (drawn in place, also returned for convenience)
    """
    pos = dict(zip(unit_labels, unit_x_centers))
    spans = [abs(pos[s] - pos[d]) for s, d, _, _ in links if s in pos and d in pos]
    max_span = max(spans) if spans else 1
    color_by_state = {'applied': cs['good'], 'next': cs['neon'], 'pending': cs['lines']}

    # Draw 'next' last so its highlight is never occluded by other arcs.
    for src, dst, _kb_key, state in sorted(links, key=lambda link: link[3] == 'next'):
        if src not in pos or dst not in pos or pos[src] == pos[dst]:
            continue
        x0, x1 = pos[src], pos[dst]
        span = abs(x1 - x0)
        arc_h = int(8 + (arch_zone_h - 16) * (span / max_span))
        cx, cy = (x0 + x1) / 2.0, arch_zone_h - 4
        axes = (max(4, int(span / 2)), max(4, arc_h))
        color = tuple(int(c) for c in color_by_state[state])
        thickness = 3 if state == 'next' else 1
        cv2.ellipse(img, (int(cx), cy), axes, 0, 180, 360, color, thickness, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
#  Stand-alone tests
# --------------------------------------------------------------------------- #
class _FakeUnit(object):
    def __init__(self, label, locked_word=None, candidates=None):
        self.label = label
        self.locked_word = locked_word
        self.candidates = candidates


def _test_module_row_art():
    rng = np.random.default_rng(0)
    vocab = ["the", "cat", "sat", "dog", "ran"]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    D, S = 60, 6
    word_codes = np.stack([np.sort(rng.choice(D, size=S, replace=False)) for _ in vocab])

    units = [
        _FakeUnit(1, locked_word="the"),
        _FakeUnit(2, locked_word="cat"),
        _FakeUnit(3, candidates=[("sat", 0.7), ("ran", 0.3)]),
        _FakeUnit(4, candidates=None),
    ]
    img, x_centers = draw_demo_row(700, 260, units, word_codes, vocab_index, D)
    assert img.shape == (260, 700, 3) and img.dtype == np.uint8
    assert len(x_centers) == 4
    assert all(0 <= x < 700 for x in x_centers)

    links = [(1, 2, (1, 2), 'applied'), (1, 3, (1, 3), 'next'), (2, 3, (2, 3), 'pending')]
    labels = [u.label for u in units]
    out = draw_kb_arches(img, labels, x_centers, links)
    assert out is img

    # empty row / no links must not crash
    img2, xc2 = draw_demo_row(400, 200, [], word_codes, vocab_index, D)
    assert xc2 == []
    draw_kb_arches(img2, [], [], [])

    print("module_row_art OK")


def _preview_window():
    import tkinter as tk
    from gui_base import blit, tk_color_from_rgb
    from layout import COLOR_SCHEME as CS

    rng = np.random.default_rng(1)
    vocab = ["once", "upon", "a", "time", "there", "was"]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    D, S = 400, 9
    word_codes = np.stack([np.sort(rng.choice(D, size=S, replace=False)) for _ in vocab])

    units = [
        _FakeUnit(1, locked_word="once"),
        _FakeUnit(2, locked_word="upon"),
        _FakeUnit(3, locked_word="a"),
        _FakeUnit(4, locked_word="time"),
        _FakeUnit(5, locked_word="there"),
        _FakeUnit(6, candidates=[("was", 0.8123), ("is", 0.1002), ("were", 0.0512)]),
    ]
    img, x_centers = draw_demo_row(1200, 320, units, word_codes, vocab_index, D)
    links = [(i, 6, (i, 6), 'applied') for i in range(1, 5)] + [(5, 6, (5, 6), 'next')]
    draw_kb_arches(img, [u.label for u in units], x_centers, links)

    root = tk.Tk()
    root.title("module_row_art preview")
    root.configure(bg=tk_color_from_rgb(CS['bg']))
    lbl = tk.Label(root, bg=tk_color_from_rgb(CS['view_bg']))
    lbl.pack(padx=8, pady=8)
    blit(lbl, img)
    root.mainloop()


if __name__ == '__main__':
    _test_module_row_art()
    _preview_window()
