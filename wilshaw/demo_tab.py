"""
Demonstration tab: visually illustrate the operation of a small Willshaw
attractor.

Layout (inside the notebook's tab frame):

  +-----------------------------------------------------------------+
  | [1.a 1.b 1.c]  Inhibition / Excitation noise sliders [keep eq]  |
  +----------------+------------------------------------------------+
  |  pattern module|   Attractor frame  (input | W | live module /  |
  |  pattern list  |                     excitation / threshold /    |
  |  (scrollable,  |                     reference)                  |
  |   clickable)   +------------------------------------------------+
  |                |   Iteration frame  (piled binary states | ref  |
  |                |                     | convergence info)        |
  +----------------+------------------------------------------------+

Left-click a pattern to damage it (a fresh random draw of inhibition +
excitation noise) and watch it recover; click it again to deselect.
Right-click a second pattern to activate 50% of its units alongside the
first (demonstrates sparse non-interference).  Mouse over an iteration
vector to inspect that step in the attractor frame; the "1.a/1.b/1.c"
buttons jump straight to the parameter presets used in each part of the demo
script.
"""

import logging
import tkinter as tk

import numpy as np

import draw_utils as du
import module_art as ma
import willshaw_core as core
from gui_base import blit, tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS

ROW_H = 16          # pixel height of one pattern row in the list
PAT_LIST_W = 250    # width of the pattern-list column
PMOD_SIZE = PAT_LIST_W - 24   # pattern-module square (above the list)
LIVE_MOD_W = 130    # live-module square (next to the W matrix), max side
LIVE_GAP = 10        # gap between the W heatmap and the live module
CELL_W = 20          # iteration-vector width (piled left, fixed size)
CELL_GAP = 4         # gap between piled iteration vectors
ITER_INFO_W = 160    # width of the convergence-info side panel

# Demo-script presets (version_2_spec.md, part 1: D=20, S=5 throughout).
DEMO_PARAMS = {'a': {'D': 20, 'S': 5, 'V': 1},
               'b': {'D': 20, 'S': 5, 'V': 2},
               'c': {'D': 20, 'S': 5, 'V': 10}}


class DemoTab(object):
    """Content + interaction for the Demonstration tab."""

    def __init__(self, parent, app):
        """
        :param parent: the tk.Frame for this notebook tab
        :param app: the owning application (provides .root, shared noise state,
            and .params via the control panel)
        """
        self.parent = parent
        self.app = app
        self._rng = np.random.default_rng(1234)

        self._patterns = None      # (V, S) int array
        self._W = None             # (D, D) bool
        self._sig = None           # param signature of the current network

        self._sel_pattern = None   # index of primary selected pattern, or None
        self._dmg = None           # core.damage() result for the primary
        self._sel_pattern2 = None  # index of secondary (right-clicked) pattern
        self._sec_extra = None     # 50%-random active subset of the secondary

        self._result = None        # recover() result for the selection
        self._sel_input = None     # currently-inspected input state (indices)
        self._live_excitation = None   # excitation vector while hovering an
                                        # iteration, else None (-> threshold view)

        self._iter_rects = []      # (x0, x1, state_index, input_state)
        self._last_attr_size = None
        self._last_iter_size = None
        self._pattern_layout_cache = None   # (key, (centers, r)) for pattern module
        self._live_layout_cache = None      # (key, (centers, r)) for live module

        self._build_widgets()

    # ------------------------------------------------------------------ #
    #  Widget construction
    # ------------------------------------------------------------------ #
    def _build_widgets(self):
        bg = tk_color_from_rgb(CS['panel_bg'])
        view_bg = tk_color_from_rgb(CS['view_bg'])
        self.parent.configure(bg=bg)
        self.parent.rowconfigure(1, weight=1)
        self.parent.columnconfigure(1, weight=1)

        # --- top row: demo-preset buttons + damage controls ---------- #
        noise = tk.Frame(self.parent, bg=bg)
        noise.grid(row=0, column=0, columnspan=2, sticky='ew', padx=4, pady=3)

        demo_btns = tk.Frame(noise, bg=bg)
        demo_btns.pack(side=tk.LEFT, padx=(2, 12))
        tk.Label(demo_btns, text="Demo:", bg=bg, fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['section']).pack(side=tk.TOP, anchor=tk.W)
        btn_row = tk.Frame(demo_btns, bg=bg)
        btn_row.pack(side=tk.TOP)
        for part in ('a', 'b', 'c'):
            tk.Button(btn_row, text=f"1.{part}", width=4, font=FONTS['buttons'],
                      command=lambda p=part: self._run_demo_part(p)
                      ).pack(side=tk.LEFT, padx=1)

        tk.Label(noise, text="Damage:", bg=bg, fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['section']).pack(side=tk.LEFT, padx=(2, 8))

        self._inh_var = tk.DoubleVar(value=self.app.inhibition_rate)
        self._exc_var = tk.DoubleVar(value=self.app.excitation_rate)
        self._equal_var = tk.BooleanVar(value=True)

        self._inh_scale = tk.Scale(
            noise, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Inhibition (missing)", length=240, bg=bg,
            highlightthickness=0, font=FONTS['small'],
            command=lambda v: self._on_noise('inh'))
        self._inh_scale.pack(side=tk.LEFT, padx=4)
        self._exc_scale = tk.Scale(
            noise, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Excitation (spurious)", length=240, bg=bg,
            highlightthickness=0, font=FONTS['small'],
            command=lambda v: self._on_noise('exc'))
        self._exc_scale.pack(side=tk.LEFT, padx=4)
        self._inh_scale.set(self.app.inhibition_rate)
        self._exc_scale.set(self.app.excitation_rate)

        tk.Checkbutton(noise, text="keep equal", variable=self._equal_var,
                       bg=bg, fg=tk_color_from_rgb(CS['text']), selectcolor=bg,
                       font=FONTS['default']).pack(side=tk.LEFT, padx=8)

        # legend
        legend = ("● correct   ● spurious   ● missing   ▮ cue-row highlight   "
                  "right-click = 2nd symbol")
        tk.Label(noise, text=legend, bg=bg, fg=tk_color_from_rgb(CS['text_dim']),
                 font=FONTS['small']).pack(side=tk.LEFT, padx=10)

        # --- pattern module + pattern list (row 1, col 0) ------------- #
        left = tk.Frame(self.parent, bg=bg, width=PAT_LIST_W)
        left.grid(row=1, column=0, sticky='ns', padx=(4, 2), pady=2)
        left.pack_propagate(False)

        pmod_fr = tk.Frame(left, bg=view_bg, width=PMOD_SIZE, height=PMOD_SIZE,
                           highlightthickness=1,
                           highlightbackground=tk_color_from_rgb(CS['lines']))
        pmod_fr.pack(side=tk.TOP, pady=(0, 4))
        pmod_fr.pack_propagate(False)
        self._pmod_label = tk.Label(pmod_fr, bg=view_bg)
        self._pmod_label.pack(fill=tk.BOTH, expand=True)
        pmod_fr.bind("<Configure>", lambda e: self._draw_pattern_module())
        self._pmod_fr = pmod_fr

        tk.Label(left, text="Patterns (click to recover)", bg=bg,
                 fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['small']).pack(side=tk.TOP, anchor=tk.W)
        canvas_wrap = tk.Frame(left, bg=bg)
        canvas_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._plist = tk.Canvas(canvas_wrap, bg=tk_color_from_rgb(CS['view_bg']),
                                highlightthickness=0, width=PAT_LIST_W - 20)
        sb = tk.Scrollbar(canvas_wrap, orient=tk.VERTICAL,
                          command=self._plist.yview)
        self._plist.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._plist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._plist.bind("<Button-1>", self._on_pattern_click)
        self._plist.bind("<Button-3>", self._on_pattern_right_click)
        self._plist.bind("<MouseWheel>", self._on_mousewheel)
        self._plist_imgref = None

        # --- right column: attractor (top) + iteration (bottom) ------ #
        right = tk.Frame(self.parent, bg=bg)
        right.grid(row=1, column=1, sticky='nsew', padx=(2, 4), pady=2)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        attr_fr = tk.Frame(right, bg=tk_color_from_rgb(CS['view_bg']),
                           highlightthickness=1,
                           highlightbackground=tk_color_from_rgb(CS['lines']))
        attr_fr.grid(row=0, column=0, sticky='nsew', pady=(0, 2))
        attr_fr.pack_propagate(False)   # packed child image must not resize frame
        self._attr_label = tk.Label(attr_fr, bg=tk_color_from_rgb(CS['view_bg']))
        self._attr_label.pack(fill=tk.BOTH, expand=True)
        attr_fr.bind("<Configure>", lambda e: self._draw_attractor())

        iter_fr = tk.Frame(right, bg=tk_color_from_rgb(CS['view_bg']),
                           highlightthickness=1,
                           highlightbackground=tk_color_from_rgb(CS['lines']))
        iter_fr.grid(row=1, column=0, sticky='nsew')

        iter_canvas_fr = tk.Frame(iter_fr, bg=tk_color_from_rgb(CS['view_bg']))
        iter_canvas_fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        iter_canvas_fr.pack_propagate(False)   # packed child image must not resize frame
        self._iter_label = tk.Label(iter_canvas_fr, bg=tk_color_from_rgb(CS['view_bg']))
        self._iter_label.pack(fill=tk.BOTH, expand=True)
        iter_canvas_fr.bind("<Configure>", lambda e: self._draw_iterations())
        self._iter_label.bind("<Motion>", self._on_iter_motion)
        self._iter_label.bind("<Leave>", self._on_iter_leave)

        iter_info_fr = tk.Frame(iter_fr, bg=tk_color_from_rgb(CS['view_bg']),
                                width=ITER_INFO_W)
        iter_info_fr.pack(side=tk.RIGHT, fill=tk.Y)
        iter_info_fr.pack_propagate(False)
        self._iter_info_lbl = tk.Label(
            iter_info_fr, text="", bg=tk_color_from_rgb(CS['view_bg']),
            fg=tk_color_from_rgb(CS['text']), font=FONTS['default'],
            justify=tk.CENTER, wraplength=ITER_INFO_W - 16)
        self._iter_info_lbl.pack(side=tk.TOP, padx=8, pady=14)

        self._attr_fr = attr_fr
        self._iter_fr = iter_fr
        self._iter_canvas_fr = iter_canvas_fr

    # ------------------------------------------------------------------ #
    #  Demo-script presets
    # ------------------------------------------------------------------ #
    def _run_demo_part(self, part):
        """Reset app state and jump straight to a demo-script preset."""
        dp = DEMO_PARAMS[part]
        params = {'D': dp['D'], 'V': dp['V'], 'S': dp['S'], 'theta': 0, 'mode': 'wta'}
        self.app.control_panel.set_params(params)
        self.app.on_params_changed(f'demo_{part}')

    # ------------------------------------------------------------------ #
    #  Network (re)generation
    # ------------------------------------------------------------------ #
    def on_params_changed(self, source=None):
        """Rebuild the pattern set + matrix and reset the view."""
        p = self.app.params
        self._rng = np.random.default_rng(1234)
        self._patterns = core.make_patterns(p['D'], p['S'], p['V'], self._rng)
        self._W = core.build_W(self._patterns, p['D'])
        self._sig = self._params_sig(p)
        self._sel_pattern = None
        self._dmg = None
        self._sel_pattern2 = None
        self._sec_extra = None
        self._result = None
        self._sel_input = None
        self._live_excitation = None
        self._render_pattern_list()
        self._draw_pattern_module()
        self._draw_attractor()
        self._draw_iterations()
        logging.info("demo: regenerated D=%d S=%d V=%d (density=%.3f)",
                     p['D'], p['S'], p['V'], core.matrix_density(self._W))

    @staticmethod
    def _params_sig(p):
        return (p['D'], p['V'], p['S'], p['theta'], p['mode'])

    def on_show(self):
        """Regenerate if never built, or if params changed while hidden."""
        if self._patterns is None or self._sig != self._params_sig(self.app.params):
            self.on_params_changed('show')

    # ------------------------------------------------------------------ #
    #  Pattern list
    # ------------------------------------------------------------------ #
    def _render_pattern_list(self):
        if self._patterns is None:
            return
        V, S = self._patterns.shape
        D = self.app.params['D']
        w = max(self._plist.winfo_width(), PAT_LIST_W - 24)
        strip_w = w - 4
        img = du.blank(w, V * ROW_H, color=CS['view_bg'])
        for mu in range(V):
            colors = du.binary_colors(self._patterns[mu], D)
            row = du.strip(colors, strip_w, ROW_H - 3, horizontal=True,
                           sep=(D <= 64))
            y0 = mu * ROW_H + 1
            img[y0:y0 + ROW_H - 3, 2:2 + strip_w] = row
            if mu == self._sel_pattern:
                img[mu * ROW_H:mu * ROW_H + ROW_H, 0:w] = self._blend_border(
                    img[mu * ROW_H:mu * ROW_H + ROW_H, 0:w], CS['sym_colors'][0])
            elif mu == self._sel_pattern2:
                img[mu * ROW_H:mu * ROW_H + ROW_H, 0:w] = self._blend_border(
                    img[mu * ROW_H:mu * ROW_H + ROW_H, 0:w], CS['sym_colors'][1])
        self._plist_arr = img
        from PIL import Image, ImageTk
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self._plist.delete('all')
        self._plist.create_image(0, 0, anchor=tk.NW, image=photo)
        self._plist_imgref = photo
        self._plist.configure(scrollregion=(0, 0, w, V * ROW_H))

    @staticmethod
    def _blend_border(sub, color, alpha=0.35):
        out = sub.copy()
        c = np.array(color, np.float32)
        out[:] = (alpha * c + (1 - alpha) * out.astype(np.float32)).astype(np.uint8)
        return out

    def _on_mousewheel(self, event):
        self._plist.yview_scroll(int(-event.delta / 120), 'units')

    def _on_pattern_click(self, event):
        if self._patterns is None:
            return
        y = int(self._plist.canvasy(event.y))
        mu = y // ROW_H
        if not (0 <= mu < len(self._patterns)):
            return
        if mu == self._sel_pattern:
            self._deselect_all()
        else:
            self._select_primary(mu)

    def _on_pattern_right_click(self, event):
        if self._patterns is None:
            return
        y = int(self._plist.canvasy(event.y))
        mu = y // ROW_H
        if 0 <= mu < len(self._patterns):
            self._select_secondary(mu)

    def _select_primary(self, mu):
        """Damage pattern mu with a fresh random draw and run recovery."""
        p = self.app.params
        self._sel_pattern = mu
        tgt = self._patterns[mu]
        self._dmg = core.damage(tgt, self.app.inhibition_rate,
                                self.app.excitation_rate, p['D'], self._rng)
        self._recompute_recovery()
        self._render_pattern_list()
        self._draw_pattern_module()
        self._draw_attractor()
        self._draw_iterations()

    def _deselect_all(self):
        """Clear the primary + secondary selection entirely."""
        self._sel_pattern = None
        self._dmg = None
        self._sel_pattern2 = None
        self._sec_extra = None
        self._result = None
        self._sel_input = None
        self._live_excitation = None
        self._render_pattern_list()
        self._draw_pattern_module()
        self._draw_attractor()
        self._draw_iterations()

    def _select_secondary(self, mu2):
        """Toggle a second symbol, activated at 50% (random half of its units)."""
        if mu2 == self._sel_pattern:
            return
        if mu2 == self._sel_pattern2:
            self._sel_pattern2 = None
            self._sec_extra = None
        else:
            sym2 = self._patterns[mu2]
            k = max(1, int(round(0.5 * len(sym2))))
            self._sel_pattern2 = mu2
            self._sec_extra = np.sort(
                self._rng.choice(sym2, size=min(k, len(sym2)), replace=False))
        if self._sel_pattern is not None:
            self._recompute_recovery()
        self._render_pattern_list()
        self._draw_pattern_module()
        self._draw_attractor()
        self._draw_iterations()

    def _recompute_recovery(self):
        """Run recall from the primary's damaged cue, unioned with the
        secondary symbol's 50% partial activation (if any)."""
        p = self.app.params
        tgt = self._patterns[self._sel_pattern]
        cue = self._dmg['cue']
        if self._sec_extra is not None and len(self._sec_extra):
            cue = np.union1d(cue, self._sec_extra)
        self._result = core.recover(self._W, cue, tgt, p['theta'], p['mode'],
                                    p['S'], core.DEFAULT_MAX_ITER)
        self._sel_input = self._result['states'][0]     # damaged/combined cue
        self._live_excitation = None

    # ------------------------------------------------------------------ #
    #  Module widgets (pattern module + live module)
    # ------------------------------------------------------------------ #
    def _get_module_layout(self, kind, D, size):
        """Cached (centers, r) circle packing for a square of the given size;
        recomputed only when (D, size) changes so circles don't re-jitter."""
        key = (D, size)
        cache_attr = f'_{kind}_layout_cache'
        cached = getattr(self, cache_attr)
        if cached is not None and cached[0] == key:
            return cached[1]
        layout = ma.pack_circles(D, size, rng=np.random.default_rng(0))
        setattr(self, cache_attr, (key, layout))
        return layout

    def _draw_pattern_module(self):
        """Square 'module' view above the pattern list: blank when nothing is
        selected, mono when one symbol is selected, colour-split when two
        symbols are selected (spec: 'color-code their representation')."""
        size = min(self._pmod_fr.winfo_width(), self._pmod_fr.winfo_height())
        if size <= 1:
            return
        if self._patterns is None:
            blit(self._pmod_label, du.blank(size, size, color=CS['view_bg']))
            return
        D = self.app.params['D']
        centers, r = self._get_module_layout('pattern', D, size)

        sel1, sel2 = self._sel_pattern, self._sel_pattern2
        if sel1 is None and sel2 is None:
            membership = ma.mono_activation([], D)
            colors = [CS['on']]
        elif sel1 is not None and sel2 is not None:
            membership = ma.symbol_membership(
                [self._patterns[sel1], self._patterns[sel2]], D)
            colors = CS['sym_colors'][:2]
        else:
            only = sel1 if sel1 is not None else sel2
            membership = ma.mono_activation(self._patterns[only], D)
            colors = [CS['on']]

        img = ma.draw_module(size, centers, r, membership, colors, mode='binary')
        blit(self._pmod_label, img)

    # ------------------------------------------------------------------ #
    #  Attractor frame
    # ------------------------------------------------------------------ #
    def _draw_attractor(self):
        w = self._attr_fr.winfo_width()
        h = self._attr_fr.winfo_height()
        if w <= 1 or h <= 1:
            return
        if self._W is None:
            img = du.blank(w, h, color=CS['view_bg'])
            du.put_label(img, "Click a pattern on the left to begin.",
                         (14, h // 2), 0.5, CS['text_dim'])
            blit(self._attr_label, img)
            return

        p = self.app.params
        D = p['D']
        pad = 8
        iv_w = 20
        label_h = 16
        bar_h = 16
        gap = 4
        bottom = 3 * (bar_h + label_h + gap) + 24     # room for the summary line
        top = label_h
        hm = max(40, min(w - iv_w - 4 * pad - LIVE_MOD_W - LIVE_GAP,
                         h - bottom - top - pad))
        content_w = iv_w + 4 + hm + LIVE_GAP + LIVE_MOD_W
        x_iv = max(pad, (w - content_w) // 2)
        x_hm = x_iv + iv_w + 4
        x_live = x_hm + hm + LIVE_GAP
        y_hm = top + 2
        live_size = max(0, min(hm, LIVE_MOD_W))
        y_live = y_hm + max(0, (hm - live_size) // 2)

        img = du.blank(w, h, color=CS['view_bg'])

        if self._sel_input is None:
            # Deselected: show the bare, un-shaded W matrix only.
            heat = du.heatmap_W(self._W, hm, hm, highlight_rows=None)
            img[y_hm:y_hm + hm, x_hm:x_hm + hm] = heat
            du.put_label(img, f"W  ({D}x{D}, density={core.matrix_density(self._W):.2f})",
                         (x_hm, top), 0.4, CS['text'])
            if live_size >= 8:
                centers, r = self._get_module_layout('live', D, live_size)
                membership = ma.mono_activation([], D)
                live_img = ma.draw_module(live_size, centers, r, membership,
                                          [CS['on']], mode='binary')
                img[y_live:y_live + live_size, x_live:x_live + live_size] = live_img
                du.put_label(img, "module", (x_live, top), 0.4, CS['text'])
            blit(self._attr_label, img)
            return

        tgt = self._patterns[self._sel_pattern]
        inp = self._sel_input
        hexc = core.excitation(self._W, inp)
        out = core.threshold_state(hexc, p['theta'], p['mode'], p['S'])

        # input vector (colour-coded vs target), aligned to heatmap rows
        ivec = du.strip(du.provenance_colors(inp, tgt, D), iv_w, hm,
                        horizontal=False, sep=(D <= 64))
        img[y_hm:y_hm + hm, x_iv:x_iv + iv_w] = ivec
        du.put_label(img, "in", (x_iv, top), 0.4, CS['text'])

        # W heatmap with active-cue rows highlighted blue
        heat = du.heatmap_W(self._W, hm, hm, highlight_rows=inp)
        img[y_hm:y_hm + hm, x_hm:x_hm + hm] = heat
        du.put_label(img, f"W  ({D}x{D}, density={core.matrix_density(self._W):.2f})",
                     (x_hm, top), 0.4, CS['text'])

        # live module: excitation while hovering an iteration, else the
        # thresholded/binary view of the currently-inspected state
        if live_size >= 8:
            centers, r = self._get_module_layout('live', D, live_size)
            if self._live_excitation is not None:
                membership = ma.mono_excitation(self._live_excitation)
                mode = 'level'
            else:
                membership = ma.mono_activation(inp, D)
                mode = 'binary'
            live_img = ma.draw_module(live_size, centers, r, membership,
                                      [CS['on']], mode=mode)
            img[y_live:y_live + live_size, x_live:x_live + live_size] = live_img
            du.put_label(img, "module", (x_live, top), 0.4, CS['text'])

        # stacked horizontal bars beneath the heatmap
        y = y_hm + hm + gap
        vmax = max(len(inp), 1)
        rows = [("Excitation  h = Wx",
                 du.strip(du.greyscale_colors(hexc, vmax), hm, bar_h,
                          horizontal=True)),
                (f"Thresholded output  (Top-{p['S']})",
                 du.strip(du.provenance_colors(out, tgt, D), hm, bar_h,
                          horizontal=True, sep=(D <= 64))),
                ("Reference pattern",
                 du.strip(du.binary_colors(tgt, D), hm, bar_h,
                          horizontal=True, sep=(D <= 64)))]
        for title, bar in rows:
            du.put_label(img, title, (x_hm, y + label_h - 4), 0.4, CS['text'])
            yb = y + label_h
            img[yb:yb + bar_h, x_hm:x_hm + hm] = bar
            y += label_h + bar_h + gap

        # match/mismatch summary
        rec, inc = core.score(out, tgt, p['S'])
        du.put_label(img, f"recovered {100*rec:.0f}%   spurious {100*inc:.0f}%",
                     (x_hm, y + 10), 0.45,
                     CS['good'] if rec >= 0.999 and inc < 1e-6 else CS['text'])
        blit(self._attr_label, img)

    # ------------------------------------------------------------------ #
    #  Iteration frame
    # ------------------------------------------------------------------ #
    def _draw_iterations(self):
        w = self._iter_canvas_fr.winfo_width()
        h = self._iter_canvas_fr.winfo_height()
        if w <= 1 or h <= 1:
            return
        self._iter_rects = []
        if self._result is None:
            img = du.blank(w, h, color=CS['view_bg'])
            du.put_label(img, "Recovery iterations appear here.",
                         (14, h // 2), 0.45, CS['text_dim'])
            blit(self._iter_label, img)
            self._update_iter_info()
            return

        p = self.app.params
        D = p['D']
        tgt = self._patterns[self._sel_pattern]
        states = self._result['states']
        pad = 8
        top = 18
        vec_h = max(30, h - top - 24)

        img = du.blank(w, h, color=CS['view_bg'])
        du.put_label(img, "Iterations  (hover to inspect):", (pad, 13), 0.42,
                     CS['text'])
        x = pad
        for i, vec in enumerate(states):
            if x + CELL_W > w - pad:
                break
            strip = du.strip(du.binary_colors(vec, D), CELL_W, vec_h,
                             horizontal=False, sep=(D <= 64))
            img[top:top + vec_h, x:x + CELL_W] = strip
            if i == self._current_sel_index():
                img[top:top + vec_h, x:x + CELL_W] = self._blend_border(
                    img[top:top + vec_h, x:x + CELL_W], CS['highlight'], 0.18)
            lbl = "cue" if i == 0 else str(i)
            du.put_label(img, lbl, (x, top + vec_h + 14), 0.4, CS['text'])
            self._iter_rects.append((x, x + CELL_W, i, states[i]))
            x += CELL_W + CELL_GAP

        # vertical separator, then the reference pattern
        sep_x = x + 3
        if sep_x + 1 < w - pad and x + CELL_W + 12 <= w - pad:
            line = np.array(CS['lines'], np.uint8)
            img[top:top + vec_h, sep_x:sep_x + 1] = line
            x = sep_x + 10
            ref_strip = du.strip(du.binary_colors(tgt, D), CELL_W, vec_h,
                                 horizontal=False, sep=(D <= 64))
            img[top:top + vec_h, x:x + CELL_W] = ref_strip
            du.put_label(img, "ref", (x, top + vec_h + 14), 0.4, CS['text'])

        blit(self._iter_label, img)
        self._update_iter_info()

    def _update_iter_info(self):
        """Small side panel explaining the iteration count / convergence."""
        if self._result is None:
            self._iter_info_lbl.config(text="")
            return
        it = self._result['iters']
        if self._result['converged']:
            text = f"Converged in\n{it} iteration{'s' if it != 1 else ''}"
        else:
            text = f"Did not reach a\nfixed point\n({it} iterations)"
        self._iter_info_lbl.config(text=text)

    def _current_sel_index(self):
        """Index (into states) of the currently-inspected input, or -1."""
        if self._result is None or self._sel_input is None:
            return -1
        for i, s in enumerate(self._result['states']):
            if s is self._sel_input:
                return i
        return -1

    def _on_iter_motion(self, event):
        if not self._iter_rects:
            return
        for x0, x1, i, state in self._iter_rects:
            if x0 <= event.x < x1:
                if state is not self._sel_input:
                    self._sel_input = state
                    excitations = self._result['excitations']
                    self._live_excitation = excitations[i] if i < len(excitations) else None
                    self._draw_attractor()
                    self._draw_iterations()
                return

    def _on_iter_leave(self, _event):
        if self._result is not None:
            default = self._result['states'][0]
            if self._sel_input is not default or self._live_excitation is not None:
                self._sel_input = default
                self._live_excitation = None
                self._draw_attractor()
                self._draw_iterations()

    # ------------------------------------------------------------------ #
    #  Noise controls
    # ------------------------------------------------------------------ #
    def _on_noise(self, which):
        inh = float(self._inh_scale.get())
        exc = float(self._exc_scale.get())
        if self._equal_var.get():
            if which == 'inh' and exc != inh:
                self._exc_scale.set(inh)
                exc = inh
            elif which == 'exc' and inh != exc:
                self._inh_scale.set(exc)
                inh = exc
        self.app.inhibition_rate = inh
        self.app.excitation_rate = exc
        # Re-damage the current primary selection with the new noise level.
        if self._sel_pattern is not None:
            self._select_primary(self._sel_pattern)


# --------------------------------------------------------------------------- #
#  Stand-alone test
# --------------------------------------------------------------------------- #
class _FakeApp(object):
    def __init__(self, root):
        self.root = root
        self.inhibition_rate = 0.25
        self.excitation_rate = 0.25
        self.params = {'D': 40, 'V': 60, 'S': 6, 'theta': 0, 'mode': 'wta'}
        self._demo = None    # set by _test_demo_tab once DemoTab exists

        class _CP(object):
            def set_params(_self, p):
                self.params = p
        self.control_panel = _CP()

    def on_params_changed(self, source=None):
        if self._demo is not None:
            self._demo.on_params_changed(source)


def _test_demo_tab():
    root = tk.Tk()
    root.title("Demo Tab Test")
    root.geometry("1300x800")
    app = _FakeApp(root)
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    demo = DemoTab(frame, app)
    app._demo = demo
    root.after(200, demo.on_show)
    root.mainloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    _test_demo_tab()
