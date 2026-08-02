"""
Named-Entity Reduction Explorer -- a stand-alone Tkinter utility.

Explore how far the TinyStories vocabulary can be shrunk by collapsing the
open-ended cast of characters / list of settings (and named animals & objects)
onto a small canonical cast, plus light lexical normalization.

The window is laid out as::

  +-----------------------------------------------------------------------+
  | status: #stories #sentences vocab | entity histograms (all/char/place) |
  +------------------+----------------------------------------------------+
  | controls         |  bar graphs  (7 bars: 10..100%)                     |
  |  unit / framing  |                                                     |
  |  export          +----------------------+-----------------------------+
  |  ontology opts   |  reduced vocabulary   |  admitted stories/sentences |
  |  (per category)  |  (2-col, by rank)     |  (reduced text)             |
  +------------------+----------------------------------------------------+

Two framings (radio):
  * "By sample %": for each of 10/20/50/75/90/95/100% of samples, the bar is
    the *vocabulary size* required to keep that fraction (cheapest-vocab first).
  * "By vocab %":  for each %, the bar is the *number of samples* usable when
    the vocabulary is clamped to that fraction of the natural reduced size.

Click a bar to fill the reduced-vocabulary list (its budget) and the admitted
stories/sentences.  Export writes the reduced corpus + a reproducible JSON
bundle (re-tokenizing the corpus yields exactly the exported vocabulary).

Run with::  .venv\\Scripts\\python.exe named_reduction_explorer.py
"""

import os

# Pin BLAS/BLIS to a single thread before numpy/spaCy load (see tiny_stories.py
# and memory: willshaw-app-gotchas) -- prevents a "libblis: Aborting." abort on
# interpreter teardown on this Windows box.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox

from gui_base import tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS
import tiny_stories as ts
from named_ontologies import OntologyConfig

WIN_SIZE = (1500, 950)

PAD = 6

# Per-category ontology control spec: (key, label, pool_max).
CATEGORY_CONTROLS = [
    ("char", "Characters", 12),
    ("place", "Places", 8),
    ("animal", "Animals", 12),
    ("object", "Objects", 12),
]

BUCKET_LABELS = ("1", "2", "3", "4", "5+")
MAX_STORIES_SHOWN = 400
MAX_VOCAB_SHOWN = 6000


class ReductionExplorer:
    """The application object."""

    def __init__(self):
        self._c_bg = tk_color_from_rgb(CS["bg"])
        self._c_panel = tk_color_from_rgb(CS["panel_bg"])
        self._c_view = tk_color_from_rgb(CS["view_bg"])
        self._c_text = tk_color_from_rgb(CS["text"])
        self._c_dim = tk_color_from_rgb(CS["text_dim"])
        self._c_line = tk_color_from_rgb(CS["lines"])
        self._c_bar = tk_color_from_rgb(CS["highlight"])
        self._c_sel = tk_color_from_rgb(CS["optimal"])
        self._c_on = tk_color_from_rgb(CS["on"])

        # Data / computation state.
        self.records = None
        self.corpus = None              # ts.Corpus (holds the normalization cache)
        self.meta = None
        self.natural_vocab = 0
        self.n_sentences = 0
        self.histograms = None
        self.reduction = None
        self._bars = []                 # geometry of drawn bars for hit-testing
        self._selection = None          # {'framing','percent','k','value'}
        self._recompute_job = None
        self._gen = 0                   # recompute generation guard
        self._q = queue.Queue()

        # Sub-sampling ("active set") state.
        self._sample_seed = 0           # bumped by the Resample button
        self._active_indices = None     # None == full corpus

        # Editable lemma / synonym-merge table {word: canonical} + corpus hits.
        self.rules = dict(ts.SYNONYMS)
        self._rule_counts = {}          # base word -> corpus occurrences

        self._init_tk()
        self._build_ui()

        self.root.after(50, self._poll)
        self.root.after(150, self._start_load)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _init_tk(self):
        self.root = tk.Tk()
        self.root.title("Named-Entity Reduction Explorer -- TinyStories")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")
        self.root.configure(bg=self._c_bg)
        self.root.minsize(1150, 760)
        style = ttk.Style(self.root)
        try:                                   # make the draggable sashes visible
            style.configure("TPanedwindow", background=self._c_line)
            style.configure("Sash", sashthickness=7, gripcount=12)
        except Exception:
            pass

    def _build_ui(self):
        """Nested ttk.PanedWindows so every frame divider is a draggable sash."""
        self.main_v = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_v.pack(fill=tk.BOTH, expand=True)

        status = tk.Frame(self.main_v, bg=self._c_panel)
        self.main_v.add(status, weight=0)
        self._build_status(status)

        self.body = ttk.PanedWindow(self.main_v, orient=tk.HORIZONTAL)
        self.main_v.add(self.body, weight=1)

        control = tk.Frame(self.body, bg=self._c_panel)
        self.body.add(control, weight=0)
        self._build_controls(control)

        self.right = ttk.PanedWindow(self.body, orient=tk.VERTICAL)
        self.body.add(self.right, weight=1)

        bars = tk.Frame(self.right, bg=self._c_view)
        self.right.add(bars, weight=3)
        self._build_bars(bars)

        self.bottom = ttk.PanedWindow(self.right, orient=tk.HORIZONTAL)
        self.right.add(self.bottom, weight=4)

        vocab = tk.Frame(self.bottom, bg=self._c_panel)
        self.bottom.add(vocab, weight=2)
        self._build_vocab(vocab)

        stories = tk.Frame(self.bottom, bg=self._c_panel)
        self.bottom.add(stories, weight=3)
        self._build_stories(stories)

        self.root.after(160, self._set_initial_sashes)

    def _set_initial_sashes(self):
        self.root.update_idletasks()

        def setpos(pw, index, pos):
            try:
                pw.sashpos(index, int(pos))
            except Exception:
                pass
        setpos(self.main_v, 0, 150)
        setpos(self.body, 0, max(370, int(self.body.winfo_width() * 0.29)))
        setpos(self.right, 0, int(self.right.winfo_height() * 0.46))
        setpos(self.bottom, 0, int(self.bottom.winfo_width() * 0.40))

    def _build_status(self, parent):
        left = tk.Frame(parent, bg=self._c_panel)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))
        tk.Label(left, text="Named-Entity Reduction Explorer", bg=self._c_panel,
                 fg=self._c_text, font=FONTS["title"]).pack(
            side=tk.TOP, anchor=tk.W, padx=PAD, pady=(2, 0))
        self._counts_lbl = tk.Label(left, text="Loading corpus…", bg=self._c_panel,
                                    fg=self._c_text, font=FONTS["status"],
                                    justify=tk.LEFT, anchor=tk.W)
        self._counts_lbl.pack(side=tk.TOP, anchor=tk.W, padx=PAD)
        self._status_lbl = tk.Label(left, text="", bg=self._c_panel,
                                    fg=self._c_dim, font=FONTS["status_small"],
                                    justify=tk.LEFT, anchor=tk.W, wraplength=560)
        self._status_lbl.pack(side=tk.TOP, anchor=tk.W, padx=PAD)

        # Entity histograms fill the rest of the status bar.
        self._hist_canvas = tk.Canvas(parent, bg=self._c_panel, highlightthickness=0)
        self._hist_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._hist_canvas.bind("<Configure>", lambda e: self._draw_histograms())

    def _build_controls(self, parent):
        outer = tk.Frame(parent, bg=self._c_panel)
        outer.pack(fill=tk.BOTH, expand=True)

        # --- top-level controls -------------------------------------- #
        top = tk.Frame(outer, bg=self._c_panel)
        top.pack(side=tk.TOP, fill=tk.X, padx=PAD, pady=(PAD, 2))

        self.unit_var = tk.StringVar(value="story")
        self.framing_var = tk.StringVar(value="sample")

        self._radio_row(top, "Sample unit:",
                        [("Stories", "story"), ("Sentences", "sentence")],
                        self.unit_var, self._on_unit_changed)
        self._radio_row(top, "Constrain by:",
                        [("Sample %", "sample"), ("Vocab %", "vocab")],
                        self.framing_var, self._on_framing_changed)

        # Sub-sampling: pick a without-replacement active set.
        tk.Label(top, text="Active set (sub-sample):", bg=self._c_panel,
                 fg=self._c_text, font=FONTS["section"], anchor=tk.W).pack(
            side=tk.TOP, anchor=tk.W, pady=(3, 0))
        samp = tk.Frame(top, bg=self._c_panel)
        samp.pack(side=tk.TOP, fill=tk.X)
        self.sample_rate_var = tk.IntVar(value=100)
        self._sample_scale = tk.Scale(
            samp, from_=5, to=100, resolution=5, orient=tk.HORIZONTAL,
            variable=self.sample_rate_var, bg=self._c_panel, highlightthickness=0,
            length=150, showvalue=1, label=None,
            command=lambda _v: self._on_sample_rate_changed())
        self._sample_scale.pack(side=tk.LEFT)
        tk.Label(samp, text="%", bg=self._c_panel, fg=self._c_dim,
                 font=FONTS["small"]).pack(side=tk.LEFT)
        self._resample_btn = tk.Button(samp, text="Resample", font=FONTS["small"],
                                       state=tk.DISABLED, command=self._on_resample)
        self._resample_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Export reduced corpus + JSON…",
                  font=FONTS["buttons"], command=self._on_export).pack(
            side=tk.TOP, fill=tk.X, pady=(4, 2))

        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(
            side=tk.TOP, fill=tk.X, padx=PAD, pady=3)

        # --- ontology options ---------------------------------------- #
        tk.Label(outer, text="Ontology options", bg=self._c_panel,
                 fg=self._c_text, font=FONTS["panel_title"]).pack(
            side=tk.TOP, anchor=tk.W, padx=PAD)

        opt = tk.Frame(outer, bg=self._c_panel)
        opt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=PAD)

        self.name_mode_var = tk.StringVar(value="pool")
        self._radio_row(opt, "Name reduction:",
                        [("Preserve", "preserve"), ("Pool", "pool"),
                         ("Symbol", "symbol")],
                        self.name_mode_var, self._on_config_changed)

        # Per-category granularity + cast-size sliders.
        self.level_vars, self.pool_vars = {}, {}
        defaults = OntologyConfig()
        grid = tk.Frame(opt, bg=self._c_panel)
        grid.pack(side=tk.TOP, fill=tk.X, pady=(2, 2))
        tk.Label(grid, text="", bg=self._c_panel, font=FONTS["small"]).grid(
            row=0, column=0)
        tk.Label(grid, text="granularity", bg=self._c_panel, fg=self._c_dim,
                 font=FONTS["small"]).grid(row=0, column=1)
        tk.Label(grid, text="cast size", bg=self._c_panel, fg=self._c_dim,
                 font=FONTS["small"]).grid(row=0, column=2)
        for row, (key, label, pool_max) in enumerate(CATEGORY_CONTROLS, start=1):
            lvl = tk.IntVar(value=getattr(defaults, f"{key}_level"))
            pool = tk.IntVar(value=getattr(defaults, f"{key}_pool"))
            self.level_vars[key] = lvl
            self.pool_vars[key] = pool
            tk.Label(grid, text=label, bg=self._c_panel, fg=self._c_text,
                     font=FONTS["default"], anchor=tk.W, width=10).grid(
                row=row, column=0, sticky=tk.W)
            tk.Scale(grid, from_=0, to=2, orient=tk.HORIZONTAL, variable=lvl,
                     bg=self._c_panel, highlightthickness=0, length=90,
                     showvalue=1, command=lambda _v: self._on_config_changed()).grid(
                row=row, column=1, sticky=tk.W)
            tk.Scale(grid, from_=1, to=pool_max, orient=tk.HORIZONTAL,
                     variable=pool, bg=self._c_panel, highlightthickness=0,
                     length=110, showvalue=1,
                     command=lambda _v: self._on_config_changed()).grid(
                row=row, column=2, sticky=tk.W)

        # Toggles.
        self.collapse_other_var = tk.BooleanVar(value=defaults.collapse_other)
        self.lowercase_var = tk.BooleanVar(value=defaults.lowercase)
        self.lemmatize_var = tk.BooleanVar(value=defaults.lemmatize)
        self.synonym_var = tk.BooleanVar(value=defaults.synonym_merge)
        toggles = tk.Frame(opt, bg=self._c_panel)
        toggles.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        for text, var in [("Collapse other named entities", self.collapse_other_var),
                          ("Lowercase", self.lowercase_var),
                          ("Lemmatize", self.lemmatize_var),
                          ("Merge synonyms", self.synonym_var)]:
            tk.Checkbutton(toggles, text=text, variable=var, bg=self._c_panel,
                           fg=self._c_text, selectcolor=self._c_panel,
                           font=FONTS["default"], anchor=tk.W,
                           command=self._on_config_changed).pack(
                side=tk.TOP, anchor=tk.W)

        seed_row = tk.Frame(opt, bg=self._c_panel)
        seed_row.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        tk.Label(seed_row, text="Name seed:", bg=self._c_panel, fg=self._c_text,
                 font=FONTS["default"]).pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=defaults.seed)
        tk.Spinbox(seed_row, from_=0, to=999, textvariable=self.seed_var, width=5,
                   font=FONTS["default"], command=self._on_config_changed).pack(
            side=tk.LEFT, padx=4)

        # --- lemma / synonym-merge editor (fills remaining height) ------- #
        self._build_lemma_editor(opt)

    def _build_lemma_editor(self, parent):
        """Compact display+edit of the merge table, grouped one row per canonical,
        with corpus-hit counts, sorted by descending hits."""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(
            side=tk.TOP, fill=tk.X, pady=(5, 3))
        head = tk.Frame(parent, bg=self._c_panel)
        head.pack(side=tk.TOP, fill=tk.X)
        tk.Label(head, text="Lemma merges", bg=self._c_panel, fg=self._c_text,
                 font=FONTS["section"]).pack(side=tk.LEFT)
        self._rule_summary_lbl = tk.Label(head, text="", bg=self._c_panel,
                                          fg=self._c_dim, font=FONTS["small"])
        self._rule_summary_lbl.pack(side=tk.RIGHT)

        # Tree: canonical (#0) | merged words | corpus hits.
        try:
            style = ttk.Style(self.root)
            style.configure("Lemma.Treeview", background=self._c_view,
                            fieldbackground=self._c_view, foreground=self._c_text,
                            font=FONTS["small"], rowheight=18)
            style.configure("Lemma.Treeview.Heading", font=FONTS["small"])
        except Exception:
            pass
        tree_wrap = tk.Frame(parent, bg=self._c_panel)
        tree_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(2, 0))
        self._rule_tree = ttk.Treeview(
            tree_wrap, columns=("words", "hits"), show="tree headings",
            style="Lemma.Treeview", height=6, selectmode="browse")
        self._rule_tree.heading("#0", text="→ canonical")
        self._rule_tree.heading("words", text="merged words")
        self._rule_tree.heading("hits", text="hits")
        self._rule_tree.column("#0", width=90, minwidth=70, stretch=False,
                               anchor=tk.W)
        self._rule_tree.column("words", width=170, minwidth=90, stretch=True,
                               anchor=tk.W)
        self._rule_tree.column("hits", width=60, minwidth=45, stretch=False,
                               anchor=tk.E)
        vs = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL,
                           command=self._rule_tree.yview)
        self._rule_tree.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self._rule_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._rule_tree.bind("<<TreeviewSelect>>", self._on_rule_select)

        # Edit row: words -> canonical, plus Add/Update / Remove / Reset.
        edit = tk.Frame(parent, bg=self._c_panel)
        edit.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        self._rule_words_var = tk.StringVar()
        self._rule_canon_var = tk.StringVar()
        tk.Label(edit, text="words:", bg=self._c_panel, fg=self._c_text,
                 font=FONTS["small"]).pack(side=tk.LEFT)
        tk.Entry(edit, textvariable=self._rule_words_var, width=22,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(2, 4))
        tk.Label(edit, text="→", bg=self._c_panel, fg=self._c_text,
                 font=FONTS["small"]).pack(side=tk.LEFT)
        tk.Entry(edit, textvariable=self._rule_canon_var, width=9,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=(2, 0))
        btns = tk.Frame(parent, bg=self._c_panel)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(2, 4))
        tk.Button(btns, text="Add / update", font=FONTS["small"],
                  command=self._on_rule_apply).pack(side=tk.LEFT)
        tk.Button(btns, text="Remove", font=FONTS["small"],
                  command=self._on_rule_remove).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="New", font=FONTS["small"],
                  command=self._on_rule_new).pack(side=tk.LEFT)
        tk.Button(btns, text="Reset", font=FONTS["small"],
                  command=self._on_rule_reset).pack(side=tk.RIGHT)
        self._refresh_lemma_editor()

    # -- lemma-editor helpers -------------------------------------------- #
    def _group_rules(self):
        """{canonical: sorted[member words]} from the rule map (canonical incl.)."""
        groups = {}
        for word, canon in self.rules.items():
            grp = groups.setdefault(canon, {canon})
            grp.add(word)
        return {c: sorted(ws) for c, ws in groups.items()}

    def _selected_canon(self):
        sel = self._rule_tree.selection()
        return sel[0] if sel else None

    def _refresh_lemma_editor(self):
        """Rebuild the tree from self.rules + self._rule_counts, hits-descending."""
        tree = self._rule_tree
        keep = self._selected_canon()
        tree.delete(*tree.get_children())
        counts = self._rule_counts or {}
        rows = []
        for canon, members in self._group_rules().items():
            total = sum(counts.get(m, 0) for m in members)
            rows.append((total, canon, members))
        rows.sort(key=lambda r: (-r[0], r[1]))
        for total, canon, members in rows:
            others = [m for m in members if m != canon]
            shown = ", ".join([canon] + others)
            tree.insert("", tk.END, iid=canon, text=f"→ {canon}",
                        values=(shown, f"{total:,}"))
        if keep and tree.exists(keep):
            tree.selection_set(keep)
        self._rule_summary_lbl.config(
            text=f"{len(rows)} groups · {len(self.rules)} words")

    def _on_rule_select(self, _evt=None):
        canon = self._selected_canon()
        if not canon:
            return
        members = self._group_rules().get(canon, [canon])
        others = [m for m in members if m != canon]
        self._rule_words_var.set(" ".join(others))
        self._rule_canon_var.set(canon)

    def _on_rule_new(self):
        self._rule_tree.selection_remove(self._rule_tree.selection())
        self._rule_words_var.set("")
        self._rule_canon_var.set("")

    def _on_rule_apply(self):
        canon = self._rule_canon_var.get().strip().lower()
        raw = self._rule_words_var.get().replace(",", " ").split()
        words = [w.strip().lower() for w in raw if w.strip()]
        if not canon:
            messagebox.showinfo("Lemma merge", "Enter a canonical form (the "
                                "word all the others collapse to).")
            return
        old = self._selected_canon()          # replace the edited group, if any
        drop = {canon} | ({old} if old else set())
        self.rules = {w: c for w, c in self.rules.items()
                      if c not in drop and w not in drop}
        for w in set(words) | {canon}:
            if w != canon:
                self.rules[w] = canon
        self._on_rule_new()
        self._refresh_lemma_editor()
        self._on_config_changed()

    def _on_rule_remove(self):
        canon = self._selected_canon()
        if not canon:
            return
        self.rules = {w: c for w, c in self.rules.items()
                      if c != canon and w != canon}
        self._on_rule_new()
        self._refresh_lemma_editor()
        self._on_config_changed()

    def _on_rule_reset(self):
        self.rules = dict(ts.SYNONYMS)
        self._on_rule_new()
        self._refresh_lemma_editor()
        self._on_config_changed()

    def _radio_row(self, parent, caption, options, var, cmd):
        row = tk.Frame(parent, bg=self._c_panel)
        row.pack(side=tk.TOP, fill=tk.X, pady=1)
        tk.Label(row, text=caption, bg=self._c_panel, fg=self._c_text,
                 font=FONTS["section"], anchor=tk.W).pack(side=tk.TOP, anchor=tk.W)
        btns = tk.Frame(row, bg=self._c_panel)
        btns.pack(side=tk.TOP, anchor=tk.W)
        for text, val in options:
            tk.Radiobutton(btns, text=text, variable=var, value=val,
                           bg=self._c_panel, fg=self._c_text,
                           selectcolor=self._c_panel, font=FONTS["default"],
                           command=cmd).pack(side=tk.LEFT)

    def _build_bars(self, parent):
        self._bars_title = tk.Label(parent, text="", bg=self._c_view,
                                    fg=self._c_text, font=FONTS["section"])
        self._bars_title.pack(side=tk.TOP, anchor=tk.W, padx=PAD, pady=(2, 0))
        self._bars_canvas = tk.Canvas(parent, bg=self._c_view, highlightthickness=0)
        self._bars_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._bars_canvas.bind("<Configure>", lambda e: self._draw_bars())
        self._bars_canvas.bind("<Button-1>", self._on_bar_click)

    def _build_vocab(self, parent):
        self._vocab_title = tk.Label(parent, text="Reduced vocabulary",
                                    bg=self._c_panel, fg=self._c_text,
                                    font=FONTS["panel_title"])
        self._vocab_title.pack(side=tk.TOP, anchor=tk.W, padx=PAD, pady=(2, 0))
        vwrap = tk.Frame(parent, bg=self._c_panel)
        vwrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(PAD, 0))
        vsb = tk.Scrollbar(vwrap, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._vocab_text = tk.Text(vwrap, wrap=tk.NONE, bg=self._c_view,
                                   fg=self._c_text, font=FONTS["value"],
                                   yscrollcommand=vsb.set, borderwidth=0,
                                   highlightthickness=0, state=tk.DISABLED)
        self._vocab_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self._vocab_text.yview)

    def _build_stories(self, parent):
        self._stories_title = tk.Label(parent, text="Stories", bg=self._c_panel,
                                      fg=self._c_text, font=FONTS["panel_title"])
        self._stories_title.pack(side=tk.TOP, anchor=tk.W, padx=PAD, pady=(2, 0))
        swrap = tk.Frame(parent, bg=self._c_panel)
        swrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(PAD, 0))
        ssb = tk.Scrollbar(swrap, orient=tk.VERTICAL)
        ssb.pack(side=tk.RIGHT, fill=tk.Y)
        self._stories_text = tk.Text(swrap, wrap=tk.WORD, bg=self._c_view,
                                     fg=self._c_text, font=FONTS["default"],
                                     yscrollcommand=ssb.set, borderwidth=0,
                                     highlightthickness=0, state=tk.DISABLED)
        self._stories_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ssb.config(command=self._stories_text.yview)
        self._stories_text.tag_configure("hdr", foreground=self._c_dim,
                                         font=FONTS["small"])

    # ------------------------------------------------------------------ #
    #  Loading (background thread)
    # ------------------------------------------------------------------ #
    def _start_load(self):
        self._set_status("Loading / annotating corpus (first run parses ~22k "
                         "stories, then caches)…")

        def work():
            try:
                recs, meta = ts.load_or_annotate(
                    progress=lambda d, t: self._q.put(("progress", d, t)))
                natural = ts.natural_vocab_size(recs)
                n_sent = ts.count_sentences(recs)
                hist = ts.entity_count_histograms(recs)
                self._q.put(("loaded", recs, meta, natural, n_sent, hist))
            except Exception as exc:                       # surface to GUI
                self._q.put(("error", repr(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total = msg
                    self._set_status(f"Annotating {done}/{total} stories…")
                elif kind == "loaded":
                    _, recs, meta, natural, n_sent, hist = msg
                    self.records = recs
                    self.meta = meta
                    self.natural_vocab = natural
                    self.n_sentences = n_sent
                    self.histograms = hist
                    self._on_loaded()
                elif kind == "reduction":
                    _, gen, red, rule_counts = msg
                    if gen == self._gen:
                        self.reduction = red
                        self._rule_counts = rule_counts
                        self._on_reduction_ready()
                elif kind == "error":
                    self._set_status(f"ERROR: {msg[1]}")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _on_loaded(self):
        self.corpus = ts.Corpus(self.records)
        self._sample_scale.config(to=100)
        self._update_counts()
        self._draw_histograms()
        self._set_status(f"Annotated in {self.meta.get('annotate_seconds', '?')}s "
                         f"(cached).  Adjust ontology options to explore.")
        self._start_recompute()

    # ------------------------------------------------------------------ #
    #  Config -> reduction (background thread)
    # ------------------------------------------------------------------ #
    def _read_config(self):
        return OntologyConfig(
            unit=self.unit_var.get(),
            name_mode=self.name_mode_var.get(),
            char_level=self.level_vars["char"].get(),
            char_pool=self.pool_vars["char"].get(),
            place_level=self.level_vars["place"].get(),
            place_pool=self.pool_vars["place"].get(),
            animal_level=self.level_vars["animal"].get(),
            animal_pool=self.pool_vars["animal"].get(),
            object_level=self.level_vars["object"].get(),
            object_pool=self.pool_vars["object"].get(),
            collapse_other=self.collapse_other_var.get(),
            lowercase=self.lowercase_var.get(),
            lemmatize=self.lemmatize_var.get(),
            synonym_merge=self.synonym_var.get(),
            synonyms=dict(self.rules),
            seed=self.seed_var.get(),
            spacy_model=self.meta.get("spacy_model", "en_core_web_sm") if self.meta else "en_core_web_sm",
            spacy_version=self.meta.get("spacy_version", "") if self.meta else "",
        )

    def _on_config_changed(self, *_):
        if self.records is None:
            return
        if self._recompute_job is not None:
            self.root.after_cancel(self._recompute_job)
        self._recompute_job = self.root.after(300, self._start_recompute)

    def _on_unit_changed(self):
        self._on_config_changed()

    # -- sub-sampling ----------------------------------------------------- #
    def _on_sample_rate_changed(self):
        self._apply_active_set()

    def _on_resample(self):
        self._sample_seed += 1
        self._apply_active_set()

    def _apply_active_set(self):
        """Recompute the active story set from the rate + current seed, then reduce."""
        if self.records is None:
            return
        rate = self.sample_rate_var.get() / 100.0
        if rate >= 1.0:
            self._active_indices = None
            self._resample_btn.config(state=tk.DISABLED)
        else:
            self._active_indices = ts.sample_active_set(
                len(self.records), rate, self._sample_seed)
            self._resample_btn.config(state=tk.NORMAL)
        self._on_config_changed()

    def _on_framing_changed(self):
        # Framing does not change the reduction, only the bar view.
        if self.reduction is None:
            return
        if self._selection is not None:
            self._selection["framing"] = self.framing_var.get()
        self._draw_bars()
        self._reselect()

    def _start_recompute(self):
        self._recompute_job = None
        if self.records is None:
            return
        cfg = self._read_config()
        active = self._active_indices          # captured for the worker
        self._gen += 1
        gen = self._gen
        self._set_status("Computing reduction…")

        def work():
            try:
                red = self.corpus.reduce(cfg, active=active)
                rule_counts = self.corpus.rule_member_counts(cfg)
                self._q.put(("reduction", gen, red, rule_counts))
            except Exception as exc:
                self._q.put(("error", repr(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_reduction_ready(self):
        self._update_counts()
        m = self.reduction.metrics(self.natural_vocab)
        self._set_status(
            f"Reduced vocab {m['reduced_vocab']} / natural {m['natural_vocab']} "
            f"(ratio {m['reduction_ratio']:.3f}) · {m['n_samples']} "
            f"{self.reduction.config.unit}s · avg {m['avg_tokens_per_sample']} "
            f"tokens · {m['canonical_names_introduced']} names · "
            f"{m['unique_ontology_symbols']} symbols")
        if self._selection is None:            # default view: the 90% bar
            self._selection = {"framing": self.framing_var.get(),
                               "percent": 90, "k": None, "value": None}
        self._draw_bars()
        self._reselect()
        self._refresh_lemma_editor()

    # ------------------------------------------------------------------ #
    #  Status / counts
    # ------------------------------------------------------------------ #
    def _update_counts(self):
        if self.records is None:
            return
        red = self.reduction
        vocab_txt = f"   ·   reduced vocab: {red.V0:,}" if red else ""
        active_txt = ""
        if self._active_indices is not None:
            active_txt = (f"   ·   active: {len(self._active_indices):,} "
                          f"({self.sample_rate_var.get()}%)")
        self._counts_lbl.config(
            text=(f"stories: {len(self.records):,}   ·   sentences: "
                  f"{self.n_sentences:,}   ·   natural vocab: "
                  f"{self.natural_vocab:,}{vocab_txt}{active_txt}"))

    def _set_status(self, text):
        self._status_lbl.config(text=text)

    # ------------------------------------------------------------------ #
    #  Entity histograms
    # ------------------------------------------------------------------ #
    def _draw_histograms(self):
        cv = self._hist_canvas
        cv.delete("all")
        if self.histograms is None:
            return
        W = cv.winfo_width()
        H = cv.winfo_height()
        if W <= 1 or H <= 1:
            return
        groups = [("all named entities", "all"),
                  ("characters", "character"),
                  ("places", "place")]
        gw = W / 3.0
        for gi, (title, key) in enumerate(groups):
            x0 = gi * gw + 8
            self._draw_hist_group(cv, x0, 4, gw - 16, H - 8, title,
                                  self.histograms[key])

    def _draw_hist_group(self, cv, x, y, w, h, title, fracs):
        cv.create_text(x, y, text=title, anchor=tk.NW, fill=self._c_text,
                       font=FONTS["small"])
        top = y + 16
        n = len(fracs)
        bh = (h - 16) / n
        label_w = 22
        bar_max = max(10, w - label_w - 40)
        vmax = max(fracs) if any(fracs) else 1.0
        for i, fr in enumerate(fracs):
            by = top + i * bh
            cy = by + bh / 2
            cv.create_text(x + label_w - 4, cy, text=BUCKET_LABELS[i], anchor=tk.E,
                           fill=self._c_dim, font=FONTS["small"])
            bx = x + label_w
            length = (fr / vmax) * bar_max if vmax else 0
            cv.create_rectangle(bx, by + 2, bx + max(1, length), by + bh - 2,
                                fill=self._c_bar, outline="")
            cv.create_text(bx + length + 4, cy, text=f"{100 * fr:.0f}%",
                           anchor=tk.W, fill=self._c_text, font=FONTS["small"])

    # ------------------------------------------------------------------ #
    #  Bars
    # ------------------------------------------------------------------ #
    def _draw_bars(self):
        cv = self._bars_canvas
        cv.delete("all")
        self._bars = []
        if self.reduction is None:
            return
        framing = self.framing_var.get()
        unit = self.reduction.config.unit
        bars = self.reduction.bars(framing)
        if framing == "sample":
            self._bars_title.config(
                text=f"Vocabulary size required to keep X% of {unit}s "
                     f"(cheapest-vocabulary-first)")
            y_max = max(self.reduction.V0, 1)
        else:
            self._bars_title.config(
                text=f"{unit.capitalize()}s usable when vocabulary is clamped "
                     f"to X% of the natural reduced size")
            y_max = max(self.reduction.N, 1)

        W = cv.winfo_width()
        H = cv.winfo_height()
        if W <= 1 or H <= 1:
            return
        left, right, top, bottom = 10, 10, 14, 34
        plot_w = W - left - right
        plot_h = H - top - bottom
        n = len(bars)
        slot = plot_w / n
        bw = slot * 0.62
        sel_pct = self._selection["percent"] if self._selection else None
        sel_framing = self._selection["framing"] if self._selection else None

        # baseline
        cv.create_line(left, top + plot_h, W - right, top + plot_h,
                       fill=self._c_line)
        for i, bar in enumerate(bars):
            cx = left + slot * (i + 0.5)
            frac = bar["value"] / y_max if y_max else 0
            bh = max(1, frac * plot_h)
            x0, x1 = cx - bw / 2, cx + bw / 2
            y1 = top + plot_h
            y0 = y1 - bh
            selected = (sel_framing == framing and sel_pct == bar["percent"])
            color = self._c_sel if selected else self._c_bar
            rect = cv.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            self._bars.append((x0, x1, top, y1, bar))
            cv.create_text(cx, y0 - 2, text=f"{bar['value']:,}", anchor=tk.S,
                           fill=self._c_text, font=FONTS["small"])
            cv.create_text(cx, y1 + 4, text=f"{bar['percent']}%", anchor=tk.N,
                           fill=self._c_text, font=FONTS["small"])

    def _on_bar_click(self, event):
        for x0, x1, y0, y1, bar in self._bars:
            if x0 <= event.x <= x1 and y0 - 12 <= event.y <= y1 + 12:
                self._selection = {"framing": self.framing_var.get(),
                                   "percent": bar["percent"], "k": bar["k"],
                                   "value": bar["value"]}
                self._draw_bars()
                self._fill_bottom(bar)
                return

    def _reselect(self):
        """Re-apply the current selection (same percent) after a recompute."""
        if self.reduction is None or self._selection is None:
            return
        framing = self.framing_var.get()
        for bar in self.reduction.bars(framing):
            if bar["percent"] == self._selection["percent"]:
                self._selection = {"framing": framing, "percent": bar["percent"],
                                   "k": bar["k"], "value": bar["value"]}
                self._fill_bottom(bar)
                return

    # ------------------------------------------------------------------ #
    #  Bottom panels
    # ------------------------------------------------------------------ #
    def _fill_bottom(self, bar):
        vocab, admitted = self.reduction.select(bar["k"])
        self._fill_vocab(vocab, bar)
        self._fill_stories(admitted, bar)

    def _fill_vocab(self, vocab, bar):
        self._vocab_title.config(
            text=f"Reduced vocabulary — top {len(vocab):,} tokens (budget k={bar['k']:,})")
        shown = vocab[:MAX_VOCAB_SHOWN]
        half = math.ceil(len(shown) / 2) if shown else 0
        counts = self.reduction.counts
        lines = []
        for i in range(half):
            a = shown[i]
            cell_a = f"{i + 1:>4} {a[:14]:<14} {counts[a]:>6}"
            j = i + half
            if j < len(shown):
                b = shown[j]
                cell_b = f"{j + 1:>4} {b[:14]:<14} {counts[b]:>6}"
            else:
                cell_b = ""
            lines.append(f"{cell_a}   {cell_b}")
        if len(vocab) > MAX_VOCAB_SHOWN:
            lines.append(f"\n… {len(vocab) - MAX_VOCAB_SHOWN:,} more tokens not shown")
        self._set_text(self._vocab_text, "\n".join(lines))

    def _fill_stories(self, admitted, bar):
        unit = self.reduction.config.unit
        self._stories_title.config(
            text=f"Admitted {unit}s — {len(admitted):,} of {self.reduction.N:,}")
        self._stories_text.config(state=tk.NORMAL)
        self._stories_text.delete("1.0", tk.END)
        syn = ts.effective_synonyms(self.reduction.config)
        rec_by_idx = {r["index"]: r for r in self.records}
        render_cache = {}
        for ref in admitted[:MAX_STORIES_SHOWN]:
            if ref[0] == "story":
                idx = ref[1]
                rec = rec_by_idx[idx]
                text = render_cache.get(idx)
                if text is None:
                    text = ts.render_story(rec, self.reduction.config, syn)
                    render_cache[idx] = text
                self._stories_text.insert(tk.END, f"— story {idx} —\n", "hdr")
                self._stories_text.insert(tk.END, text + "\n\n")
            else:
                _kind, idx, sid = ref
                rec = rec_by_idx[idx]
                lines = render_cache.get(idx)
                if lines is None:
                    lines = ts.render_sentences(rec, self.reduction.config, syn)
                    render_cache[idx] = lines
                sent = lines[sid] if sid < len(lines) else ""
                self._stories_text.insert(tk.END, f"— story {idx} · sentence {sid} —\n",
                                          "hdr")
                self._stories_text.insert(tk.END, sent + "\n\n")
        if len(admitted) > MAX_STORIES_SHOWN:
            self._stories_text.insert(
                tk.END, f"… {len(admitted) - MAX_STORIES_SHOWN:,} more not shown\n",
                "hdr")
        self._stories_text.config(state=tk.DISABLED)

    def _set_text(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    #  Export
    # ------------------------------------------------------------------ #
    def _on_export(self):
        if self.reduction is None:
            messagebox.showinfo("Export", "Nothing to export yet — still computing.")
            return
        corpus_path = filedialog.asksaveasfilename(
            title="Export reduced corpus", defaultextension=".txt",
            initialfile="reduced_corpus.txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not corpus_path:
            return
        base, _ = os.path.splitext(corpus_path)
        json_path = base + "_bundle.json"

        k = self._selection["k"] if self._selection else None
        selection = dict(self._selection) if self._selection else None
        sampling = {
            "rate_percent": self.sample_rate_var.get(),
            "seed": self._sample_seed,
            "active_count": len(self.reduction.active),
            # Record indices only when actually sub-sampled (else it's all stories).
            "active_indices": (list(self.reduction.active)
                               if self._active_indices is not None else None),
        }
        try:
            written = ts.export_corpus(corpus_path, self.records,
                                       self.reduction.config, self.reduction, k=k)
            ts.export_bundle_json(json_path, self.reduction.config, self.reduction,
                                  self.natural_vocab, self.meta,
                                  written_counts=written, selection=selection,
                                  sampling=sampling)
            ok, detail = ts.verify_reproducible(corpus_path, json_path)
        except Exception as exc:
            messagebox.showerror("Export failed", repr(exc))
            return
        filt = "" if k is None else f"\nFiltered to vocab budget k={k:,}."
        sub = ("" if self._active_indices is None
               else f"\nActive sub-sample: {len(self.reduction.active):,} stories "
                    f"({self.sample_rate_var.get()}%).")
        messagebox.showinfo(
            "Export complete",
            f"Corpus: {corpus_path}\nBundle: {json_path}\n\n"
            f"Vocabulary: {len(written):,} tokens.{filt}{sub}\n\n"
            f"Reproducibility check: {'PASS' if ok else 'FAIL'}\n{detail}")

    def start(self):
        self.root.mainloop()


def main():
    if not os.path.exists(ts.CORPUS_PATH):
        raise SystemExit(f"Corpus not found: {ts.CORPUS_PATH}")
    ReductionExplorer().start()
    # Hard-exit once the window closes: the bundled BLIS otherwise aborts
    # ("libblis: Aborting.") during normal interpreter teardown on this box.
    os._exit(0)


def _selftest(n_stories=80):
    """Env-gated smoke test (NRE_SELFTEST=1): build the UI on a small slice,
    exercise every draw/fill path off-screen, then exit -- no mainloop."""
    stories = ts.load_corpus()[:n_stories]
    recs = ts.annotate(stories)
    app = ReductionExplorer()                 # after()-scheduled load never fires
    app.root.geometry("1300x850+2000+2000")   # off to the side
    app.root.update()
    app._set_initial_sashes()
    app.records = recs
    app.corpus = ts.Corpus(recs)
    app.natural_vocab = ts.natural_vocab_size(recs)
    app.n_sentences = ts.count_sentences(recs)
    app.histograms = ts.entity_count_histograms(recs)
    app.reduction = app.corpus.reduce(app._read_config())
    app._update_counts()
    app._draw_histograms()
    app._draw_bars()
    app.root.update()
    assert len(app._bars) == 7, app._bars
    for framing in ("sample", "vocab"):
        app.framing_var.set(framing)
        app._draw_bars()
        app.root.update()
        bar = app.reduction.bars(framing)[3]
        app._selection = {"framing": framing, "percent": bar["percent"],
                          "k": bar["k"], "value": bar["value"]}
        app._fill_bottom(bar)
        app.root.update()
        print(f"[selftest] framing={framing} bar75%={bar['value']} "
              f"vocab_shown+stories filled")

    # Sub-sampling path: 50% active set, entity-only recompute reuses the cache.
    app.sample_rate_var.set(50)
    app._active_indices = ts.sample_active_set(len(recs), 0.5, app._sample_seed)
    sig = app.corpus._lex_sig
    app.reduction = app.corpus.reduce(
        app._read_config(), active=app._active_indices)
    assert app.corpus._lex_sig == sig, "sub-sample rebuilt lexical cache"
    app._update_counts()
    app._draw_bars()
    app.root.update()
    print(f"[selftest] subsample active={len(app.reduction.active)} "
          f"samples={app.reduction.N} vocab={app.reduction.V0}")

    # Lemma editor: counts populate, add a rule, edit it, remove it.
    app._rule_counts = app.corpus.rule_member_counts(app._read_config())
    app._refresh_lemma_editor()
    app.root.update()
    n_groups0 = len(app._rule_tree.get_children())
    app._rule_words_var.set("big large huge")
    app._rule_canon_var.set("big")
    app._on_rule_apply()
    app.root.update()
    assert app.rules.get("large") == "big" and app.rules.get("huge") == "big"
    app._rule_tree.selection_set("big")
    app._on_rule_remove()
    app.root.update()
    assert "large" not in app.rules and "big" not in app.rules.values()
    app._on_rule_reset()
    app.root.update()
    assert len(app._rule_tree.get_children()) == n_groups0
    print(f"[selftest] lemma editor groups={n_groups0} add/edit/remove/reset OK")

    app.root.destroy()
    print("[selftest] OK")
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    if os.environ.get("NRE_SELFTEST"):
        _selftest()
    else:
        main()
