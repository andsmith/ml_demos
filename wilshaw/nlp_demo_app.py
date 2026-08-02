"""
Stand-alone confabulation demo app (see ``nl_demo_tab.md``): visually
steps through high-level word-prediction confabulation using the
knowledge bases in ``knowledge_bases/`` (``extract_tiny_kbs.sh``).

Window layout::

  +--------+--------------------------------------------+
  | Title/ |  control bar (Step, D slider, mode toggle)  |
  | Status +--------------------------------------------+
  +--------+                                            |
  | corpus |   tabbed demo display                      |
  | list   |   [ Demo 1 ] [ Demo 2 ] [ Demo 3 ]          |
  | (click |                                            |
  |  to    |                                            |
  |  load) |                                            |
  +--------+                                            |
  | manual |                                            |
  | entry  |                                            |
  +--------+--------------------------------------------+

Run with::  python nlp_demo_app.py

This is intentionally a separate top-level app for now (own ``tk.Tk()`` /
``mainloop()``) rather than a tab nested in ``willshaw_app.py`` -- the
spec calls out embedding as a "(future)" step.
"""

import logging
import math
import tkinter as tk
import tkinter.ttk as ttk

import numpy as np

import confabulation_high_level as chl
import confabulation_stepper as stepper
from gui_base import tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS
from nlp_demo_sidebar import NLPDemoSidebar
from nlp_demo_tabs import Demo1Tab, Demo2Tab, Demo3Tab

WIN_SIZE = (1500, 900)
LAYOUT = {
    'sidebar': {'x_rel': (0.0, 0.22), 'y_rel': (0.0, 1.0)},
    'main':    {'x_rel': (0.22, 1.0), 'y_rel': (0.0, 1.0)},
    'margin_rel': 0.004,
}

D_MIN, D_MAX, D_DEFAULT = 100, 5000, 1000
POS_MAX = 1000          # slider position resolution for the log-scaled D slider
D_COMMIT_MS = 150        # debounce before rebuilding word codes on D drag


def _log_map(pos, vmin, vmax):
    """Map an integer slider position (0..POS_MAX) to a log-spaced value."""
    f = pos / POS_MAX
    return vmin * (vmax / vmin) ** f


def _log_unmap(val, vmin, vmax):
    """Inverse of :func:`_log_map` -> nearest integer slider position."""
    val = min(max(val, vmin), vmax)
    return int(round(POS_MAX * math.log(val / vmin) / math.log(vmax / vmin)))


class NLPDemoApp(object):
    """Top-level application object."""

    def __init__(self):
        self.mode = 'absolute'
        self.D = D_DEFAULT
        self._rng = np.random.default_rng(0)
        self._d_commit_job = None

        logging.info("Loading knowledge bases...")
        self.kb_sets = {
            'absolute': chl.load_kb_set('absolute'),
            'relative': chl.load_kb_set('relative'),
        }
        self.vocab = next(iter(self.kb_sets['absolute'].values()))['vocab']
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)}
        self.word_codes, self._S = stepper.build_word_codes(self.vocab, self.D, self._rng)
        logging.info("Loaded %d vocab tokens, %d absolute KBs, %d relative KBs",
                     len(self.vocab), len(self.kb_sets['absolute']), len(self.kb_sets['relative']))

        self._init_tk()
        self._init_sidebar()
        self._init_main()

        self.root.after(150, lambda: self._active_tab().on_show())

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _init_tk(self):
        self.root = tk.Tk()
        self.root.title("Confabulation Demo")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")
        self.root.configure(bg=tk_color_from_rgb(CS['bg']))
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _init_sidebar(self):
        self.sidebar = NLPDemoSidebar(self, LAYOUT['sidebar'], margin_rel=LAYOUT['margin_rel'])
        self.sidebar.set_status(
            f"corpus: {len(self.sidebar._sentences)} sentences, {len(self.vocab)} vocab\n"
            f"KBs: {len(self.kb_sets['absolute'])} absolute, "
            f"{len(self.kb_sets['relative'])} relative")

    def _init_main(self):
        bb = LAYOUT['main']
        main = tk.Frame(self.root, bg=tk_color_from_rgb(CS['bg']))
        main.place(relx=bb['x_rel'][0], rely=bb['y_rel'][0],
                   relwidth=bb['x_rel'][1] - bb['x_rel'][0],
                   relheight=bb['y_rel'][1] - bb['y_rel'][0])

        self._init_control_bar(main)

        style = ttk.Style(main)
        style.configure('TNotebook.Tab', font=FONTS['tabs'], padding=(18, 8))
        self._notebook = ttk.Notebook(main, style='TNotebook')
        self._notebook.pack(fill=tk.BOTH, expand=True)

        d1 = tk.Frame(self._notebook, bg=tk_color_from_rgb(CS['panel_bg']))
        d2 = tk.Frame(self._notebook, bg=tk_color_from_rgb(CS['panel_bg']))
        d3 = tk.Frame(self._notebook, bg=tk_color_from_rgb(CS['panel_bg']))
        self._notebook.add(d1, text="  Demo 1: Infer word 6  ")
        self._notebook.add(d2, text="  Demo 2: Infer words 5+6 jointly  ")
        self._notebook.add(d3, text="  Demo 3: Sliding window  ")

        self._tabs = [Demo1Tab(d1, self), Demo2Tab(d2, self), Demo3Tab(d3, self)]
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _init_control_bar(self, parent):
        bg = tk_color_from_rgb(CS['panel_bg'])
        bar = tk.Frame(parent, bg=bg)
        bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        self._step_btn = tk.Button(bar, text="Step", font=('Helvetica', 14, 'bold'),
                                   bg=tk_color_from_rgb(CS['good']), fg='white',
                                   activebackground=tk_color_from_rgb(CS['good']),
                                   command=self.on_step, width=8, height=2)
        self._step_btn.pack(side=tk.LEFT, padx=(6, 12), pady=4)

        d_col = tk.Frame(bar, bg=bg)
        d_col.pack(side=tk.LEFT, padx=8)
        tk.Label(d_col, text="D", bg=bg, fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['section']).pack(side=tk.TOP)
        self._d_scale = tk.Scale(d_col, from_=0, to=POS_MAX, orient=tk.HORIZONTAL,
                                 showvalue=0, length=140, bg=bg, highlightthickness=0,
                                 command=self._on_d_scale)
        self._d_scale.set(_log_unmap(self.D, D_MIN, D_MAX))
        self._d_scale.pack(side=tk.TOP)
        self._d_val = tk.Label(d_col, text=str(self.D), bg=bg,
                               fg=tk_color_from_rgb(CS['text']), font=FONTS['small'])
        self._d_val.pack(side=tk.TOP)

        mode_col = tk.Frame(bar, bg=bg)
        mode_col.pack(side=tk.LEFT, padx=12)
        self._mode_var = tk.StringVar(value=self.mode)
        for txt, val in (("Absolute KBs", 'absolute'), ("Relative KBs", 'relative')):
            tk.Radiobutton(mode_col, text=txt, variable=self._mode_var, value=val,
                           bg=bg, fg=tk_color_from_rgb(CS['text']), selectcolor=bg,
                           font=FONTS['default'], command=self._on_mode).pack(side=tk.TOP, anchor=tk.W)

        self._action_lbl = tk.Label(bar, text="", bg=bg, fg=tk_color_from_rgb(CS['text']),
                                    font=FONTS['default'], wraplength=600, justify=tk.LEFT,
                                    anchor=tk.W)
        self._action_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)

    # ------------------------------------------------------------------ #
    #  Shared API used by the sidebar / tabs
    # ------------------------------------------------------------------ #
    def load_words(self, words):
        self._active_tab().on_load_words(words)

    def set_action_text(self, text):
        self._action_lbl.config(text=text)

    def on_step(self):
        self._active_tab().on_step()

    def _active_tab(self):
        # Query the notebook directly rather than caching the index --
        # <<NotebookTabChanged>> is dispatched asynchronously (see
        # willshaw-app-gotchas), so a cached index can be stale for calls
        # made right after .select().
        return self._tabs[self._notebook.index(self._notebook.select())]

    def _on_tab_changed(self, _event):
        self._active_tab().on_show()

    # ------------------------------------------------------------------ #
    #  D slider / mode toggle
    # ------------------------------------------------------------------ #
    def _on_d_scale(self, _=None):
        self.D = int(round(_log_map(self._d_scale.get(), D_MIN, D_MAX)))
        self._d_val.config(text=str(self.D))
        if self._d_commit_job is not None:
            self.root.after_cancel(self._d_commit_job)
        self._d_commit_job = self.root.after(D_COMMIT_MS, self._commit_d)

    def _commit_d(self):
        self._d_commit_job = None
        self.word_codes, self._S = stepper.build_word_codes(self.vocab, self.D, self._rng)
        for tab in self._tabs:
            tab.on_d_changed()

    def _on_mode(self):
        self.mode = self._mode_var.get()
        for tab in self._tabs:
            tab.on_mode_changed()

    def start(self):
        self.root.mainloop()


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _self_test(app):
    """Env-gated smoke test (NLP_DEMO_SELFTEST=1): step every demo tab end
    to end on a real corpus sentence, then close."""
    sentence = app.sidebar._sentences[0]

    def step_until_done(tab, on_done, guard=None):
        guard = [0] if guard is None else guard
        if tab.is_done() or guard[0] > 200:
            logging.info("SELFTEST %s done (guard=%d)", type(tab).__name__, guard[0])
            on_done()
            return
        tab.on_step()
        guard[0] += 1
        app.root.after(1, lambda: step_until_done(tab, on_done, guard))

    def run_demo1():
        app._notebook.select(0)
        app.load_words(sentence)
        step_until_done(app._tabs[0], run_demo2)

    def run_demo2():
        app._notebook.select(1)
        app.load_words(sentence)
        step_until_done(app._tabs[1], run_demo3)

    def run_demo3():
        app._notebook.select(2)
        app._tabs[2]._max_steps_var.set(3)
        app.load_words(sentence)
        step_until_done(app._tabs[2], finish)

    def finish():
        logging.info("SELFTEST OK")
        app.root.after(300, app.root.destroy)

    app.root.after(300, run_demo1)


def main():
    import os
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    app = NLPDemoApp()
    if os.environ.get('NLP_DEMO_SELFTEST'):
        _self_test(app)
    app.start()


if __name__ == '__main__':
    main()
