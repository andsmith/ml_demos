"""
The three confabulation demo tabs (see ``nl_demo_tab.md``), each a thin Tk
wrapper around ``confabulation_stepper.py``'s pure-logic engines and
``module_row_art.py``'s pure rendering -- mirrors ``demo_tab.py``'s role
in the Willshaw app (interaction glue, no math/drawing of its own beyond
compositing).

All three tabs share ``app``'s state: ``app.kb_sets`` (``{'absolute':
kb_set, 'relative': kb_set}``), ``app.mode`` (current toggle), ``app.vocab``
/ ``app.vocab_index``, ``app.D`` / ``app.word_codes`` (cosmetic sparse
codes, rebuilt by the app when the D slider commits), and call back into
``app.set_action_text(text)`` after every redraw so the control bar's
"what will Step do" label stays in sync.
"""

import tkinter as tk

import confabulation_stepper as stepper
import draw_utils as du
import module_row_art as mra
from gui_base import blit, tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS


class _LockedUnit(object):
    """A known (already-locked) word, duck-typed like a converged UnitState."""
    __slots__ = ('label', 'locked_word', 'candidates')

    def __init__(self, label, word):
        self.label = label
        self.locked_word = word
        self.candidates = None


def _display_units(known_words, mode, engine_units):
    """Known words (as locked pseudo-units) + the live inferring unit(s),
    left to right, ready for ``module_row_art.draw_demo_row``."""
    if mode == 'absolute':
        labels = list(range(1, len(known_words) + 1))
    else:
        labels = list(known_words)   # the word itself doubles as its label
    locked = [_LockedUnit(lbl, w) for lbl, w in zip(labels, known_words)]
    return locked + list(engine_units)


# --------------------------------------------------------------------------- #
#  Shared base
# --------------------------------------------------------------------------- #
class _BaseDemoTab(object):
    n_known = 5
    target_pos = 6

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self._last_words = None
        self.engine = None
        self._build_widgets()

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _build_widgets(self):
        bg = tk_color_from_rgb(CS['panel_bg'])
        self.parent.configure(bg=bg)
        self._build_extra_controls(self.parent)

        frame = tk.Frame(self.parent, bg=tk_color_from_rgb(CS['view_bg']),
                          highlightthickness=1,
                          highlightbackground=tk_color_from_rgb(CS['lines']))
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        frame.pack_propagate(False)
        self._label = tk.Label(frame, bg=tk_color_from_rgb(CS['view_bg']))
        self._label.pack(fill=tk.BOTH, expand=True)
        frame.bind("<Configure>", lambda e: self._redraw())
        self._frame = frame

    def _build_extra_controls(self, parent):
        """Hook for a subclass to add widgets above the drawing area."""
        pass

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks the app calls
    # ------------------------------------------------------------------ #
    def on_load_words(self, words):
        self._last_words = list(words)
        self._rebuild_engine()
        self._redraw()

    def on_d_changed(self):
        """D only affects cosmetic word codes (app rebuilds those); redraw
        with the current state. Per the spec, a fresh D also restarts the
        current demo's step progress -- reload the same words."""
        if self._last_words:
            self._rebuild_engine()
        self._redraw()

    def on_mode_changed(self):
        if self._last_words:
            self._rebuild_engine()
        self._redraw()

    def on_step(self):
        if self.engine is None:
            return
        self.engine.step()
        self._redraw()

    def next_action_text(self):
        if self.engine is None:
            return "Click a sentence on the left, or enter words manually, then Step."
        return self.engine.next_action()[0]

    def is_done(self):
        return self.engine is not None and self.engine.is_done

    def on_show(self):
        self._redraw()

    # ------------------------------------------------------------------ #
    #  Subclass hooks
    # ------------------------------------------------------------------ #
    def _rebuild_engine(self):
        raise NotImplementedError

    def _display_units(self):
        raise NotImplementedError

    def _current_highlight_links(self):
        if self.engine is None:
            return []
        return self.engine.arch_links()

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #
    def _redraw(self):
        w, h = self._frame.winfo_width(), self._frame.winfo_height()
        if w <= 1 or h <= 1:
            return
        units = self._display_units()
        if not units:
            img = du.blank(w, h, color=CS['view_bg'])
            du.put_label(img, "Click a corpus sentence, or enter words manually.",
                         (14, h // 2), 0.5, CS['text_dim'])
        else:
            img, x_centers = mra.draw_demo_row(w, h, units, self.app.word_codes,
                                               self.app.vocab_index, self.app.D)
            links = self._current_highlight_links()
            labels = [u.label for u in units]
            mra.draw_kb_arches(img, labels, x_centers, links)
        blit(self._label, img)
        if hasattr(self.app, 'set_action_text'):
            self.app.set_action_text(self.next_action_text())


# --------------------------------------------------------------------------- #
#  Demo 1 -- infer the 6th word from the first 5
# --------------------------------------------------------------------------- #
class Demo1Tab(_BaseDemoTab):
    n_known = 5
    target_pos = 6

    def _rebuild_engine(self):
        mode = self.app.mode
        words = self._last_words[:self.n_known]
        self.engine = stepper.single_engine(
            words, mode, self.app.kb_sets[mode], self.app.vocab, self.app.vocab_index,
            target_pos=self.target_pos)

    def _display_units(self):
        if self.engine is None:
            return []
        return _display_units(self._last_words[:self.n_known], self.app.mode, self.engine.units)


# --------------------------------------------------------------------------- #
#  Demo 2 -- jointly infer the 5th and 6th words together
# --------------------------------------------------------------------------- #
class Demo2Tab(_BaseDemoTab):
    n_known = 4
    target_pos = 6

    def _rebuild_engine(self):
        mode = self.app.mode
        words = self._last_words[:self.n_known]
        self.engine = stepper.joint_engine(
            words, mode, self.app.kb_sets[mode], self.app.vocab, self.app.vocab_index,
            target_pos=self.target_pos)

    def _display_units(self):
        if self.engine is None:
            return []
        return _display_units(self._last_words[:self.n_known], self.app.mode, self.engine.units)


# --------------------------------------------------------------------------- #
#  Demo 3 -- sliding window, generate up to a step cap
# --------------------------------------------------------------------------- #
class Demo3Tab(_BaseDemoTab):
    n_known = 5
    target_pos = 6
    default_max_steps = 12

    def __init__(self, parent, app):
        self.runner = None
        self._max_steps_var = None
        self._progress_lbl = None
        super().__init__(parent, app)

    def _build_extra_controls(self, parent):
        bg = tk_color_from_rgb(CS['panel_bg'])
        row = tk.Frame(parent, bg=bg)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        tk.Label(row, text="Max words to generate:", bg=bg,
                 fg=tk_color_from_rgb(CS['text']), font=FONTS['default']).pack(side=tk.LEFT)
        self._max_steps_var = tk.IntVar(value=self.default_max_steps)
        tk.Spinbox(row, from_=1, to=60, width=4, textvariable=self._max_steps_var,
                   font=FONTS['default']).pack(side=tk.LEFT, padx=(4, 12))
        tk.Button(row, text="Stop", font=FONTS['buttons'],
                  command=self._on_stop).pack(side=tk.LEFT)
        self._progress_lbl = tk.Label(row, text="", bg=bg,
                                       fg=tk_color_from_rgb(CS['text_dim']),
                                       font=FONTS['small'])
        self._progress_lbl.pack(side=tk.LEFT, padx=12)

    def _on_stop(self):
        if self.runner is not None and not self.runner.done:
            self.runner.max_steps = self.runner.n_generated
            self._redraw()

    def _rebuild_engine(self):
        mode = self.app.mode
        words = self._last_words[:self.n_known]
        self.runner = stepper.SlidingRunner(
            words, mode, self.app.kb_sets[mode], self.app.vocab, self.app.vocab_index,
            max_steps=self._max_steps_var.get(), target_pos=self.target_pos)

    def on_step(self):
        if self.runner is None:
            return
        self.runner.step()
        self._redraw()

    def next_action_text(self):
        if self.runner is None:
            return "Click a sentence on the left, or enter words manually, then Step."
        return self.runner.next_action()[0]

    def is_done(self):
        return self.runner is not None and self.runner.done

    def _current_highlight_links(self):
        if self.runner is None or self.runner.done:
            return []
        return self.runner.engine.arch_links()

    def _display_units(self):
        if self.runner is None:
            return []
        if self._progress_lbl is not None:
            self._progress_lbl.config(
                text=f"generated {self.runner.n_generated}/{self.runner.max_steps}: "
                     f"{' '.join(self.runner.history)}")
        return _display_units(self.runner.window, self.app.mode, self.runner.engine.units)
