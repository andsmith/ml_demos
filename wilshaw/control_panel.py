"""
The common left-hand control panel: vertical sliders for Dimensionality (D),
number of Patterns (V) and Sparsity (S), plus a button for the theoretical
optimal S.  Thresholding is always Top-S (winner-take-all) since S is always
known -- there is no user-facing threshold control.

D and V are log-scaled.  V's range tracks D (1 .. 10D, so a vocabulary of a
single symbol is always reachable); S's unit count tracks D.  Moving a slider
updates its readout immediately and schedules a
single debounced ``app.on_params_changed(source)`` callback so the owning tab
can react (regenerate the demo, or stop/clear the simulation).

The panel keeps a live ``self.params`` dict:  {'D','V','S','theta','mode'} --
``theta``/``mode`` are constants (``0``/``'wta'``) kept only because the
Simulation tab's work items and both tabs' param-signature helpers still key
off them.
"""

import logging
import math
import tkinter as tk

from gui_base import Panel, tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS
import willshaw_core as core

POS_MAX = 1000                     # slider position resolution for log sliders
S_FRAC_MIN = 0.0001                # 0.01 %
S_FRAC_MAX = 0.5                   # 50 %
V_MIN = 1                          # minimum vocabulary size (all demos need this reachable)
COMMIT_MS = 200                    # debounce delay before firing on_params_changed


def _log_map(pos, vmin, vmax):
    """Map an integer slider position (0..POS_MAX) to a log-spaced value."""
    f = pos / POS_MAX
    return vmin * (vmax / vmin) ** f


def _log_unmap(val, vmin, vmax):
    """Inverse of :func:`_log_map` -> nearest integer slider position."""
    val = min(max(val, vmin), vmax)
    return int(round(POS_MAX * math.log(val / vmin) / math.log(vmax / vmin)))


class ControlPanel(Panel):
    """Left parameter panel shared by both tabs."""

    def __init__(self, app, bbox_rel, margin_rel=0.0):
        # Tab-dependent D range; default to the demonstration range.
        self._d_min, self._d_max = 10, 100
        self._commit_job = None
        self._suppress = False        # when True, slider moves don't commit
        super().__init__(app, bbox_rel, margin_rel)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _init_widgets(self):
        bg = self._color_bg
        tk.Label(self._frame, text="Parameters", bg=bg, fg=self._color_text,
                 font=FONTS['panel_title']).pack(side=tk.TOP, pady=(6, 2))

        sliders = tk.Frame(self._frame, bg=bg)
        sliders.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2)

        # Four vertical sliders side by side.  D and V log-scaled; S log-scaled
        # over fraction; theta linear integer.
        self._d_scale, self._d_val = self._make_slider(
            sliders, "D", POS_MAX, self._on_d)
        self._v_scale, self._v_val = self._make_slider(
            sliders, "V", POS_MAX, self._on_v)
        self._s_scale, self._s_val = self._make_slider(
            sliders, "S", POS_MAX, self._on_s)

        # Initial values: mid-range D, V ~= D, S ~= optimal.
        self._d_scale.set(_log_unmap(40, self._d_min, self._d_max))
        d = self._current_D()
        self._v_scale.set(_log_unmap(d, V_MIN, 10 * d))
        s_frac = core.optimal_sparsity(d) / d
        self._s_scale.set(_log_unmap(s_frac, S_FRAC_MIN, S_FRAC_MAX))

        # Optimal-S button.
        btns = tk.Frame(self._frame, bg=bg)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(4, 2))
        tk.Button(btns, text="Optimal S", font=FONTS['buttons'],
                  command=self._set_optimal_s).pack(side=tk.TOP, fill=tk.X, padx=6, pady=1)

        # Derived-quantity readout.
        self._info = tk.Label(self._frame, text="", bg=bg, fg=CS['text_dim'] and
                              tk_color_from_rgb(CS['text_dim']),
                              font=FONTS['small'], justify=tk.LEFT, anchor=tk.W)
        self._info.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 6))

        # Establish dependent ranges/labels.
        self._recompute_dependent()

    def _make_slider(self, parent, caption, to_pos, cb):
        """Build one captioned vertical slider; return (scale, value_label)."""
        bg = self._color_bg
        col = tk.Frame(parent, bg=bg)
        col.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=1)
        tk.Label(col, text=caption, bg=bg, fg=self._color_text,
                 font=FONTS['section']).pack(side=tk.TOP)
        scale = tk.Scale(col, from_=to_pos, to=0, orient=tk.VERTICAL,
                         showvalue=0, resolution=1, command=cb,
                         bg=bg, highlightthickness=0, troughcolor='#cfd2d9',
                         length=300, width=16, sliderlength=18)
        scale.pack(side=tk.TOP, fill=tk.Y, expand=True)
        val = tk.Label(col, text="", bg=bg, fg=self._color_text,
                       font=FONTS['small'])
        val.pack(side=tk.TOP)
        return scale, val

    # ------------------------------------------------------------------ #
    #  Value accessors
    # ------------------------------------------------------------------ #
    def _current_D(self):
        return int(round(_log_map(self._d_scale.get(), self._d_min, self._d_max)))

    def _current_V(self):
        d = self._current_D()
        return int(round(_log_map(self._v_scale.get(), V_MIN, 10 * d)))

    def _current_S(self):
        d = self._current_D()
        frac = _log_map(self._s_scale.get(), S_FRAC_MIN, S_FRAC_MAX)
        return int(min(max(round(frac * d), 1), d))

    @property
    def params(self):
        """Current parameters as a dict.

        ``theta``/``mode`` are fixed constants -- thresholding is always
        Top-S (winner-take-all); the keys are kept only because the
        Simulation tab's work items and both tabs' param-signature helpers
        still key off them.
        """
        return {'D': self._current_D(), 'V': self._current_V(),
                'S': self._current_S(), 'theta': 0, 'mode': 'wta'}

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #
    def _on_d(self, _=None):
        self._recompute_dependent()
        self._schedule_commit('D')

    def _on_v(self, _=None):
        self._update_labels()
        self._schedule_commit('V')

    def _on_s(self, _=None):
        self._update_labels()
        self._schedule_commit('S')

    def _set_optimal_s(self):
        d = self._current_D()
        frac = core.optimal_sparsity(d) / d
        self._s_scale.set(_log_unmap(frac, S_FRAC_MIN, S_FRAC_MAX))
        self._on_s()

    # ------------------------------------------------------------------ #
    #  Dependent ranges / derived quantities
    # ------------------------------------------------------------------ #
    def _recompute_dependent(self):
        """Re-clamp V (range tracks D)."""
        # V range tracks D: keep the same absolute V if possible.
        v_val = self._current_V()
        self._v_scale.set(_log_unmap(v_val, V_MIN, 10 * self._current_D()))
        self._update_labels()

    def _update_labels(self):
        p = self.params
        self._d_val.config(text=f"{p['D']}")
        self._v_val.config(text=f"{p['V']}")
        self._s_val.config(text=f"{p['S']}\n{100.0 * p['S'] / p['D']:.2g}%")
        pd = core.p_density(p['D'], p['S'], p['V'])
        lam = p['V'] * p['S'] ** 2 / p['D'] ** 2
        self._info.config(
            text=(f"p (density) = {pd:.3f}\n"
                  f"λ = Mk²/N² = {lam:.3f}\n"
                  f"opt S = {core.optimal_sparsity(p['D'])}"))

    # ------------------------------------------------------------------ #
    #  Commit / debounce / external control
    # ------------------------------------------------------------------ #
    def _schedule_commit(self, source):
        if self._suppress:
            return
        if self._commit_job is not None:
            self.app.root.after_cancel(self._commit_job)
        self._commit_job = self.app.root.after(
            COMMIT_MS, lambda: self._commit(source))

    def _commit(self, source):
        self._commit_job = None
        if hasattr(self.app, 'on_params_changed'):
            self.app.on_params_changed(source)

    def set_tab_mode(self, mode):
        """Re-range the D slider for the active tab ('demo' or 'sim').

        Switching tabs is not a user parameter change, so commits are
        suppressed -- otherwise the debounced commit could later stop a sim the
        user just started.
        """
        self._suppress = True
        d = self._current_D()
        self._d_min, self._d_max = (10, 100) if mode == 'demo' else (10, 10000)
        d = min(max(d, self._d_min), self._d_max)
        self._d_scale.set(_log_unmap(d, self._d_min, self._d_max))
        self._recompute_dependent()
        self._suppress = False

    def set_params(self, params):
        """Restore a saved parameter dict (used by 'Revert & Resume' and the
        Demo tab's preset buttons).

        Suppresses the change-commit so restoring values doesn't itself fire
        ``app.on_params_changed``.  ``params['theta']``/``['mode']`` are
        accepted but ignored -- there's no UI for them any more.
        """
        self._suppress = True
        self._d_scale.set(_log_unmap(min(max(params['D'], self._d_min),
                                         self._d_max), self._d_min, self._d_max))
        d = self._current_D()
        self._v_scale.set(_log_unmap(params['V'], V_MIN, 10 * d))
        self._s_scale.set(_log_unmap(params['S'] / d, S_FRAC_MIN, S_FRAC_MAX))
        self._update_labels()
        self._suppress = False


# --------------------------------------------------------------------------- #
#  Stand-alone test
# --------------------------------------------------------------------------- #
class _FakeApp(object):
    def __init__(self, root):
        self.root = root
        self.inhibition_rate = 0.25
        self.excitation_rate = 0.25

    def on_params_changed(self, source=None):
        logging.info("params changed (%s): %s", source, self._panel.params)


def _test_control_panel():
    root = tk.Tk()
    root.title("Control Panel Test")
    root.geometry("300x950")
    root.configure(bg=tk_color_from_rgb(CS['bg']))
    app = _FakeApp(root)
    panel = ControlPanel(app, {'x_rel': (0.0, 1.0), 'y_rel': (0.0, 1.0)})
    app._panel = panel
    root.mainloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    _test_control_panel()
