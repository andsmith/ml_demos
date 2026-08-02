"""
Simulation tab: run a continuous Monte-Carlo experiment over large Willshaw
attractors, accumulate recovery statistics, and show them as bar graphs.

A background driver thread streams "chunks" of work to a multiprocessing Pool
(one chunk = several networks, each evaluated on all its symbols with fresh
uniform-random damage) and pushes results onto a queue; the Tk side drains the
queue on a timer, merges the tallies and redraws the charts.

Controls (top of the tab):
  * Start / Stop        -- continuous MC over many fresh random networks.
  * Run N trials / set  -- one-shot: N trials on a single fresh network.
  * Revert & Resume     -- restore the parameters the current stats belong to
                           and continue (enabled after a control change).

Per the shared design, the cue damage reuses the Demonstration tab's
inhibition / excitation noise levels (shown here read-only), drawn uniformly at
random per symbol.
"""

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import tkinter as tk

# Spawned workers inherit this environment; keep BLAS single-threaded so each
# process's virtual-memory reservation stays small (avoids "paging file too
# small" on Windows when many workers start together).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

import sim_worker
import willshaw_core as core
from gui_base import blit, tk_color_from_rgb
from layout import FONTS, COLOR_SCHEME as CS
from plot_to_img import PlotRenderer

TARGET_SYMBOLS_PER_CHUNK = 3000      # size chunks to amortise IPC overhead
REDRAW_MS = 250                      # min interval between chart redraws
MEM_BUDGET_BYTES = 1.5e9             # cap workers so n_cpu * D^2 stays under this


def _empty_stats():
    return {'bin_counts': np.zeros(core.N_BINS, dtype=np.int64),
            'bin_inc_sum': np.zeros(core.N_BINS, dtype=np.float64),
            'total': 0, 'perfect': 0, 'iter_sum': 0, 'chunks': 0}


class SimulationTab(object):
    """Content + Monte-Carlo driver for the Simulation tab."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self._running = False
        self._dirty = False               # a control changed since stats began
        self._has_run = False
        self._active_params = None        # params the current stats belong to
        self._stats = _empty_stats()
        self._queue = queue.Queue()
        self._driver = None
        self._pool = None
        self._seed_gen = np.random.default_rng()
        self._t0 = None
        self._last_draw = 0.0
        self._need_redraw = False
        self._plot = PlotRenderer((640, 480))
        self._last_view = None

        self._build_widgets()
        self._poll_queue()                # start the queue pump

    # ------------------------------------------------------------------ #
    #  Widgets
    # ------------------------------------------------------------------ #
    def _build_widgets(self):
        bg = tk_color_from_rgb(CS['panel_bg'])
        self.parent.configure(bg=bg)
        self.parent.rowconfigure(1, weight=1)
        self.parent.columnconfigure(0, weight=1)

        ctl = tk.Frame(self.parent, bg=bg)
        ctl.grid(row=0, column=0, sticky='ew', padx=4, pady=3)

        self._start_btn = tk.Button(ctl, text="▶ Start", width=12,
                                    font=FONTS['buttons'], command=self._toggle_start)
        self._start_btn.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(ctl, text="N trials:", bg=bg, fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['default']).pack(side=tk.LEFT)
        self._n_var = tk.IntVar(value=5)
        tk.Spinbox(ctl, from_=1, to=1000, width=5, textvariable=self._n_var,
                   font=FONTS['default']).pack(side=tk.LEFT, padx=(2, 4))
        tk.Button(ctl, text="Run N / new set", font=FONTS['buttons'],
                  command=self._run_oneshot).pack(side=tk.LEFT, padx=4)

        self._revert_btn = tk.Button(ctl, text="↩ Revert & Resume",
                                    font=FONTS['buttons'], state=tk.DISABLED,
                                    command=self._revert_resume)
        self._revert_btn.pack(side=tk.LEFT, padx=8)

        self._status = tk.Label(ctl, text="idle", bg=bg,
                               fg=tk_color_from_rgb(CS['text_dim']),
                               font=FONTS['small'])
        self._status.pack(side=tk.LEFT, padx=10)

        # stats readout row
        self._stat_label = tk.Label(self.parent, text="", bg=bg, justify=tk.LEFT,
                                   anchor=tk.W, fg=tk_color_from_rgb(CS['text']),
                                   font=FONTS['value'])
        self._stat_label.grid(row=2, column=0, sticky='ew', padx=6, pady=(0, 4))

        view = tk.Frame(self.parent, bg=tk_color_from_rgb(CS['view_bg']),
                        highlightthickness=1,
                        highlightbackground=tk_color_from_rgb(CS['lines']))
        view.grid(row=1, column=0, sticky='nsew', padx=4, pady=2)
        view.pack_propagate(False)      # chart image must not resize the frame
        self._view = view
        self._chart = tk.Label(view, bg=tk_color_from_rgb(CS['view_bg']))
        self._chart.pack(fill=tk.BOTH, expand=True)
        view.bind("<Configure>", lambda e: self._request_redraw(force=True))

    # ------------------------------------------------------------------ #
    #  External hooks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _params_sig(p):
        return (p['D'], p['V'], p['S'], p['theta'], p['mode'])

    def on_params_changed(self, source=None):
        """A control changed: stop the run but keep the (now stale) stats.

        Ignore commits that don't actually change the parameters (e.g. the
        deferred tab-switch re-ranging), so they can't stop a running sim.
        """
        if (self._active_params is not None
                and self._params_sig(self.app.params)
                == self._params_sig(self._active_params)):
            return
        if self._running:
            self._stop()
        if self._has_run:
            self._dirty = True
            self._revert_btn.config(state=tk.NORMAL)
            self._start_btn.config(text="▶ Start (clears)")
            self._set_status("paused -- params changed")

    def on_show(self):
        self._request_redraw(force=True)

    def on_close(self):
        self._stop()

    # ------------------------------------------------------------------ #
    #  Run control
    # ------------------------------------------------------------------ #
    def _toggle_start(self):
        if self._running:
            self._stop()
            self._set_status("stopped")
            self._start_btn.config(text="▶ Start")
            return
        self._start_continuous(clear=True)

    def _start_continuous(self, clear):
        params = self.app.params
        if clear:
            self._stats = _empty_stats()
            self._active_params = params
            self._has_run = True
            self._t0 = time.perf_counter()
        self._dirty = False
        self._revert_btn.config(state=tk.DISABLED)
        self._start_btn.config(text="■ Stop")
        self._running = True
        self._launch_driver(params, continuous=True)
        self._set_status("running (continuous)")

    def _run_oneshot(self):
        if self._running:
            self._stop()
        params = self.app.params
        self._stats = _empty_stats()
        self._active_params = params
        self._has_run = True
        self._dirty = False
        self._t0 = time.perf_counter()
        self._revert_btn.config(state=tk.DISABLED)
        self._running = True
        self._start_btn.config(text="■ Stop")
        self._launch_driver(params, continuous=False,
                            total_trials=int(self._n_var.get()))
        self._set_status(f"running {self._n_var.get()} trials on one network")

    def _revert_resume(self):
        if self._active_params is None:
            return
        self.app.control_panel.set_params(self._active_params)
        self._dirty = False
        self._revert_btn.config(state=tk.DISABLED)
        self._start_continuous(clear=False)

    def _stop(self):
        self._running = False
        drv = self._driver
        if drv is not None and drv.is_alive():
            drv.join(timeout=5.0)
        self._driver = None
        self._start_btn.config(text="▶ Start")

    # ------------------------------------------------------------------ #
    #  Driver thread + multiprocessing
    # ------------------------------------------------------------------ #
    def _effective_ncpu(self, D):
        base = max(1, mp.cpu_count() - 2)
        by_mem = max(1, int(MEM_BUDGET_BYTES / max(D * D, 1)))
        return max(1, min(base, by_mem))

    def _next_seed(self):
        return int(self._seed_gen.integers(0, 2 ** 31 - 1))

    def _make_work(self, params, n_sets, n_trials, net_seed):
        return {'D': params['D'], 'S': params['S'], 'V': params['V'],
                'theta': params['theta'], 'mode': params['mode'],
                'max_iter': core.DEFAULT_MAX_ITER,
                'inh_rate': float(self.app.inhibition_rate),
                'exc_rate': float(self.app.excitation_rate),
                'n_sets': n_sets, 'n_trials': n_trials,
                'net_seed': net_seed, 'dmg_seed': self._next_seed()}

    def _launch_driver(self, params, continuous, total_trials=0):
        if self._driver is not None and self._driver.is_alive():
            logging.warning("driver already running; not launching another")
            return
        n_cpu = self._effective_ncpu(params['D'])
        sets_per_chunk = max(1, round(TARGET_SYMBOLS_PER_CHUNK / max(params['V'], 1)))
        self._driver = threading.Thread(
            target=self._driver_loop,
            args=(params, continuous, total_trials, n_cpu, sets_per_chunk),
            daemon=True)
        self._driver.start()

    def _driver_loop(self, params, continuous, total_trials, n_cpu, sets_per_chunk):
        """Runs off the Tk thread: stream chunks through a spawn Pool."""
        ctx = mp.get_context('spawn')
        pool = None
        try:
            pool = ctx.Pool(n_cpu)
            self._pool = pool
            if continuous:
                def work_gen():
                    while self._running:
                        yield self._make_work(params, n_sets=sets_per_chunk,
                                              n_trials=1, net_seed=self._next_seed())
                for res in pool.imap_unordered(sim_worker.run_chunk, work_gen()):
                    self._queue.put(('update', res))
                    if not self._running:
                        break
            else:
                # One-shot: N trials on a single fixed network, split as evenly
                # as possible across at most n_cpu chunks (exactly N total).
                net_seed = self._next_seed()
                n_chunks = max(1, min(n_cpu, total_trials))
                base = total_trials // n_chunks
                remainder = total_trials - base * n_chunks
                works = []
                for i in range(n_chunks):
                    nt = base + (1 if i < remainder else 0)
                    if nt <= 0:
                        continue
                    works.append(self._make_work(params, n_sets=1, n_trials=nt,
                                                 net_seed=net_seed))
                for res in pool.imap_unordered(sim_worker.run_chunk, works):
                    self._queue.put(('update', res))
                self._queue.put(('done', None))
        except Exception as exc:                       # pragma: no cover
            self._queue.put(('error', repr(exc)))
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
            self._pool = None

    # ------------------------------------------------------------------ #
    #  Queue pump + merge
    # ------------------------------------------------------------------ #
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'update':
                    self._merge(payload)
                    self._request_redraw()
                elif kind == 'done':
                    self._running = False
                    self._start_btn.config(text="▶ Start")
                    self._set_status("done (one-shot complete)")
                    self._request_redraw(force=True)
                elif kind == 'error':
                    self._running = False
                    self._start_btn.config(text="▶ Start")
                    self._set_status(f"error: {payload}")
                    logging.error("simulation worker error: %s", payload)
        except queue.Empty:
            pass
        self._maybe_redraw()
        self.app.root.after(100, self._poll_queue)

    def _merge(self, res):
        s = self._stats
        s['bin_counts'] += np.asarray(res['bin_counts'], dtype=np.int64)
        s['bin_inc_sum'] += np.asarray(res['bin_inc_sum'], dtype=np.float64)
        s['total'] += res['total']
        s['perfect'] += res['perfect']
        s['iter_sum'] += res['iter_sum']
        s['chunks'] += 1

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #
    def _request_redraw(self, force=False):
        self._need_redraw = True
        if force:
            self._last_draw = 0.0

    def _maybe_redraw(self):
        if not self._need_redraw:
            return
        if (time.perf_counter() - self._last_draw) * 1000.0 < REDRAW_MS:
            return
        self._draw_charts()
        self._update_stat_label()
        self._need_redraw = False
        self._last_draw = time.perf_counter()

    def _draw_charts(self):
        w = self._view.winfo_width()
        h = self._view.winfo_height()
        if w <= 1 or h <= 1:
            return
        self._plot.set_size((w, h))
        fig, axes = self._plot.get_axis(n_rows=2, n_cols=1)
        counts = self._stats['bin_counts']
        total = max(self._stats['total'], 1)
        inc_sum = self._stats['bin_inc_sum']
        x = np.arange(core.N_BINS)
        bar_c = ['#3a6fd0'] * (core.N_BINS - 1) + ['#1e8c3c']   # perfect bin green

        frac = 100.0 * counts / total
        axes[0].bar(x, frac, color=bar_c)
        axes[0].set_ylabel("% of symbols")
        axes[0].set_title(f"Recovery distribution   (n = {self._stats['total']:,} symbols)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(core.BIN_LABELS, rotation=0, fontsize=8)
        axes[0].set_ylim(0, max(5, frac.max() * 1.15) if counts.sum() else 5)
        for xi, f in zip(x, frac):
            if f > 0.5:
                axes[0].text(xi, f, f"{f:.0f}", ha='center', va='bottom', fontsize=7)

        with np.errstate(invalid='ignore', divide='ignore'):
            avg_inc = np.where(counts > 0, inc_sum / np.maximum(counts, 1), 0.0)
        axes[1].bar(x, 100.0 * avg_inc, color='#b0301e')
        axes[1].set_ylabel("avg spurious %\n(of S, per symbol)")
        axes[1].set_title("Average incorrect-activation rate by recovery bin")
        axes[1].set_xlabel("recovery-rate bin (%)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(core.BIN_LABELS, rotation=0, fontsize=8)

        fig.tight_layout()
        blit(self._chart, self._plot.render_fig(fig))

    def _update_stat_label(self):
        s = self._stats
        p = self._active_params or self.app.params
        elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
        rate = s['total'] / elapsed if elapsed > 0 else 0.0
        perfect_pct = 100.0 * s['perfect'] / s['total'] if s['total'] else 0.0
        avg_iter = s['iter_sum'] / s['total'] if s['total'] else 0.0
        pd = core.p_density(p['D'], p['S'], p['V'])
        ncpu = self._effective_ncpu(p['D'])
        self._stat_label.config(text=(
            f"D={p['D']}  V={p['V']}  S={p['S']}  θ={p['theta']} ({p['mode']})   "
            f"noise: inh≤{self.app.inhibition_rate:.2f} exc≤{self.app.excitation_rate:.2f} "
            f"(set on Demo tab)\n"
            f"symbols={s['total']:,}   perfect={perfect_pct:.1f}%   "
            f"avg iters={avg_iter:.2f}   p(density)={pd:.3f}   "
            f"throughput={rate:,.0f} sym/s   cores={ncpu}"))

    def _set_status(self, text):
        self._status.config(text=text)


# --------------------------------------------------------------------------- #
#  Stand-alone test
# --------------------------------------------------------------------------- #
class _FakeApp(object):
    def __init__(self, root):
        self.root = root
        self.inhibition_rate = 0.4
        self.excitation_rate = 0.4
        self.params = {'D': 500, 'S': 7, 'V': 3000, 'theta': 0, 'mode': 'wta'}

        class _CP(object):
            def set_params(self, p):
                logging.info("revert to %s", p)
        self.control_panel = _CP()


def _test_simulation_tab():
    root = tk.Tk()
    root.title("Simulation Tab Test")
    root.geometry("1000x800")
    app = _FakeApp(root)
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    sim = SimulationTab(frame, app)
    root.after(300, sim.on_show)
    root.protocol("WM_DELETE_WINDOW", lambda: (sim.on_close(), root.destroy()))
    root.mainloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    mp.freeze_support()
    _test_simulation_tab()
