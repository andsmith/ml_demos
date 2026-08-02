"""
Willshaw Feature Attractor -- exploration app.

A small Tkinter application to investigate the storage capacity and
error-correcting behaviour of the Willshaw sparse associative memory, and to
demonstrate its operation.

Window layout::

  +---------------------------------------------------------------+
  |  status / title bar                                           |
  +----------+----------------------------------------------------+
  | control  |  main view  (notebook)                             |
  | panel    |    [ Demonstration ] [ Simulation ]                |
  |  D V S θ |                                                    |
  +----------+----------------------------------------------------+

Run with::  python willshaw_app.py

Only the Demonstration and Simulation tabs are implemented (the Parameter Grid
tab from the spec is intentionally omitted).
"""

import logging
import multiprocessing as mp
import tkinter as tk
import tkinter.ttk as ttk

from gui_base import tk_color_from_rgb
from layout import LAYOUT, FONTS, WIN_SIZE, COLOR_SCHEME as CS

# NOTE: control_panel / demo_tab / simulation_tab are imported lazily inside the
# _init_* methods below.  They pull in cv2 and matplotlib, and multiprocessing's
# "spawn" start method re-imports this module in every worker process -- keeping
# those heavy imports out of module scope stops each worker from loading cv2 +
# matplotlib (which otherwise exhausts memory when many workers start at once).


class WillshawApp(object):
    """Top-level application object."""

    def __init__(self):
        # Shared damage state (set on the Demo tab, read by the Simulation).
        self.inhibition_rate = 0.25
        self.excitation_rate = 0.25
        self._active = 'demo'

        self._init_tk()
        self._init_status()
        self._init_control()
        self._init_tabs()

        # Kick off the initial demonstration once the event loop is running.
        self.root.after(250, lambda: self._demo.on_show())

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _init_tk(self):
        self.root = tk.Tk()
        self.root.title("Willshaw Feature Attractor")
        self.root.geometry(f"{WIN_SIZE[0]}x{WIN_SIZE[1]}")
        self.root.configure(bg=tk_color_from_rgb(CS['bg']))
        self.root.minsize(1100, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_status(self):
        bb = LAYOUT['status']
        frame = tk.Frame(self.root, bg=tk_color_from_rgb(CS['panel_bg']))
        frame.place(relx=bb['x_rel'][0], rely=bb['y_rel'][0],
                    relwidth=bb['x_rel'][1] - bb['x_rel'][0],
                    relheight=bb['y_rel'][1] - bb['y_rel'][0])
        tk.Label(frame, text="Willshaw Feature Attractor",
                 bg=tk_color_from_rgb(CS['panel_bg']),
                 fg=tk_color_from_rgb(CS['text']),
                 font=FONTS['title']).pack(side=tk.LEFT, padx=12)
        self._status_lbl = tk.Label(
            frame, text="", bg=tk_color_from_rgb(CS['panel_bg']),
            fg=tk_color_from_rgb(CS['text_dim']), font=FONTS['status'])
        self._status_lbl.pack(side=tk.LEFT, padx=20)

    def _init_control(self):
        from control_panel import ControlPanel
        self.control_panel = ControlPanel(
            self, LAYOUT['control'], margin_rel=LAYOUT['margin_rel'])

    def _init_tabs(self):
        from demo_tab import DemoTab
        from simulation_tab import SimulationTab
        bb = LAYOUT['main']
        main = tk.Frame(self.root, bg=tk_color_from_rgb(CS['bg']))
        main.place(relx=bb['x_rel'][0], rely=bb['y_rel'][0],
                   relwidth=bb['x_rel'][1] - bb['x_rel'][0],
                   relheight=bb['y_rel'][1] - bb['y_rel'][0])

        style = ttk.Style(main)
        style.configure('TNotebook.Tab', font=FONTS['tabs'], padding=(18, 8))
        self._notebook = ttk.Notebook(main, style='TNotebook')
        self._notebook.pack(fill=tk.BOTH, expand=True)

        demo_frame = tk.Frame(self._notebook, bg=tk_color_from_rgb(CS['panel_bg']))
        sim_frame = tk.Frame(self._notebook, bg=tk_color_from_rgb(CS['panel_bg']))
        self._notebook.add(demo_frame, text="  Demonstration  ")
        self._notebook.add(sim_frame, text="  Simulation  ")

        self._demo = DemoTab(demo_frame, self)
        self._sim = SimulationTab(sim_frame, self)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # ------------------------------------------------------------------ #
    #  Shared API used by panels / tabs
    # ------------------------------------------------------------------ #
    @property
    def params(self):
        """Current parameters from the control panel."""
        return self.control_panel.params

    def on_params_changed(self, source=None):
        """Route a parameter change to the active tab and refresh the status."""
        self._update_status()
        tab = self._demo if self._active == 'demo' else self._sim
        tab.on_params_changed(source)

    def _on_tab_changed(self, _event):
        idx = self._notebook.index(self._notebook.select())
        self._active = 'demo' if idx == 0 else 'sim'
        self.control_panel.set_tab_mode(self._active)
        self._update_status()
        if self._active == 'demo':
            self._demo.on_show()
        else:
            self._sim.on_show()

    def _update_status(self):
        p = self.params
        self._status_lbl.config(
            text=(f"D={p['D']}   V={p['V']}   S={p['S']} "
                  f"({100.0 * p['S'] / p['D']:.2g}%)    |    "
                  f"{'Demonstration' if self._active == 'demo' else 'Simulation'} mode"))

    def _on_close(self):
        try:
            self._sim.on_close()
        finally:
            self.root.destroy()

    def start(self):
        self.root.mainloop()


def _self_test(app):
    """Env-gated smoke test (WILLSHAW_SELFTEST=1): exercise the sim then quit.

    Runs in the real __main__ process so it validates the multiprocessing spawn
    path (workers importing this module) end to end, then closes the window.
    """
    def step_run():
        app._notebook.select(1)
        app.control_panel.set_params(
            {'D': 200, 'V': 1000, 'S': 7, 'theta': 0, 'mode': 'wta'})
        app._sim._toggle_start()
        app.root.after(12000, step_stop)

    def step_stop():
        app._sim._toggle_start()
        logging.info("SELFTEST sim total=%d chunks=%d",
                     app._sim._stats['total'], app._sim._stats['chunks'])
        app.root.after(800, lambda: app._on_close())

    app.root.after(1200, step_run)


def main():
    import os
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    app = WillshawApp()
    if os.environ.get('WILLSHAW_SELFTEST'):
        _self_test(app)
    app.start()


if __name__ == '__main__':
    mp.freeze_support()
    main()
