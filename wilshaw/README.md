# Willshaw Feature Attractor

A Tkinter app to explore the storage capacity, error-correcting ability, and
working operation of the **Willshaw feature attractor** — a sparse binary
associative memory (confabulation network).

```
python willshaw_app.py
```

Requires Python 3.12 with `numpy`, `scipy`, `matplotlib`, `opencv-python`,
`Pillow`, and `tkinter`.

## What it does

Patterns are binary vectors of exactly **S** active units in a **D**-dimensional
space (`1 <= S << D`); **V** of them are stored in the connectivity matrix
`W_ij = OR_μ x_i^μ x_j^μ`. Recall computes the excitation `h = W x`, thresholds
it, and iterates.

The narrow left panel holds the shared controls — vertical sliders for **D**,
**V**, **S** and threshold **θ**, a threshold-rule selector (fixed-θ or
winner-take-all), and buttons for the theoretical optimal **S** (`≈ log₂ D`) and
optimal **θ** (the midpoint `θ = c(1+p)/2` from `thresholds.md`). The main view
has two tabs:

- **Demonstration** — a small attractor you can watch work. Pick inhibition /
  excitation noise levels, click a pattern to damage and recover it, see the
  colour-coded input, the `W` heatmap with highlighted cue rows, the excitation /
  thresholded-output / reference stacks, and hover the iteration vectors to
  inspect each recall step.
- **Simulation** — a continuous multi-core Monte-Carlo run over large attractors.
  Each symbol is damaged with uniform-random inhibition/excitation noise (the same
  model as the demo), recovered, scored, and binned. Bar graphs show the recovery
  distribution and the average incorrect-activation rate per bin.

The Parameter-Grid tab from the spec is intentionally not implemented yet.

## Files

| File | Role |
|------|------|
| `willshaw_core.py` | Pure-numpy engine: patterns, `W`, recall, damage, theory helpers, scoring. Run it to self-test. |
| `sim_worker.py` | Multiprocessing worker (`run_chunk`). |
| `simulation_tab.py` | Simulation tab + threaded Pool driver. |
| `demo_tab.py` | Demonstration tab + interaction. |
| `control_panel.py` | Shared parameter sliders. |
| `draw_utils.py` / `plot_to_img.py` | Image primitives and matplotlib→image bridge. |
| `layout.py` / `gui_base.py` | Layout constants and the `Panel` base. |
| `willshaw_app.py` | Entry point. |

Most GUI modules run standalone (`python <module>.py`) to test that widget in
isolation. `WILLSHAW_SELFTEST=1 python willshaw_app.py` runs a short automated
simulation and exits (smoke test of the multiprocessing path).

## Notes

The workers run BLAS single-threaded (`OMP_NUM_THREADS=1` etc.) — we parallelise
across processes, and on Windows this keeps each spawned worker's memory
reservation small. Worker count defaults to `cpu_count() - 2`, capped further for
very large `D` to bound memory (`n_cpu · D²` bytes for the boolean `W`).
