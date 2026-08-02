# Future Work

Explicitly deferred past Version 1. Do not start any of these during the restart.

## Additional algorithms (README "coming soon")

- **Monte Carlo** — `monte_carlo.py` is an empty file. First model-free demo; will need
  episode-trace visualization (much of `gameplay.Match`/`plot_trace` is reusable).
- **Q-learning** — `q_learning.py` stub. Needs a Q(s,a) visualization concept (per-action
  coloring; `drawing.get_action_dist_image` is a starting point).
- **Policy gradients / PPO / GLPO** — `policy_grad.py` stub; README sections 6–8.

## NEAT / backprop integration

`evolve_feedforward.py`, `backprop_net.py`, `neat_util.py`, `visualize.py` are a
finalized, separate command-line experiment (see commit 2cf9145). Folding them into the
demo GUI would require significant design work — new panel types (fitness/speciation
plots, network diagrams), a different learn-loop shape (generations, populations), and
probably a process boundary for training. **Deliberately not part of the restart.**
Revisit after V1; until then they live where they are and run standalone.

## Visualization

- **`tree_step_viz.ValFuncViz`** — intended successor to
  `step_visualizer.StateUpdateStep` (mouseover value-function subtree with captioned
  tiles). Parked incomplete; finish or fold its ideas back into step_visualizer.
- **`CompactBoxOrganizer`** (node_placement.py) — denser state-embedding layout;
  `_calc_box_positions` never written. Parked.
- **Simulated-annealing layer optimizer** — `layer_optimizer.py`'s docstring describes
  annealing-based within-layer placement; only deterministic zone bucketing was built.
- **Competition window** (README "main window" item 3) — continuous
  best-agent-vs-baseline matches in a separate process with live win/loss rates. The
  commented-out `Tournament` class in `gameplay.py` was a start; removed in M6, ideas
  live here.
- Micro/tiny histogram unification (two histogram implementations with overlapping
  purpose and a duplicated `get_n_bins`).

## Performance (beyond M5's fixes)

- Replace per-refresh numpy→PIL→PhotoImage conversion with reusable PhotoImage buffers
  or a faster blit path.
- `TinyHistogram` creates and destroys a matplotlib figure per histogram per draw —
  cache figures/axes.
- `game_graph.py` idle loop rebuilds the full 1920×1080 frame every `waitKey`
  iteration; make it event-driven. Also un-hardcode its resolution.
- `SlopeDiagram` recomputes full layout (incl. two MicroHists) on every draw.

## Educational tooling ideas (long-term vision)

- Multiple levels of detail per algorithm (equation view ↔ state-sweep view ↔
  aggregate convergence view).
- README equation placeholders ([ADD EQN], pseudocode) as rendered in-app content.
- Symmetry-reduced state space as an optional view (contrast with the full 8.5k view).
