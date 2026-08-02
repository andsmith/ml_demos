# Design Decisions

A running log. Add an entry whenever a non-obvious decision is made.

## 2026-08-02 — Project restart (Version 1 scope)

- **Finish the migration, retire the old app.** The repo contained two app
  generations. Decision: port the working PE/PI math from `game_learn.py` into the new
  `rl_demo.py`/`DemoAlg` architecture, then delete `gui_components.py` and
  `game_learn.py`. Rationale: the new architecture is the clear in-flight intent
  (pluggable algorithms, panel layout from `layout.py`) and is better suited to the
  long-term goal of many algorithms/visualizations.
- **V1 algorithm scope = PE/PI + Dynamic Programming.** Monte Carlo, Q-learning, and
  Policy Gradients remain documented future work (`monte_carlo.py` is empty; adding it
  would be new-algorithm work, which the restart explicitly excludes).
- **NEAT/backprop experiment stays in place.** `evolve_feedforward.py`,
  `backprop_net.py`, `neat_util.py`, `visualize.py` are a finalized separate
  experiment. Folding them into the demo GUI needs real design work and is out of
  scope for the restart (recorded in `future_work.md`).
- **WIP committed as a baseline first** (commit `58982fd`) so all repair work diffs
  against a known snapshot.

## 2026-08-02 — M4: what the DP demos mean

- The four model-based demos are differentiated on two axes — backup type and
  update discipline:
  - **PE** (`pi-pe`): policy-expectation backup, two-array (Jacobi) commits per epoch.
  - **In-Place PE** (`pi-pe-inplace`): same backup, Gauss-Seidel (immediate) commits.
  - **DP** (`dp`): optimality (max) backup with backward induction — children before
    parents, in place — so V* is exact after ONE pass. This is the README's "compute
    values directly by following the state-transition graph".
  - **In-Place DP** (`dp-inplace`): same max backup, but in default screen order —
    asynchronous value iteration, converging in a few passes without the "right" order.
- All four share `PolicyEvalDemoAlg`'s machinery (`_expected_return`, tabs, control
  points, greedy improvement); the DP subclasses only override the backup and the
  sweep order. Verified: DP one-pass values match converged PE/PI values to 2e-16.
- Convergence is tracked as max |dV| per epoch (works identically for Jacobi and
  in-place variants), tolerance 1e-6.

## Inherited decisions (reconstructed from the code)

- **Opponent folded into the environment.** The agent sees a single-player stochastic
  MDP: `Environment` composes the game tree with the opponent policy's action
  distribution, so transition probabilities are opponent moves. This keeps the RL
  algorithms textbook-shaped (Sutton & Barto ch. 4).
- **Full-enumeration model.** No symmetry reduction of the state space — deliberately,
  so the visualization shows every raw state (~8.5k) and edge counts match intuition.
- **cv2-into-numpy rendering, Tk for chrome.** All visual content is drawn with OpenCV
  into numpy buffers and pushed to Tk as PhotoImages. Rationale: full pixel control,
  identical rendering in standalone cv2 apps and the Tk app; Tk supplies widgets
  (tabs, buttons, sliders) only.
- **Three-layer image caching per tab** (`base` → `marked` → `display`) with selective
  invalidation, so mouse interaction never redraws the expensive all-states base image.
- **Layered placement, not graph embedding.** States are placed in bands by move count
  with within-layer semantic zoning (`SimpleTreeOptimizer`), instead of a force-directed
  layout. Deterministic, fast, and readable.
- **Terminal rewards:** win +1.0, loss −1.0, draw −0.5 (`game_base.TERM_REWARDS`,
  keyed by player). Draws are deliberately penalized so the agent prefers wins.
- **Blocking-checkbox stepping model.** Algorithm granularity control is implemented as
  named control points inside the learn loop that block on a shared `threading.Event`,
  rather than a command queue. Simple, but couples rendering to the algorithm thread
  (defect tracked in `known_issues.md`, fix in M5).
