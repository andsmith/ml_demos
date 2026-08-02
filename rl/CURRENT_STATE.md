# CURRENT_STATE

Updated: 2026-08-02 (restart baseline: commit 58982fd; docs reconstructed)

## What currently works

- Game-tree generation + cache (`tic_tac_toe.py`, `game_tree_X.pkl`), full-tree
  browser (`game_graph.py`).
- Old standalone PI demo math (`game_learn.py`): correct iterative policy evaluation +
  greedy improvement (`policy_optim.ValueFuncPolicy`). Its GUI is legacy.
- Players: `HeuristicPlayer` rule ladder, `MiniMaxPolicy` (cached perfect player).
- Match playing/statistics (`gameplay.py` Match/ResultSet) and most visualization
  components (drawing, node placement, state embedding, color keys, step_visualizer).
- NEAT/backprop experiment (standalone, finalized separately).

## What is incomplete

- New app (`rl_demo.py` + `DemoAlg`): architecture complete, but `policy_eval.py`
  math is placeholder (random value updates, forced convergence).
- `dynamic_prog.py` (V1 scope), `q_learning.py`, `policy_grad.py` are stubs;
  `monte_carlo.py` is empty.
- `tree_step_viz.ValFuncViz` and `CompactBoxOrganizer` are parked WIP.

## What is known to be broken

- Step-visualization panel shows placeholder content (real per-state update tree
  exists in `step_visualizer.py`; wiring decision in M6).

## Highest-priority remaining work

1. **M6 — dead-code removal & small bug fixes.** ← next
2. M7 — README update & final polish.

## Immediate next milestone

**M6**: legacy modules retired (`gui_components.py`, `test_panels.py`, old app),
small known bugs fixed, ValFuncViz/CompactBoxOrganizer parked or completed.

## Milestone log

- M0 (58982fd): WIP baseline committed.
- M1 (bf6d908): docs/ + CURRENT_STATE.md reconstructed.
- M2 (c777bae): `python rl_demo.py` starts and renders; rename finished.
- M3 (79aecbf): real PE/PI math in `PolicyEvalDemoAlg` (Bellman backups, greedy
  improvement, true convergence). Verified: 4-iteration convergence; 500/500 wins vs
  training opponent Heuristic(6); 500/500 draws (0 losses) when trained vs MiniMax.
- M5: all Tk access on the main thread (worker posts render requests to a
  `root.after` loop); image caches locked; motion refresh coalesced; clean thread
  shutdown; GIL yield keeps free-run rendering at ~7-8 FPS. Verified with a 25s
  full-speed scripted run.
- M4 (57cecd0): DP demos implemented (backward-induction DP exact in one pass; async in-place
  VI); In-Place PE is now genuinely in-place. DP values match converged PI values to
  2e-16; greedy(DP) policy 500/500 wins vs Heuristic(6). All 4 model-based demos
  selectable in the app.
