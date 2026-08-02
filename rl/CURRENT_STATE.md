# CURRENT_STATE

Updated: 2026-08-02 — restart Version 1 complete (M0–M7).

## What currently works

- The demo app (`python rl_demo.py`): four model-based algorithms selectable and
  verified correct — Policy Evaluation + PI, In-Place PE, backward-induction
  Dynamic Programming, and In-Place (asynchronous) DP. Stepping at
  state/epoch/policy granularity, stop-states, save/load, opponent difficulty.
- All Tk access on the main thread; ~7–8 FPS rendering during full-speed runs.
- Game-tree generation + cache (`tic_tac_toe.py`), full-tree browser
  (`game_graph.py`), players (`HeuristicPlayer`, `MiniMaxPolicy`), match
  playing/statistics (`gameplay.py`), visualization components.
- NEAT/backprop experiment (standalone, finalized separately).
- Documentation: `docs/` + this file, kept in sync per milestone.

## What is incomplete

- Step-visualization panel (bottom-left) shows placeholder content; the real
  per-state update tree exists in `step_visualizer.py` and needs wiring (top of
  the post-V1 list).
- `q_learning.py`, `policy_grad.py` stubs; `monte_carlo.py` empty (future work).
- Parked WIP: `tree_step_viz.ValFuncViz`, `CompactBoxOrganizer`.

## What is known to be broken

- Nothing known-broken in the app path. Remaining small items are listed in
  `docs/known_issues.md`.

## Highest-priority remaining work (post-V1)

1. Wire `step_visualizer.StateUpdateStep` into the step-visualization panel.
2. Monte Carlo demo (first model-free algorithm).
3. Render-path optimization (PhotoImage reuse); see `docs/future_work.md`.

## Immediate next milestone

None active — Version 1 milestones complete. Next effort starts with the post-V1
list above.

## Milestone log

- M7: README rewritten to match reality (implemented demos table, controls,
  docs pointers; editorial FIXME markers removed).

- M6: legacy modules deleted (`gui_components.py`, `test_panels.py`,
  `game_learn.py`); Tournament block, dead helpers, `if False:` blocks, ipdb
  traces, and app-path debug prints removed; small bugs fixed (get_test_trace,
  RandomPlayer distribution, ValueFuncPolicy optional old_policy, visualize.py
  kwarg). ValFuncViz + CompactBoxOrganizer parked. All verifications re-pass.

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
