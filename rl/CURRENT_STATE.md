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

- **The new app does not start**: half-finished `TERM_REWARDS` rename breaks imports
  (`rl_demo.py:25`, `policy_eval.py:7`, plus mis-keyed lookups). See
  `docs/known_issues.md` for the full register.
- Tk widgets are updated from the algorithm thread (crash risk).

## Highest-priority remaining work

1. **M2 — make `rl_demo.py` start** (finish the rename, fix signatures). ← next
2. M3 — port real PE/PI math from `game_learn.py` into `PolicyEvalDemoAlg`.
3. M4 — implement the DP demo algorithms.
4. M5 — threading/responsiveness repair; M6 — dead-code removal; M7 — README.

## Immediate next milestone

**M2**: `python rl_demo.py` launches, all panels render, stub algorithms greyed out.
