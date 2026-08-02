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

- Tk widgets are updated from the algorithm thread (crash risk; fix in M5).
- The PE demo's displayed "learning" is placeholder math until M3 lands.

## Highest-priority remaining work

1. **M3 — port real PE/PI math from `game_learn.py` into `PolicyEvalDemoAlg`.** ← next
2. M4 — implement the DP demo algorithms.
3. M5 — threading/responsiveness repair; M6 — dead-code removal; M7 — README.

## Immediate next milestone

**M3**: PE/PI runs to convergence in the GUI with real Bellman updates; converged
greedy policy never loses as X vs Heuristic(6) and MiniMax.

## Milestone log

- M0 (58982fd): WIP baseline committed.
- M1 (bf6d908): docs/ + CURRENT_STATE.md reconstructed.
- M2: `python rl_demo.py` starts and renders; rename finished, signatures fixed.
