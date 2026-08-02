# Known Issues

Status: recorded at restart baseline (commit 58982fd). Items are removed here when
fixed; milestone tags (M2–M6) refer to the restart roadmap.

## Blocking (app cannot run)

(All fixed in M2: TERM_REWARDS rename completed everywhere; `PolicyEvalDemoAlg` takes
`(app, env, pi_seed=None, gamma)` matching how `rl_demo` constructs algorithms;
`PIPhases` now defined once in `reinforcement_base.py` with members
POLICY_EVAL/POLICY_OPTIM.)

## Faked / stubbed math — M4

- (M3 fixed the PE/PI math: real Bellman backups via `_expected_return`, greedy
  per-state improvement into a `TabularPolicy`, real convergence checks. Verified
  headless: converges in 4 PI iterations; never loses to its training opponent —
  500/500 wins vs Heuristic(6), 500/500 draws vs MiniMax.)
- (M4 implemented `dynamic_prog.py`: backward-induction DP + async in-place VI,
  both verified against converged PE/PI values.)
- `q_learning.py`, `policy_grad.py` are name-only stubs; `monte_carlo.py` is empty.
- `step_visualizer.py`: `PIStep` / `ContinuousStep` are `pass # TODO` (:471, :488).
- The step-visualization panel (`get_viz_image`) still shows placeholder text/state
  icon rather than the per-state update tree from `step_visualizer.StateUpdateStep` —
  wiring that in is part of the remaining viz work (M6 decision on
  step_visualizer vs tree_step_viz).

## Threading / responsiveness

(M5 fixed the structural issues: the algorithm thread now only posts render
requests/callables that a main-thread `root.after` loop consumes — no Tk calls off
the main thread; tab image caches are guarded by a per-page lock; mouse-move
refreshes coalesce to ≤60 Hz; `DemoAlg.stop()` is a single set + join with timeout,
and `_maybe_pause` never re-blocks after shutdown; the learn loop yields the GIL
briefly every 25ms so free-run rendering holds ~7–8 FPS while the loop still
processes hundreds of backups/sec.)

Remaining (documented in future_work.md): per-frame numpy→PIL→PhotoImage rebuild is
the main render cost; `BoxOrganizer.get_state_at` linear scan survives only in
legacy `gui_components.py` and standalone `game_graph.py` (the app path uses the
`MouseBoxManager` KDTree).

## Parked WIP (deliberately incomplete, not imported by the app)

- `tree_step_viz.ValFuncViz`: unfinished successor to
  `step_visualizer.StateUpdateStep` (bare statement, missing `_dims['tiles']`,
  "UNCHECKED BELOW HERE"). Parked in M6; see future_work.md.
- `CompactBoxOrganizer` (node_placement.py): `_calc_box_positions` is `pass`; its
  test also spells a constructor arg `draw_darams`. Parked.
- `MicroHist` 'bins' mode raises NotImplementedError; `SlopeDiagram` `fast` param
  unused.

## Small bugs (remaining)

- `selection_panel.py` radio buttons pass a plain str as `variable` instead of a
  `tk.Variable`; grouping works only via command callbacks.
- `tab_content.py` `_draw_marked` uses `self._embed`, which only exists on the
  subclass — base class unusable standalone (rename-to-MouseContentPage TODO).
- `step_visualizer.py:560` test harness references undefined `old_values` (stale
  signature).

(M6 fixed: `gameplay.get_test_trace` trace-before-assignment;
`RandomPlayer.recommend_action` now returns a proper distribution;
`ValueFuncPolicy` old_policy arg optional; `visualize.py` bad `h_spacing` kwarg.)

## Dead code / residue

(M6 removed: `gui_components.py`, `test_panels.py`, `game_learn.py` (old app —
math fully absorbed in M3), the commented `Tournament` block,
`LayerwiseBoxOrganizer._row_col_adjustX`, `if False:` blocks, live/commented ipdb
traces, and app-path debug prints. Test-harness prints under `__main__` remain by
design.)

## Non-code

- README describes the RL demo apps as if complete ("COMPLETED WORK ABOVE HERE" line
  is accurate — everything below it is aspirational); update in M7.
- Generated artifacts are tracked in git (`*.pkl`, scratch `*.png`).
- The empty board's O-first children merge (`get_game_tree(generic=True)`) means some
  drawn edges are only valid for one first-player choice — a display subtlety noted in
  the README, worth an in-app explanation eventually.
