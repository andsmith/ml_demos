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

## Threading / responsiveness — M5

- `RLDemoApp.tick` (rl_demo.py:229) runs on the algorithm thread and calls Tk widget
  methods directly (`label.config(image=...)`) — formally unsafe, intermittent-crash
  risk.
- Mouse-move path: every motion event copies the full base image
  (tab_content.py:175) and rebuilds a PhotoImage (alg_panels.py:215) — no throttling.
- Paused-tick path (rl_demo.py:234-240) has no FPS gate; every pause hit refreshes all
  panels.
- `BoxOrganizer.get_state_at` (node_placement.py:242) is an O(n) linear scan;
  `MouseBoxManager` already has a KDTree — not used everywhere.
- `DemoAlg.stop()` (rl_alg_base.py:199-217) join/retry/sleep workaround for threads
  not exiting cleanly.
- `rl_demo.py:263`: `self._ticks_skipped` bare expression — skip counter never resets.

## Broken / incomplete visualization — M6 (default: park)

- `tree_step_viz.ValFuncViz`: bare `action_tile` statement (:393),
  `self._dims['tiles']` never populated → KeyError (:419), live
  `import ipdb; ipdb.set_trace()` in test (:480), "UNCHECKED BELOW HERE" (:433).
- `CompactBoxOrganizer` (node_placement.py:~539): `_calc_box_positions` is `pass`; its
  test calls a constructor arg (`draw_params`) spelled `draw_darams` in the signature.
- `MicroHist` 'bins' mode raises NotImplementedError; `SlopeDiagram` `fast` param
  unused.
- `visualize.py:195` passes nonexistent `h_spacing` kwarg (crashes 9-input NEAT case).

## Small bugs — M6

- `gameplay.py:586` (`get_test_trace`): `trace` referenced before assignment when
  `required_result` is given.
- `baseline_players.py:21` `RandomPlayer.recommend_action` returns a bare action, not
  an `[(action, prob)]` distribution — violates the `Policy` contract.
- `game_learn.py:230` `filename="value_function_converged.pkl" % (self._iter+1)` →
  TypeError on convergence (moot once game_learn.py is retired).
- `policy_optim.py test_value_func_policy` calls `ValueFuncPolicy` without the required
  `old_policy` arg; `step_visualizer.py:560` test references undefined `old_values`.
- `selection_panel.py` radio buttons pass a plain str as `variable` instead of a
  `tk.Variable`; grouping works only via command callbacks.
- `tab_content.py` `_draw_marked` uses `self._embed`, which only exists on the
  subclass — base class unusable standalone.

## Dead code / residue — M6

- `gui_components.py` (legacy `RLDemoWindow`, unreferenced), `test_panels.py` (fully
  commented out, references nonexistent `StatePanel`), `gameplay.py` `Tournament`
  block (commented), `LayerwiseBoxOrganizer._row_col_adjustX`, assorted `if False:`
  blocks, commented ipdb traces, stray debug `print()`s (incl. policy_eval.py:481),
  debug `cv2.imwrite` side effects.

## Non-code

- README describes the RL demo apps as if complete ("COMPLETED WORK ABOVE HERE" line
  is accurate — everything below it is aspirational); update in M7.
- Generated artifacts are tracked in git (`*.pkl`, scratch `*.png`).
- The empty board's O-first children merge (`get_game_tree(generic=True)`) means some
  drawn edges are only valid for one first-player choice — a display subtlety noted in
  the README, worth an in-app explanation eventually.
