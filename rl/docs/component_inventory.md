# Component Inventory

Per-file responsibilities. Status: **active** (V1), **legacy** (retire after migration),
**stub** (declared but unimplemented), **wip** (in-flight, incomplete), **util/test**.

## Application / GUI

| File | Status | Purpose |
|---|---|---|
| `rl_demo.py` | active | Entry point. `RLDemoApp`: Tk root, panel construction, algorithm lifecycle (`change_alg`, `save_state`/`load_state`, `reset_state`), selected-state list, `tick()` render entry. `ALGORITHMS` registry. |
| `gui_base.py` | active | `Panel(ABC)`: placed tk.Frame from relative bbox in `layout.py`, resize binding, `change_algorithm`. Also `Key`/`TextKey` cv2 legend elements + `KeySizeTester`. |
| `alg_panels.py` | active | `AlgDepPanel` ABC; `TabPanel` (Notebook of tab pages, routes mouse events, numpy→PhotoImage in `refresh_images`); `VisualizationPanel` (shows `alg.get_viz_image()`). |
| `selection_panel.py` | active | Algorithm radio buttons (stubs disabled), opponent-difficulty slider, Save/Load/Reset/Fullscreen. Changes are pending until Reset. |
| `status_ctrl_panel.py` | active | Status lines from `alg.get_status()`, checkpoint checkboxes from `alg.get_run_control_options()`, Clear Stops / Go. |
| `tab_content.py` | active | `TabContentPage(ABC)`: three cached image layers (base/marked/display), selective invalidation on mouse events, owns `MouseBoxManager`. TODO in file: rename to MouseContentPage. |
| `state_tab_content.py` | active | `FullStateContentPage` (all states as icons); `ValueFunctionContentPage` (boxes colored by value, incremental `set_value` redraw). |
| `mouse_state_manager.py` | active | `MouseBoxManager`: KDTree hit-testing over box centers, mouseover/selection tracking, highlight drawing. |
| `layout.py`, `colors.py` | active | `LAYOUT` (frame bboxes, fonts, space sizes), `WIN_SIZE`, `COLOR_SCHEME`. |
(Removed in M6: `gui_components.py` — pre-refactor monolithic window; `test_panels.py` — fully commented out.)

## Algorithms / game logic

| File | Status | Purpose |
|---|---|---|
| `game_base.py` | active | `Mark`, `Result`, `OTHER_GUY`, terminal reward tables (`TERM_REWARDS[player][result]`), `get_reward()`. |
| `tic_tac_toe.py` | active | `Game` (3x3 int8 board, hashable), `GameTree` (exhaustive enumeration; terminal/children/parents dicts), `get_game_tree_cached` (pickle cache). |
| `reinforcement_base.py` | active | `Environment`: folds opponent policy into MDP dynamics (`opp_move_dist`, `extract_dynamics`); terminal/nonterminal state access. `PIPhases`. |
| `rl_alg_base.py` | active | `DemoAlg(ABC)`: learn-loop thread, `_maybe_pause` control points, tabs registry, save/load, `is_stub()`. |
| `policy_eval.py` | active | `PolicyEvalDemoAlg` / `InPlacePEDemoAlg`: iterative policy evaluation + greedy policy improvement (real Bellman math since M3; in-place variant is Gauss-Seidel). |
| `policy_optim.py` | active | `ValueFuncPolicy`: greedy policy improvement over a value function (argmax expected reward + discounted value, uniform ties). |
| `policies.py` | active | `Policy(ABC)`: `recommend_action(state)` returns action distribution; `InvPolicy` (opponent-policy wrapper via board inversion). |
| `baseline_players.py` | active | `HeuristicPlayer(n_rules)` rule ladder (win/block/center/...), `RandomPlayer` (bug: returns bare action, fix in M6). |
| `perfect_player.py` | active | `MiniMaxPolicy`: memoized minimax, pickle-cached per player. |
| `gameplay.py` | active | `Match` (one game + trace), `ResultSet` (stats + cv2 results rendering). |
| `dynamic_prog.py` | active | `DynamicProgDemoAlg` (backward-induction DP, exact in one pass) and `InPlaceDPDemoAlg` (async value iteration); optimality backups over the PE machinery (M4). |
| `q_learning.py`, `policy_grad.py` | stub | Name-only. Future work. |
| `monte_carlo.py` | stub | Empty file. Future work. |
| `game_util.py`, `util.py` | util | `sort_states_into_layers`, `get_box_placer`, `get_state_icons`, misc helpers. |

## Visualization

| File | Status | Purpose |
|---|---|---|
| `drawing.py` | active | `GameStateArtist` (board icon at 3 detail tiers, sub-pixel cv2 lines), `get_action_dist_image`, `place_string`. |
| `node_placement.py` | active | `BoxOrganizer` ABC; `FixedCellBoxOrganizer`, `FixedCellWithKey`, `LayerwiseBoxOrganizer`. `CompactBoxOrganizer` is a **wip stub** (parked). |
| `state_embedding.py` | active | `StateEmbedding`: layer sorting, box placer per window size (cached), tree-optimizer reorder. `StateEmbeddingKey` legend (broken import, see known_issues). |
| `layer_optimizer.py` | active | `SimpleTreeOptimizer`: deterministic zone bucketing (docstring promises simulated annealing that was never built). |
| `step_visualizer.py` | active | `PEStep` hierarchy: `StateUpdateStep` (per-state update tree), `EpochStep` (delta-V + histograms). `PIStep`/`ContinuousStep` stubs. |
| `tree_step_viz.py` | wip | `CaptionedTile` (works) + `ValFuncViz` (unfinished successor to StateUpdateStep). Parked; see known_issues. |
| `slope_diagram.py` | active | `SlopeDiagram`: V(s) change diagram between iterations, flanked by `MicroHist`s. |
| `micro_histogram.py` | active | `MicroHist`: cv2 line-histogram ('bins' mode unimplemented). |
| `tiny_histogram.py` | active | `TinyHistogram`/`MultiHistogram`: matplotlib-rendered histograms pasted into frames. |
| `plot_trace.py` | active | `TraceArtist`: renders one game trace as a column of state/action images. |
| `plot_util.py` | util | Scaling, tick placement, vectorized alpha-line drawing. |
| `color_key.py` | active | `ColorKey`, `ProbabilityColorKey`, `SelfAdjustingColorKey` legends. |
| `result_viz.py` | active | `PolicyEvaluationResultViz`: W/D/L stats band + sample-game rendering. |
| `game_graph.py` | active (standalone) | Full-tree browser app (cv2 window, hardcoded 1920x1080). Independent of rl_demo. |
| `colormaps.py` | util | Dev scratch utility for colormap variants; not imported by the app. |
| `visualize.py`, `neat_util.py`, `backprop_net.py`, `evolve_feedforward.py`, `baseline vs NEAT scripts` | separate | NEAT/backprop experiment (finalized per commit 2cf9145). Not part of the demo GUI; integration is deferred future work. |
| `loop_timing/` | vendored | Loop profiler utility (own git repo). |
