# Architecture

## Overview

This project is an educational Tic-Tac-Toe reinforcement-learning demo. It renders the
entire game-state space (~8.5k states from player X's perspective) as an image of small
board icons, and animates RL algorithms (value updates, policy changes) over that image
while the user steps through the algorithm at several granularities.

The codebase contains **two generations of the application**:

1. **Old app (being retired):** `game_learn.py` + `gui_components.py`. A monolithic
   window (`RLDemoWindow`) driven directly by `PolicyEvaluationPIDemo`. Its algorithm
   math is correct; its GUI is superseded. Scheduled for removal once the math is fully
   ported (see `future_work.md` / roadmap M6).
2. **New app (Version 1 target):** `rl_demo.py` + a pluggable panel/algorithm
   architecture. This is the app all current work targets.

## New app layers

```
rl_demo.py (RLDemoApp)                  -- Tk root, app lifecycle, tick/render entry
 ├─ selection_panel.SelectionPanel      -- choose algorithm, opponent difficulty, reset
 ├─ status_ctrl_panel.StatusControlPanel-- status lines, run-control checkpoints, Go
 ├─ alg_panels.TabPanel                 -- ttk.Notebook of TabContentPage pages
 │    └─ tab_content.TabContentPage     -- cached-image interactive pages
 │         └─ state_tab_content.*       -- full state embedding / value-function pages
 └─ alg_panels.VisualizationPanel       -- current step-visualization image

rl_alg_base.DemoAlg (ABC)               -- algorithm base: thread, pause points, tabs
 ├─ policy_eval.PolicyEvalDemoAlg       -- Iterative PE + PI (math being ported)
 ├─ policy_eval.InPlacePEDemoAlg        -- in-place variant
 ├─ dynamic_prog.*                      -- DP demos (stubs, V1 scope)
 ├─ q_learning.*, policy_grad.py        -- stubs (future work)
 └─ monte_carlo.py                      -- empty (future work)

reinforcement_base.Environment          -- MDP built from game tree + opponent policy
tic_tac_toe.GameTree / Game             -- exhaustive state space, cached in pickle
policies / baseline_players / perfect_player -- Policy ABC, heuristic + minimax players
gameplay.Match / ResultSet              -- play games, accumulate + render results
```

## Rendering pipeline

All drawing is done with OpenCV primitives into numpy RGB arrays; Tk only displays
finished frames.

```
drawing.GameStateArtist        -- one board icon at 3 detail tiers (by pixel size)
node_placement.BoxOrganizer    -- per-state pixel bounding boxes, layer layout
state_embedding.StateEmbedding -- states -> 6 layers (by move count) -> box placer
layer_optimizer.SimpleTreeOptimizer -- reorder states within a layer into zones
                                       (X-wins left, draws middle, O-wins right)
tab_content.TabContentPage     -- caches base/marked/display image layers
alg_panels                     -- numpy -> PIL -> ImageTk.PhotoImage -> tk.Label
```

Step visualizations (per-state Bellman-update trees, epoch histograms) come from
`step_visualizer.py` (working) with a half-finished successor in `tree_step_viz.py`
(parked; see `known_issues.md`).

## Threading model

`DemoAlg` runs its learn loop on a daemon thread (`rl_alg_base.py`). At each control
point it calls `_maybe_pause(cp)`; if the matching checkbox (or a selected stop-state)
is active it renders via `app.tick(is_paused=True)` and blocks on a `threading.Event`
until the Go button sets it. Otherwise it ticks (FPS-throttled) and continues.

**Known defect:** `tick()` currently updates Tk widgets from the algorithm thread.
Fixing this (marshal to the main thread via `root.after`/queue) is roadmap milestone M5.

## State space and caching

`tic_tac_toe.GameTree` exhaustively enumerates every game from both first-player
perspectives and stores terminal status, children (with the action and mover), parents,
and initial states. The whole object is pickled to `game_tree_X.pkl` (~50 MB) on first
run. `perfect_player.MiniMaxPolicy` similarly caches to `minimax_policy_cache_{X,O}.pkl`.

See also: `component_inventory.md` (per-file detail), `data_flow.md`, `event_flow.md`.
