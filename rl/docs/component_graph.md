# Component Graph

Import/ownership relationships for the Version-1 (new) app. Arrows point from user to
used component.

```
rl_demo.RLDemoApp
  ├──> selection_panel.SelectionPanel ──> gui_base.Panel
  ├──> status_ctrl_panel.StatusControlPanel ──> gui_base.Panel
  ├──> alg_panels.TabPanel ──> gui_base.Panel
  │       └──> tab_content.TabContentPage (per tab, from alg.get_tabs())
  │               ├──> mouse_state_manager.MouseBoxManager
  │               ├──> state_embedding.StateEmbedding      (state pages)
  │               │       ├──> node_placement.FixedCellBoxOrganizer
  │               │       ├──> layer_optimizer.SimpleTreeOptimizer
  │               │       └──> drawing.GameStateArtist
  │               └──> color_key.* (value pages)
  ├──> alg_panels.VisualizationPanel ──> gui_base.Panel
  ├──> reinforcement_base.Environment
  │       ├──> tic_tac_toe.GameTree (via get_game_tree_cached)
  │       └──> policies.Policy (opponent: baseline_players.HeuristicPlayer)
  └──> rl_alg_base.DemoAlg subclasses (ALGORITHMS registry)
          ├──> policy_eval.PolicyEvalDemoAlg / InPlacePEDemoAlg
          │       ├──> policy_optim.ValueFuncPolicy   (after M3)
          │       └──> step_visualizer.PEStep hierarchy
          ├──> dynamic_prog.* (M4)
          └──> q_learning / policy_grad / monte_carlo (stubs)

Shared foundations (used nearly everywhere):
  layout.LAYOUT / WIN_SIZE, colors.COLOR_SCHEME,
  game_base (Mark, Result, TERM_REWARDS, get_reward),
  tic_tac_toe.Game, drawing.GameStateArtist

Evaluation / results path:
  gameplay.Match ──> policies.Policy implementations
  gameplay.ResultSet ──> result_viz.PolicyEvaluationResultViz
  plot_trace.TraceArtist, tiny_histogram, micro_histogram, slope_diagram
```

## Standalone apps (no dependency on rl_demo)

```
game_graph.py        full-tree browser (cv2)  ──> tic_tac_toe, drawing
tic_tac_toe.py       CLI tree generator/statistics
perfect_player.py    minimax cache builder (CLI)
evolve_feedforward.py / backprop_net.py   NEAT-backprop experiment ──> visualize.py, neat_util.py
```

## Legacy (retire in M6)

```
game_learn.PolicyImprovementDemo ──> gui_components.RLDemoWindow
                                 ──> policy_optim.ValueFuncPolicy
                                 ──> step_visualizer
test_panels.py (dead)
```
