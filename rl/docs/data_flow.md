# Data Flow

## 1. State space construction (startup, cached)

```
tic_tac_toe.GameTree._build_tree_recursive
    plays every possible game (agent-first and each opponent-first opening)
        ├─ _terminal: state -> Result | None
        ├─ _children: state -> {child_state: (action, mover)}
        ├─ _parents:  state -> [parent states]
        └─ _initial:  starting states
    └─> pickled whole to game_tree_X.pkl (~50 MB) by get_game_tree_cached
```

## 2. Environment / MDP construction (per algorithm reset)

```
reinforcement_base.Environment(opponent_policy, player_mark)
    game tree  +  opponent policy
        └─> opp_move_dist / extract_dynamics:
            for agent state s and action a:
              P(s' | s, a) = opponent's action distribution applied to the
                             intermediate board; terminal intermediates yield
                             reward TERM_REWARDS[player][result] directly.
```

The opponent is folded into the MDP: from the agent's point of view the environment is
stochastic, with transition probabilities given by the opponent policy.

## 3. Learning loop (PE/PI, after M3)

```
seed policy pi_0 (HeuristicPlayer)
   └─> Policy Evaluation: for each nonterminal s (repeat until |dV| < eps):
          V_{t+1}(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) [ R(s') + gamma * V_t(s') ]
   └─> Policy Improvement (policy_optim.ValueFuncPolicy):
          pi_{k+1}(s) = argmax_a  E_{s'} [ R(s') + gamma * V(s') ]   (ties uniform)
   └─> repeat until pi_{k+1} == pi_k
```

Data structures: `V` and `V_new` are plain dicts `Game -> float`; terminal values are
fixed to `TERM_REWARDS[player][result]`. Policies map `Game -> [(action, prob)]`.

## 4. Rendering data flow

```
V(s), dV(s), pi changes                      (algorithm thread)
   └─> DemoAlg.get_status()          -> StatusControlPanel text lines
   └─> DemoAlg.get_viz_image()       -> VisualizationPanel (step visualization)
   └─> ValueFunctionContentPage.set_value(s, v)
          └─> ColorKey value->color  -> BoxOrganizer.draw_box (single box repaint)
                (full-base invalidation only if the self-adjusting color range grows)

TabContentPage image layers:
   _base_image   (all state icons / colored boxes; expensive, cached)
     └─ _marked_image  (base copy + selection/mouseover highlights)
         └─ _disp_image (final; converted numpy -> PIL -> ImageTk.PhotoImage)
```

## 5. Evaluation / results

```
gameplay.Match(policy_A, policy_B).play() -> trace (states, actions, result)
gameplay.ResultSet.add(trace) ... -> win/draw/loss counts
   ├─> result_viz.PolicyEvaluationResultViz (stats band + sample games)
   └─> plot_trace.TraceArtist (single-trace column render)
```

## 6. Persistence

| Artifact | Producer | Notes |
|---|---|---|
| `game_tree_X.pkl` / `game_tree_O.pkl` | `get_game_tree_cached` | full GameTree pickle |
| `minimax_policy_cache_{X,O}.pkl` | `MiniMaxPolicy` | optimal policy cache |
| algorithm save/load pickles | `DemoAlg.save_state`/`load_state` | type-marked snapshots via Save/Load buttons |
