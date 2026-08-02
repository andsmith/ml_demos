# Event Flow

## Startup

```
python rl_demo.py
  RLDemoApp.__init__
    _init_tk()            create root, bind fullscreen/quit keys
    SelectionPanel        (algorithm list, opponent slider)
    _init_alg_panels()    build HeuristicPlayer opponent -> Environment
                          instantiate current DemoAlg
                          StatusControlPanel, TabPanel, VisualizationPanel
  app.start()
    alg.start(advance_event)   -> daemon Thread(_learn_loop)
    root.mainloop()
```

## Algorithm control (Go / checkpoints / stop-states)

```
_learn_loop (algorithm thread)
   ... work ...
   _maybe_pause(control_point):
       pause needed?  (checkpoint checkbox on, or current state in app's
                       selected stop-states)
         yes: app.tick(is_paused=True)   render everything
              advance_event.wait()       block until Go
         no:  app.tick(is_paused=False)  FPS-throttled render; continue

StatusControlPanel [Go] button -> pushes run-control settings -> alg.advance()
                                   -> advance_event.set() (unblocks the loop)
[Clear Stops] -> clears selected stop-states
```

Checkpoint granularities come from `alg.get_run_control_options()` — per state update,
per sweep ("epoch"), per PI iteration, or free-run.

## Rendering tick

```
app.tick(is_paused, ...)        (CURRENTLY on the algorithm thread — M5 fixes this)
   throttle: skip if < 1/FPS since last paint (running mode only)
   StatusControlPanel.refresh_status()
   TabPanel.refresh_images()     current tab only: numpy -> PIL -> PhotoImage
   VisualizationPanel.refresh()  alg.get_viz_image()
```

## Mouse interaction

```
Tk <Motion>/<Button>/<Leave> on TabPanel
  └─> current TabContentPage.mouse_move/click/leave
        └─> MouseBoxManager (KDTree lookup: pixel -> state box)
              mouseover change -> invalidate _marked_image only
              click            -> RLDemoApp.toggle_selected_state(state)
                                  (selected states become stop-states)
        └─> TabPanel.refresh_images() re-annotates + rebuilds PhotoImage
```

## Selection panel events

```
algorithm radio / opponent slider  -> stored as *pending*
[Reset] -> _proc_reset -> RLDemoApp.change_alg / reset_state
             rebuild Environment + DemoAlg, re-create alg-dependent panels
[Save]/[Load] -> DemoAlg.save_state / load_state (pickle snapshot)
[Fullscreen]  -> toggles root fullscreen attribute
```

## Shutdown

```
q / ESC / window close -> root destroyed
DemoAlg.stop() sets _shutdown, repeatedly sets advance_event and joins the
learn thread (retry loop; see known_issues — fragile).
```
