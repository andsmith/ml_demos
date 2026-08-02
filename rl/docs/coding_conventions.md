# Coding Conventions

Conventions observed in the codebase and to be followed for new work.

## Style

- Python 3.12, 4-space indent, `snake_case` functions/variables, `CamelCase` classes,
  leading-underscore for private attributes/methods.
- Docstrings use `:param:`/`:returns:` reST-ish style.
- `logging` for progress/info output. Bare `print()` is debug residue — remove it, do
  not add more.
- Module-level constants in CAPS; shared layout/color constants live only in
  `layout.py` (`LAYOUT`, `WIN_SIZE`) and `colors.py` (`COLOR_SCHEME`, `UI_COLORS`).
  New magic numbers for sizes/fonts/spacing belong in `LAYOUT`, not in module locals.

## Images and drawing

- Images are numpy `uint8` RGB arrays shaped `(H, W, 3)`; sizes are passed as
  `(width, height)` tuples. Convert to BGR only at the `cv2.imshow` boundary
  (`img[:, :, ::-1]`).
- Draw with cv2 primitives; use `SHIFT_BITS`/`SHIFT_MUL` from `layout.py` for
  sub-pixel accuracy on lines/circles.
- Expensive images are cached and invalidated explicitly (see `TabContentPage`'s
  base/marked/display layering). Never redraw the full state embedding in a
  per-mouse-event or per-frame path.

## Architecture rules

- New demo algorithms subclass `rl_alg_base.DemoAlg`, register in
  `rl_demo.ALGORITHMS`, and report `is_stub()` honestly (stubs render greyed-out).
- New panels subclass `gui_base.Panel` (or `alg_panels.AlgDepPanel` if they depend on
  the current algorithm); geometry comes from `LAYOUT['frames']`.
- Tab pages subclass `tab_content.TabContentPage`; hit-testing goes through
  `MouseBoxManager` (KDTree) — no linear scans over boxes.
- Game/state primitives (`Game`, `Mark`, `Result`, rewards) come from
  `tic_tac_toe.py`/`game_base.py`; policies implement `policies.Policy` and
  `recommend_action` MUST return a full `[(action, prob)]` distribution, never a bare
  action.
- Tk widgets must only be touched from the main thread (post-M5; use the marshalling
  helper rather than calling widget methods from worker threads).

## Testing

- Modules carry a `test_*()` harness guarded by `if __name__ == "__main__":` that
  opens a cv2 window or runs a quick check. Keep these runnable — they are the de
  facto test suite. Scriptable (non-GUI) verification preferred for algorithm math
  (e.g. converged-policy-never-loses checks against `MiniMaxPolicy`).

## Repo hygiene

- Do not commit generated artifacts: `*.pkl` caches, `*.png` scratch output,
  `__pycache__`. (Several are already tracked; do not add more.)
- Update `CURRENT_STATE.md` and the relevant `docs/*.md` in the same commit as any
  change that affects status or architecture.
