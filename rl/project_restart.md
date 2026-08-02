````markdown
# project_restart.md

# Project Restart Instructions

This document defines the process for restarting and stabilizing this project.

The project is a reinforcement learning visualization and education demo centered on Tic-Tac-Toe. The immediate objective is **not** to expand its scope, but to understand, repair, simplify, and complete the implementation that already exists while establishing a solid architectural foundation for future growth.

The long-term vision is to become a high-quality educational environment capable of visualizing many reinforcement learning algorithms at multiple levels of detail. However, **new algorithms are not part of the current implementation phase.**

---

# Session Start

Begin work by entering Plan Mode.

The entry command for this repository is:

> /plan Begin the work in project_restart.md

Assume the remainder of this document defines the project goals and workflow.

---

# Overall Goals

Primary objectives, in order:

1. Understand the existing project completely.
2. Reconstruct the intended architecture.
3. Repair and complete the existing implementation.
4. Improve performance and responsiveness.
5. Simplify the architecture where appropriate.
6. Make the implementation cleanly extensible.
7. Produce the documentation that should have existed from the beginning.

Do **not** begin by adding features.

Do **not** begin by adding additional RL algorithms.

The first objective is understanding.

---

# Design Philosophy

Treat yourself as the lead engineer inheriting an existing codebase.

Your first responsibility is understanding the project before proposing architectural changes.

Avoid rewrites unless they clearly reduce complexity while preserving behavior.

When project intent is unclear, identify the uncertainty and ask questions instead of making assumptions.

Prefer incremental improvements over large redesigns.

---

# Current Implementation Goals

The implementation phase focuses on producing a polished Version 1.

This includes:

- Completing partially implemented RL algorithms.
- Repairing existing algorithms.
- Repairing bugs.
- Improving runtime performance.
- Improving rendering efficiency.
- Improving GUI responsiveness.
- Improving interaction between visualization and learning algorithms.
- Eliminating unnecessary complexity.
- Preserving existing functionality whenever practical.

This phase explicitly excludes adding additional RL algorithms.

---

# Long-Term Direction

While implementing Version 1, continually keep future extensibility in mind.

Architectural decisions should make it straightforward to later add:

- additional RL algorithms
- additional visualizations
- additional debugging views
- additional educational tooling

Document opportunities for future expansion, but do not implement them during the Version 1 effort.

---

# Work Phases

Complete each phase before moving to the next.

Do not skip ahead.

---

## Phase 1 — Understand the Project

Study the project before modifying it.

Build a mental model of:

- overall architecture
- subsystem responsibilities
- execution flow
- rendering pipeline
- simulation loop
- GUI interaction
- RL algorithm interaction
- data flow
- event flow

Compare the implementation with the README and identify where they differ.

Produce questions whenever project intent is ambiguous.

---

## Phase 2 — Inventory the Codebase

Systematically inspect the repository.

For each major directory or subsystem identify:

- purpose
- responsibilities
- important classes
- important functions
- dependencies
- interactions
- undocumented behavior
- possible design issues

---

## Phase 3 — Reconstruct Documentation

Create or update documentation so the repository accurately reflects the implementation.

Documentation should become the primary source of architectural understanding.

Create and maintain at least the following documents:

```
docs/
    architecture.md
    component_inventory.md
    component_graph.md
    data_flow.md
    event_flow.md
    design_decisions.md
    coding_conventions.md
    future_work.md
    known_issues.md
```

Also maintain:

```
CURRENT_STATE.md
```

CURRENT_STATE.md should remain concise and answer only:

- What currently works?
- What is incomplete?
- What is known to be broken?
- Highest-priority remaining work.
- Immediate next milestone.

This document should always represent the current status of the project.

---

## Phase 4 — Identify Technical Debt

Before implementing fixes, identify and prioritize:

- bugs
- architectural problems
- maintainability problems
- performance bottlenecks
- unnecessary complexity
- dead code
- abandoned ideas

Prioritize by impact.

---

## Phase 5 — Produce an Implementation Roadmap

Create an implementation roadmap focused exclusively on Version 1.

Break work into small milestones.

Each milestone should describe:

- objective
- affected components
- expected risks
- validation/testing approach

Only one milestone should be actively worked on at a time.

---

## Phase 6 — Implementation

Once planning is complete:

- implement one milestone
- verify correctness
- update documentation
- update CURRENT_STATE.md
- then proceed to the next milestone

Avoid working on multiple unrelated objectives simultaneously.

---

# Working Style

Remain focused on the current milestone.

Prefer understanding over implementation.

Prefer documentation over assumptions.

Prefer small, reviewable changes over large refactors.

Keep architecture and documentation synchronized with the codebase.

Whenever a design decision is made, record it in `docs/design_decisions.md`.

Whenever project status changes, update `CURRENT_STATE.md`.

The repository should gradually become self-documenting as work progresses.

---

# Success Criteria

A successful Version 1 should:

- run efficiently
- be responsive
- be architecturally understandable
- have complete supporting documentation
- faithfully demonstrate the existing reinforcement learning algorithms
- provide a clean foundation for future expansion
- leave the repository in a state where future work is straightforward rather than exploratory
````
