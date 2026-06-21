# ARIS TRAIN >90 Cycle 4 Plan

## Source-of-truth checkpoint

- Previous high-KNN holdout was rejected by `source_feature_memory_cost_guard`.
- Bounded projected memory v4 completed on CUDA and did not hang.
- v4 failed quality on `nothing-can-wait` holdout:
  - `candidate_sdr_db=-0.681043`
  - `improvement_db=7.093329`
  - failed check: `sdr_below_threshold`
  - `release_ready=false`
  - `delivery_allowed=false`
- Current vocal corpus has 8 vocal examples across 3 track IDs:
  - 6 examples from `Square a Saw - Oxygen (Tevlo Vocals Resynthesized)`
  - 1 example from `nothing-can-wait`
  - 1 example from the Hotstepper/Evian project-owned track hash
- When `nothing-can-wait` is held out, training sees 7 vocal rows from only 2 track families.

## Brainstorm decision

Reject:
- Raising runtime timeout: v4 is no longer a timeout problem.
- Relaxing thresholds: forbidden and would hide quality failure.
- Reusing rejected stems as success: forbidden.
- Per-track fix for `nothing-can-wait`: forbidden.

Implement next:
- Run a small reproducible matrix focused on track-family balance and bounded memory.
- Compare no-memory and projected-memory variants under the same holdout gate.
- Keep CUDA required and keep all artifacts blocked from release/listen delivery.

## Matrix intent

The dominant risk is overfitting to one track family. The next matrix varies:

- `max_examples_per_track`
- source-feature memory on/off
- context radius
- hidden size
- active-frame oversampling

No threshold, registry, release, or runtime-gate changes are allowed.

## Success contract

- A config can pass only through holdout validation.
- `success_rate` must later be evaluated by train-ready decision tooling.
- `release_ready=false` for every artifact.
- If all variants fail, the next route is a broader verified-reference corpus expansion or a different train architecture, not fallback output.
