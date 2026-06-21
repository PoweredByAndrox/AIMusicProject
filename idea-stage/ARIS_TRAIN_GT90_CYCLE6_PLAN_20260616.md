# ARIS TRAIN GT90 Cycle 6 Plan

## Checkpoint

V6 evaluated explicit complex-mask shared vocal adapters on the locked
`nothing-can-wait` holdout. The best config was `complex_wide_context`:

- candidate SDR: `0.303320 dB`
- improvement over source: `+8.077691 dB`
- status: failed, `holdout_metric_below_threshold`
- train-ready decision: `success_rate=0.0`

The route was observable and used CUDA. It did not hang, did not publish audio,
and did not mark release-ready.

## Failure Pattern

Magnitude-only and complex-only shared MLP adapters improve the source baseline
but remain far below the absolute SDR gate. The failure is not a runtime
observability issue anymore; it is a model capacity/generalization issue on a
small verified vocal corpus with only two non-holdout track families.

## Next Route

Evaluate explicit complex source-feature memory:

- `mask_output_mode=complex`
- `source_feature_memory_neighbors>0`
- deterministic projection enabled
- deterministic frame cap enabled
- complex frame masks stored as checkpoint metadata
- KNN search remains guarded by `max_source_feature_memory_distance_ops`

This is not a fallback. It is an explicit matrix candidate with provenance,
diagnostics, and holdout gating. It cannot set `delivery_allowed=true` or
`release_ready=true`.

## V7 Matrix

Run three configs:

- `complex_memory_one_per_track`
- `complex_memory_two_per_track`
- `complex_memory_wide_context`

All configs use locked holdout `nothing-can-wait`, `device=cuda`, unchanged
thresholds, and no registry edits.

## Safety

- fallback_used=false
- release_ready=false
- delivery_allowed=false
- thresholds_relaxed=false
- model_registry_edited=false
- no new separator model
- no dirty stems as ground truth
