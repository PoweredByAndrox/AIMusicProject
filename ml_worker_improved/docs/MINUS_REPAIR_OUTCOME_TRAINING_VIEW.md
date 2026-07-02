# Minus Repair Outcome Training View

`tools/build_minus_repair_outcome_training_view.py` builds a metadata-only
training/evidence view from failed or accepted minus repair attempts.

It does not render audio, publish listen files, relax gates, train a separator,
train a model, or mark any candidate as release-ready.

## Inputs

The builder accepts existing source-of-truth repair artifacts:

- `--ralph-manifests`: `ralph_minus_improvement_manifest.json` files produced
  by `tools/run_ralph_minus_improvement_loop.py`;
- `--matrix-rows`: JSONL sweep/matrix rows, for example focus-strength sweeps,
  guide-blend sweeps, historical dedicated-minus candidate matrices, and
  level-normalized historical matrices.
- `--listen-preview-manifests`: accepted
  `no_reference_minus_listen_preview_manifest.json` files. Only rows whose
  manifest has `delivery_allowed=true`, `delivery_status=accepted`, and
  `presentation_status=accepted_for_listen_pack` become positive repair-outcome
  examples. Blocked manifests never become positive labels.

This layer is intentionally separate from `audio_error_training_views.v1`.
Target audio-error critics learn whether a candidate/window is acceptable.
The repair-outcome view learns whether a repair action improved the minus
without causing detail dips, overboost, or persistent vocal leakage.

## Outputs

The tool writes short Windows-safe artifacts:

```text
manifest.json
rows.jsonl
summary.md
progress.jsonl
```

Each row records:

- source artifact path and source row index;
- `source_kind` (`ralph_manifest_candidate`, `matrix_row`, or
  `listen_preview_manifest_row`);
- source/base candidate id when available;
- repair candidate/route id and route family;
- repair outcome `status` and strict `label` (`1` accepted/release-ready,
  `0` rejected/blocked);
- blocker labels;
- grouped blocker flags for detail damage and leakage;
- scalar tradeoff features:
  - vocal improvement;
  - peak correlation improvement;
  - RMS/presence/air dips;
  - RMS/presence/air boosts;
  - focus-window improvement and focus RMS damage when available;
- route parameters such as strength, alpha, or scale.

Rows without a strict outcome label or metric evidence are kept as diagnostic
rows with `safe_for_repair_outcome_training=false`.
Routing-only statuses such as `passed_for_candidate_routing` are also
diagnostic-only: they authorize another candidate route, not a successful repair
outcome, and are not positive labels.

`can_train_repair_outcome_model` is intentionally conservative. It requires at
least 10 safe rows, 3 positive repair-outcome rows, and 3 negative rows. A single
accepted listen-preview hardcase is useful evidence, but it is not enough to
declare the repair-outcome model training-ready.

## Safety

All outputs keep:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `presentation_status=repair_outcome_view_not_for_listen_pack`
- `release_ready=false`
- `separator_training_started=false`
- `model_training_started=false`
- `model_registry_edited=false`
- `thresholds_relaxed=false`
- `runtime_gates_changed=false`
- `audio_written=false`

This view may feed a future repair-action router or model-level training stage.
It is not a delivery gate and cannot make rejected diagnostic audio presentable.
Accepted listen-preview rows are used only as training/evidence labels for the
repair-outcome critic; the view itself still writes no audio and keeps
`delivery_allowed=false`.

## Current Hardcase Example

The current hardcase batch can be materialized as metadata:

```bash
docker compose run --rm --no-deps -T \
  -v "D:\AIMusicProject\artifacts:/host_artifacts" \
  ml-worker python /app/tools/build_minus_repair_outcome_training_view.py \
  --output-dir /host_artifacts/loop/it075/mro/r001 \
  --ralph-manifests /host_artifacts/loop/it075/ralph/r006/ralph_minus_improvement_manifest.json \
  --matrix-rows \
    /host_artifacts/loop/it075/fswp/r001/rows.jsonl \
    /host_artifacts/loop/it075/gblend/r001/rows.jsonl \
    /host_artifacts/loop/it075/hist/r002/rows.jsonl \
    /host_artifacts/loop/it075/hnorm/r001/rows.jsonl \
  --listen-preview-manifests \
    /host_artifacts/loop/it076/deliv/r001/no_reference_minus_listen_preview_manifest.json \
  --short-output-names
```

Expected interpretation for the current batch: the accepted listen-preview
candidate can contribute one positive repair-outcome row, while rejected
RALPH/matrix attempts remain negative evidence. No listen result is produced by
this builder and no candidate becomes releasable through this view.
