# Posthoc Release Evaluator

`ml_worker_improved/tools/run_posthoc_release_evaluator.py` summarizes existing adaptive audio
manifests without rerunning model inference.

It is a Docker-only audit step for already generated artifacts. It answers: "Which outputs are
release-ready, which are diagnostic-only, and why?"

## Inputs

The evaluator scans an artifact root for:

- `adaptive_vocal_manifest.json` from `adaptive_vocal_isolation.py`;
- `adaptive_manifest.json` from `adaptive_minus_repair.py`.

Unsupported manifest modes are skipped.

## Rebuilt Fields

Older artifacts may not contain the newest gate fields. When enough source fields exist, the
evaluator rebuilds them in-memory:

- vocal: `detected_issue_action_contract`, `defect_effectiveness_gate`,
  `issue_effectiveness_ledger`;
- minus: `selected_minus_release_gate`.

For minus artifacts, the evaluator does not trust a persisted `release_ready` gate when the selected
candidate is missing the required release metrics. It rebuilds the gate as `diagnostic_only` with
`missing_minus_release_metrics` so older manifests cannot be promoted by stale safe defaults.

The tool does not rewrite source manifests. Rebuilt field names are reported in each summary item
under `rebuilt_fields`.

## Output Summary

The summary JSON includes:

- `release_status_counts`;
- `gate_status_counts`;
- `error_count` and `errors` for unreadable or corrupt manifests;
- selected profile and score;
- blocker labels and reasons;
- uncovered labels;
- worst vocal issue windows;
- selected/listen output paths;
- `release_status_drift` when an artifact's saved status differs from the rebuilt gate.

## Batch Delivery Integration

`run_adaptive_vocal_contract_batch.py` uses the same release status contract when it summarizes
newly generated vocal listen packs:

- `deliverable_vocals_flac` and `deliverable_listen_vocals_flac` are populated only when
  `release_status == "release_ready"`;
- `diagnostic_vocals_flac` and `diagnostic_listen_vocals_flac` keep the same files for debugging
  when the gate says `diagnostic_only`;
- `artifact_release_status` and `release_status_drift` expose mismatches between a saved artifact
  status and the gate status used by the batch summary;
- `blocker_reasons`, `blocker_labels`, `blocked_window_count`, and `worst_windows` explain why the
  listen output must not be treated as final.
- root summary counters include `release_ready_count`, `diagnostic_only_count`, and
  `release_blocked_count`.

The original selected/listen paths remain in the summary for backward compatibility, but automation
that publishes final results must use the `deliverable_*` fields.

If a batch manifest has `selected_artifacts.release_status == "release_ready"` but no
`defect_effectiveness_gate.release_status`, the batch summary treats it as `diagnostic_only` and
sets `release_status_drift`. Saved artifact status is informational; a gate is required to authorize
delivery.

## Release Delivery Manifest

`ml_worker_improved/tools/build_release_delivery_manifest.py` converts an adaptive vocal batch
summary into a compact `release_delivery_manifest.json`.

It is the consumer-facing file for automation that needs to decide what can be handed to a listener:

- `tracks[*].deliverable.*` is populated only for release-ready outputs with explicit
  `deliverable_*` fields in the batch summary;
- legacy records with only `selected_vocals_flac` or `listen_selected_vocals_flac` are treated as
  `diagnostic_only`, even if they claim `release_status == "release_ready"`;
- failed routes are reported as `delivery_status == "route_failed"`;
- `tracks[*].diagnostic.*`, `tracks[*].blockers`, and `tracks[*].blockers.worst_windows` preserve
  debugging paths and reasons.

Example:

```bash
docker compose run --rm --no-deps -T ml-worker \
  python /app/tools/build_release_delivery_manifest.py \
  /app/lab_artifacts/golden_audio/adaptive_vocal_contract_batch_summary.json \
  /app/lab_artifacts/golden_audio/release_delivery_manifest.json
```

## Contract

`release_ready` means the stored or rebuilt gate found no release blocker.

`diagnostic_only` means the artifact can be inspected for debugging, but must not be presented as a
clean/professional listening result.

When an older minus manifest lacks the required release metrics, the evaluator emits
`diagnostic_only` with `missing_minus_release_metrics` instead of treating default metric values as
success.

This evaluator uses the same proxy gates as the adaptive routes. It does not claim SDR/SI-SDR
without isolated ground-truth stems.

## Example

```bash
docker compose run --rm --no-deps -T ml-worker \
  python /app/tools/run_posthoc_release_evaluator.py \
  /app/lab_artifacts/golden_audio \
  /app/lab_artifacts/golden_audio/posthoc_release_evaluation_summary.json \
  --kind auto
```

`--limit 0` or any negative limit writes an empty summary and evaluates no manifests.
