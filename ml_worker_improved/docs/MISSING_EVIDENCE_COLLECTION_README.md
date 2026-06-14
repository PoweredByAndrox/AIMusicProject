# Missing Evidence Collection Plan

`tools/plan_missing_evidence_collection.py` consumes Phase 8 root-cause rows and builds a
planner-only queue for cases whose safe next step is missing reference/evidence collection.

This phase does not repair audio, copy/export audio, start fine-tuning, change policy, add models,
add weights, add cleanup/debleed/denoise stages, or relax thresholds.

## Inputs

- `root_cause_plan.jsonl`
- `root_cause_plan_manifest.json`
- `autonomous_decision_queue.jsonl`
- `hard_case_benchmark.jsonl`
- `control_plane_dashboard.json`
- `golden_corpus.jsonl`

The root-cause plan is the filter source. The autonomous decision queue and benchmark are used to
restore source manifest, failed checks, window bounds, and other provenance fields. Golden rows are
used only as protected exclusions and never for training or threshold tuning.

## Outputs

- `missing_evidence_plan.jsonl`
- `missing_evidence_plan_summary.md`
- `missing_evidence_plan_manifest.json`

Each output row includes:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `source_manifest`
- `dominant_failure_reason`
- `failure_labels`
- `original_decision`
- `root_cause_bucket`
- `missing_evidence_types`
- `required_tools_to_run`
- `required_inputs`
- `can_collect_automatically`
- `can_use_for_finetune_after_collection`
- `blocked_reason`
- `forbidden_actions`

## Evidence Routing

- Missing source manifests route to `manual_source_manifest_inspection` and cannot be collected
  automatically.
- Missing stem type or window bounds route to `manifest_repair_only`.
- Missing reference metrics route to `run_ground_truth_quality_gate` or
  `waveform_forensic_compare` when reference paths already exist.
- Missing ground-truth bundle or full-mix reconstruction routes to
  `build_flstudio_ground_truth_bundle` and/or `waveform_forensic_compare`.
- Missing residual or leakage metrics route to `waveform_forensic_compare` and
  `localize_reference_quality_failures`.
- Audio paths can be referenced only when they already exist in a source manifest; the planner
  never copies or exports audio.

## Protection Rules

- Only `split=train` rows can become fine-tune candidates after evidence collection.
- Dev and holdout rows cannot become fine-tune candidates after evidence collection.
- Golden rows cannot be used for training or threshold tuning.
- Planner output is deterministic and contains only metadata/provenance.
- Rows that cannot be collected automatically must include `blocked_reason`.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_missing_evidence_collection.py \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --root-cause-manifest-json /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan_manifest.json \
  --autonomous-decisions-jsonl /app/lab_artifacts/control_plane_phase7_20260528/autonomous_decision_queue.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --dashboard-json /app/lab_artifacts/control_plane_phase2_20260528/control_plane_dashboard.json \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10a_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_missing_evidence_collection.py \
  tests/test_plan_autonomous_root_causes.py \
  tests/test_artifact_status_contract.py \
  -q
```
