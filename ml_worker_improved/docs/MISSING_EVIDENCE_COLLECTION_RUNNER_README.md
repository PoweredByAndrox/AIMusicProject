# Missing Evidence Collection Runner

`tools/run_missing_evidence_collection.py` consumes the Phase 10A
`missing_evidence_plan.jsonl` and processes only rows where
`can_collect_automatically=true`.

This runner collects internal evidence only. It does not promote artifacts, export audio as a user
result, start training, change policy, change thresholds, add model candidates, add weights, or run
cleanup/debleed/denoise stages.

## Inputs

- `missing_evidence_plan.jsonl`
- `missing_evidence_plan_manifest.json`
- `root_cause_plan.jsonl`
- `autonomous_decision_queue.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`

Blocked rows from the plan are not processed. They are copied to
`failed_evidence_collection.jsonl` with `collection_status=skipped_blocked`.

## Outputs

- `collected_evidence.jsonl`
- `collected_evidence_summary.md`
- `collected_evidence_manifest.json`
- `failed_evidence_collection.jsonl`

Every collected row records:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `source_manifest`
- `tools_requested`
- `tools_executed`
- `tools_skipped`
- `evidence_collected`
- `evidence_missing_after_run`
- `collection_status`
- `can_reclassify_after_collection`
- `can_enter_finetune_pool_after_collection`
- `can_promote_after_collection=false`
- `audio_exported=false`
- `training_started=false`
- `policy_changed=false`

## Execution Policy

- `stem_quality_control` is executed only through read-only metric functions. The CLI path is not
  used because it can write or copy audio.
- `waveform_forensic_compare` is executed only when both candidate audio and reference audio already
  exist in manifest provenance. The CLI writer is not used because it can emit residual audio.
- `localize_reference_quality_failures` and `run_ground_truth_quality_gate` require existing
  ground-truth/reference manifest inputs.
- `build_flstudio_ground_truth_bundle` is skipped in this phase because it writes bundle FLACs and
  needs full bundle inputs; the runner records the skip reason instead.
- Missing inputs skip the specific tool and do not crash the run.

## Protection Rules

- Rows with `can_collect_automatically=false` are never processed.
- Holdout rows cannot enter the fine-tune pool after collection.
- Golden rows cannot enter training or threshold tuning.
- `can_promote_after_collection` is always false.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/run_missing_evidence_collection.py \
  --missing-evidence-plan-jsonl /app/lab_artifacts/control_plane_phase10a_20260528/missing_evidence_plan.jsonl \
  --missing-evidence-manifest-json /app/lab_artifacts/control_plane_phase10a_20260528/missing_evidence_plan_manifest.json \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --autonomous-decisions-jsonl /app/lab_artifacts/control_plane_phase7_20260528/autonomous_decision_queue.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10b_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_run_missing_evidence_collection.py \
  tests/test_plan_missing_evidence_collection.py \
  tests/test_artifact_status_contract.py \
  -q
```
