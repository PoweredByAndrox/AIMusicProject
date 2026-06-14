# Fine-tune Evidence Gap Report

Phase 12 is a metadata-only blocker report for rows that Phase 11 marked as
future fine-tune candidates but could not admit into dataset construction.

## Inputs

- `finetune_candidate_preflight.jsonl`
- `finetune_candidate_preflight_manifest.json`
- `root_cause_plan.jsonl`
- `autonomous_decision_queue.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`
- `reference_path_recovery_plan.jsonl`

The tool reads only JSON/JSONL metadata. It does not inspect, hash, copy, or
export audio.

## Outputs

- `finetune_evidence_gap_report.jsonl`
- `finetune_evidence_gap_manifest.json`
- `finetune_evidence_gap_summary.md`

Every report row includes an `exact_blocker`, a non-empty
`required_to_unblock`, explicit missing evidence buckets, and
`can_start_training_now=false`.

## Safety Contract

This phase must not:

- add models or weights;
- start training;
- build a dataset;
- build a ground-truth bundle;
- export audio;
- add cleanup, debleed, or denoise stages;
- relax thresholds;
- change policy;
- promote any artifact.

Holdout and golden rows remain protected and cannot be routed into training or
threshold tuning.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/build_finetune_evidence_gap_report.py \
  --finetune-preflight-jsonl /app/lab_artifacts/control_plane_phase11_20260528/finetune_candidate_preflight.jsonl \
  --finetune-preflight-manifest-json /app/lab_artifacts/control_plane_phase11_20260528/finetune_candidate_preflight_manifest.json \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --autonomous-decisions-jsonl /app/lab_artifacts/control_plane_phase7_20260528/autonomous_decision_queue.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --reference-path-recovery-jsonl /app/lab_artifacts/control_plane_phase10f_20260528/reference_path_recovery_plan.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase12_20260528
```
