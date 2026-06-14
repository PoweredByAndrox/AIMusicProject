# Fine-tune Candidate Unblock Plan

Phase 13 converts Phase 12 evidence-gap rows into explicit metadata-only
unblock actions. It does not execute fixes.

## Inputs

- `finetune_evidence_gap_report.jsonl`
- `finetune_evidence_gap_manifest.json`
- `finetune_candidate_preflight.jsonl`
- `root_cause_plan.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`
- `reference_path_recovery_plan.jsonl`

All inputs are metadata files. The planner rejects non-JSON/JSONL inputs before
hashing, so audio files are not read or hashed.

## Outputs

- `finetune_candidate_unblock_plan.jsonl`
- `finetune_candidate_unblock_manifest.json`
- `finetune_candidate_unblock_summary.md`

## Safety Contract

This phase must not:

- modify manifests;
- copy, read, hash, or export audio;
- build a dataset;
- build a ground-truth bundle;
- start training;
- change policy;
- relax thresholds;
- add models, weights, cleanup, debleed, or denoise stages;
- promote artifacts.

`can_start_training_now` is always `false`. Golden and holdout rows remain
protected by forbidden actions.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_finetune_candidate_unblock.py \
  --evidence-gap-report-jsonl /app/lab_artifacts/control_plane_phase12_20260528/finetune_evidence_gap_report.jsonl \
  --evidence-gap-manifest-json /app/lab_artifacts/control_plane_phase12_20260528/finetune_evidence_gap_manifest.json \
  --finetune-preflight-jsonl /app/lab_artifacts/control_plane_phase11_20260528/finetune_candidate_preflight.jsonl \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --reference-path-recovery-jsonl /app/lab_artifacts/control_plane_phase10f_20260528/reference_path_recovery_plan.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase13_20260528
```
