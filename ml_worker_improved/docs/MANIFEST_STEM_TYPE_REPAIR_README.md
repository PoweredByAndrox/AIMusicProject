# Manifest Stem Type Repair

Phase 14A applies safe stem-type metadata repair as overlay patches only.
Original source manifests are never modified.

## Inputs

- `finetune_candidate_unblock_plan.jsonl`
- `finetune_candidate_unblock_manifest.json`
- `finetune_evidence_gap_report.jsonl`
- `finetune_candidate_preflight.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`

All inputs are metadata files. The tool rejects non-JSON/JSONL inputs before
hashing.

## Outputs

- `manifest_stem_type_repair_report.jsonl`
- `manifest_stem_type_repair_manifest.json`
- `manifest_stem_type_repair_summary.md`
- `repaired_manifest_overlay.jsonl`

`repaired_manifest_overlay.jsonl` contains patch records such as:

```json
{"overlay_patch": {"op": "add_or_replace", "path": "/stem_type", "value": "drums"}}
```

## Safety Contract

This phase must not:

- mutate source manifests directly;
- copy, read, hash, or export audio;
- build a dataset;
- build a ground-truth bundle;
- start training;
- change promotion policy;
- relax thresholds;
- add models, weights, cleanup, debleed, or denoise stages;
- promote artifacts.

`can_start_training_now` is always `false`.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/apply_manifest_stem_type_repair.py \
  --unblock-plan-jsonl /app/lab_artifacts/control_plane_phase13_20260528/finetune_candidate_unblock_plan.jsonl \
  --unblock-manifest-json /app/lab_artifacts/control_plane_phase13_20260528/finetune_candidate_unblock_manifest.json \
  --evidence-gap-report-jsonl /app/lab_artifacts/control_plane_phase12_20260528/finetune_evidence_gap_report.jsonl \
  --finetune-preflight-jsonl /app/lab_artifacts/control_plane_phase11_20260528/finetune_candidate_preflight.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase14a_20260528
```
