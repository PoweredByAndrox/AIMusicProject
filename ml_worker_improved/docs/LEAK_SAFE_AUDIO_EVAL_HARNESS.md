# Leak-Safe Audio Fingerprint Evaluation Harness

EVAL-1 adds a strict evaluation split layer before tensor critics, repair adapters, or separator fine-tuning.

The harness is artifact-only:

- no critic training
- no separator training
- no repair adapter training
- no runtime gate changes
- no release-ready output
- no publishable result

## Command

```bash
docker compose run --rm ml-worker python tools/build_leak_safe_audio_eval_harness.py \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --model-dir /app/lab_artifacts/audio_error_critic/model_2c_reliable_target_critics_001 \
  --output-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001
```

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_leak_safe_audio_eval_harness.py \
  --output-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001
```

## Outputs

- `leak_safe_split_manifest.json`
- `leak_safe_split_rows.jsonl`
- `audio_fingerprint_index.jsonl`
- `leakage_report.json`
- `hardcase_holdout_manifest.json`
- `model_2c_locked_re_eval.json`
- `locked_eval_summary.md`
- `leak_safe_eval_progress.jsonl`

## Leakage Checks

Critical leakage:

- `track_id` overlap across splits
- input audio fingerprint overlap across splits
- `candidate_id` overlap across splits
- same candidate in train/test
- candidate exact audio hash overlap
- candidate near-duplicate audio hash overlap
- overlapping windows for the same track/candidate/stem across splits

Warning leakage:

- `route_id` overlap
- candidate family overlap
- source model family overlap

Route/model family overlap is reported as warning by default because current artifacts may use the same separator family across all tracks. Future stricter experiments can promote those checks to hard blockers.

## Hard-Case Holdout

Rows selected into the deterministic hard-case holdout from severe blockers such as `severe_phase_collapse` or `severe_vocal_bleed_in_minus` are locked into eval/holdout use:

- `locked=true`
- `train_allowed=false`
- `release_ready=false`

Future R1/R2/R3/R4 work must consume this split instead of rebuilding its own ad hoc train/test partition.

Not every hard negative is forced into holdout. Tensor/audio critics need leak-safe hard negatives in train, so non-holdout hard negatives may remain trainable while the locked holdout remains unavailable for training.

## MODEL-2C Locked Re-Eval

If `--model-dir` is provided, the harness loads existing MODEL-2C shadow critic artifacts and evaluates only rows assigned to the locked `test` split. This is diagnostic scoring only:

- `critic_training_started=false`
- `training_started=false`
- `separator_training_started=false`
- `release_ready=false`

If no model directory is provided, the re-eval section is written with `status=skipped_missing_model_dir`.
