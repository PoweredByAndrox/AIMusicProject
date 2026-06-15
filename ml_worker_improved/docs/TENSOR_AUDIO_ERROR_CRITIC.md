# Tensor Audio Error Critic

R1 trains a shadow-only tensor critic/localizer from no-reference audio error maps.

The route is tensor-only:

- requires `.npz` error maps
- consumes the EVAL-1 locked split
- does not use scalar MODEL-2C fallback features
- does not train separator weights
- does not train a repair adapter
- does not change runtime gates or thresholds
- never marks `release_ready`

## Build Tensor Dataset

```bash
docker compose run --rm ml-worker python tools/build_locked_tensor_audio_error_dataset.py \
  --split-dir /app/lab_artifacts/audio_error_grounding/leak_safe_eval_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001
```

Outputs:

- `locked_tensor_audio_error_dataset_manifest.json`
- `tensor_audio_error_dataset.jsonl`
- `tensor_audio_error_skipped_pairs.jsonl`
- `tensor_maps/*.npz`
- `tensor_dataset_progress.jsonl`

The builder fails if `leakage_report.json` or `leak_safe_split_manifest.json` reports critical leakage.

## Train Shadow Critic

```bash
docker compose run --rm ml-worker python tools/train_tensor_audio_error_critic.py \
  --dataset-dir /app/lab_artifacts/audio_error_critic/tensor_r1_dataset_001 \
  --output-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001
```

Outputs:

- `tensor_audio_error_critic_manifest.json`
- `tensor_audio_error_critic.joblib`
- `tensor_audio_error_critic_eval.jsonl`
- `tensor_audio_error_localization.jsonl`
- `tensor_audio_error_critic_summary.json`

## Validate

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_tensor_audio_error_critic.py \
  --model-dir /app/lab_artifacts/audio_error_critic/tensor_r1_model_001 \
  --require-trained
```

## Contract

`tensor_maps_required=true` and `scalar_fallback_used=false` are hard contract fields.

If training data lacks a train positive or train negative class, the trainer writes a skipped manifest with an explicit `critic_not_usable_reason` instead of training a misleading model.

Localization is diagnostic: each eval row records tensor-channel energy and `top_error_channel`. It is not a repair instruction and not a release artifact.

The manifest includes both raw 0.5-threshold metrics and a conservative `threshold_metrics` section. The conservative threshold is chosen to avoid false accepts on the locked eval split; if that forces high false rejects, the model remains shadow-only and must not be used as a runtime accept gate.
