# Audio Error Training Views

MODEL-1C splits the repaired audio error grounding dataset into target-specific
training views. It does not train a separator, train a critic, download models,
relax thresholds, publish results, or mark `release_ready`.

## Inputs

The builder reads a repaired grounding dataset:

- `repaired_audio_error_grounding_manifest.json`
- `repaired_audio_error_window_dataset.jsonl`
- `repaired_audio_error_maps_index.jsonl`
- `provenance_repair_report.json`

Rows are treated as source-of-truth only when they carry exact artifact evidence:
real `candidate_id`, real `route_id`, window boundaries, candidate audio path,
mix path, error maps or scalar error features, and the relevant artifact source
file.

## Views

Outputs:

- `audio_error_training_views_manifest.json`
- `minus_critic_training_view.jsonl`
- `vocal_critic_training_view.jsonl`
- `pair_critic_training_view.jsonl`
- `overall_accept_training_view.jsonl`
- `audio_error_training_views_summary.md`

`minus_critic` and `vocal_critic` labels come only from exact windowed judge
evidence. They do not require `pair_compatibility_status`.

`pair_critic` requires proven pair compatibility evidence.

`overall_accept` requires selected-candidate evidence. `final_status` alone is
never a supervised label.

Unsafe rows remain diagnostic rows with `safe_for_this_view=false` and
`label=null`; they are not training labels.

## Safety

The views keep:

- `training_started=false`
- `separator_training_started=false`
- `model_downloaded=false`
- `model_registry_edited=false`
- `thresholds_relaxed=false`
- `cleanup_routes_added=false`
- `release_ready=false`
- `melody_transcription_ran=false`
- `recommendation_index_updated=false`

## Commands

Build:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_audio_error_training_views.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired \
  --output-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views \
  --run-roots /app/lab_artifacts/audio_error_grounding/negative_harvest_batch_001
```

Omit `--run-roots` for repaired-dataset-only accounting. Include harvest/runtime
run roots when strict negative windows need to be added from real autonomous
runtime artifacts.

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_audio_error_training_views.py \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views
```

Strict negative harvest batch:

```bash
docker compose run --rm ml-worker python tools/run_strict_negative_harvest_batch.py \
  --input-audio-dir /host_testing_stuff_music \
  --output-dir /app/lab_artifacts/audio_error_grounding/negative_harvest_batch_001 \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --max-tracks 20 \
  --profile balanced \
  --candidate-diversity-profile balanced \
  --enable-conditional-hard-case-routes true \
  --enable-audio-preview-render true \
  --require-preview-audio-for-preview-ready true \
  --continue-on-failure true
```
