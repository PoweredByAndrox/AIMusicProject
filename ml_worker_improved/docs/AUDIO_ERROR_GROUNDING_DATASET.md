# Audio Error Grounding Dataset

MODEL-1 builds a no-reference, training-ready dataset that grounds separator
errors in waveform and time/frequency maps from existing runtime artifacts. It
does not train a separator, download models, relax thresholds, run melody
transcription, update recommendation indexes, publish final results, or mark
anything release-ready.

## Inputs

The builder accepts one or more runtime roots and discovers root or nested run
directories containing:

- `candidate_pool.jsonl`
- `vocal_windowed_judge.jsonl` or `vocal_judge.jsonl`
- `minus_windowed_judge.jsonl` or `minus_judge.jsonl`
- `pair_compatibility.jsonl`
- `preview_render_contract.json`

Audio paths are read from judge rows first, then candidate rows. The original
mix is used only as a mixture-consistency constraint. It is never treated as a
true source reference, and no fake ground-truth stems are created.

The builder is strict about provenance. It supports only explicit schema aliases
from known runtime artifacts. It does not invent candidate ids, route ids,
window timing, pair status, or judge decisions. If required provenance is
missing, the row remains visible for accounting but is marked
`safe_for_training=false` with a concrete `unsafe_reason`.

## Outputs

The output directory contains:

- `audio_error_grounding_manifest.json`
- `audio_error_window_dataset.jsonl`
- `audio_error_maps_index.jsonl`
- `audio_error_dataset_summary.md`
- `audio_error_grounding_progress.jsonl`
- `error_maps/*.npz`

Each dataset row includes:

- track/run/window identity: `dataset_id`, `track_id`, `run_id`, `window_id`
- bounded window timing: `window_start_sec`, `window_end_sec`
- audio paths: `mix_path`, `vocal_candidate_path`, `minus_candidate_path`
- candidate provenance: `candidate_id`, `route_id`, `stem_type`
- judge and pair status: `blockers`, `judge_decision`,
  `pair_compatibility_status`
- scalar error metrics: `residual_energy`, `mixture_consistency_error`,
  `phase_collapse_score`, `stereo_width_score`,
  `vocal_bleed_proxy_score`, `instrumental_bleed_proxy_score`,
  `missing_detail_score`, `transient_damage_score`,
  `artifact_noise_score`
- map references: `error_map_paths`
- training readiness: `train_split`, `safe_for_training`, `unsafe_reason`
- safety flags, always including `release_ready=false`

Rows with missing audio remain in the dataset for hard-case accounting, but
`safe_for_training=false` and no `.npz` maps are attached.
Rows with missing provenance or missing window-level judge status are handled the
same way; no placeholder or synthetic value is treated as training-safe.

## Error Maps

For every safe candidate window the extractor computes:

- waveform residual: `mix - vocal - minus`
- STFT residual magnitude
- phase difference map
- stereo width/correlation map
- vocal bleed proxy map
- instrumental bleed proxy map
- missing detail proxy map
- transient loss proxy map
- noise/artifact proxy map
- reconstruction error map

Maps are stored in compressed NumPy `.npz` files. Large waveforms and
time/frequency maps are resolution-capped to keep the dataset bounded while
preserving localized error structure.

## Safety Contract

- `reference_required=false`
- `human_csv_required=false`
- `training_started=false`
- `separator_training_started=false`
- `model_downloaded=false`
- `model_registry_edited=false`
- `models_changed=false`
- `thresholds_relaxed=false`
- `cleanup_routes_added=false`
- `melody_transcription_ran=false`
- `recommendation_index_updated=false`
- `release_ready=false`

The learned critic or future training code must not treat blocked/dirty stems as
ground truth. Accepted windows can be used as positive examples, and blocked
windows can be used as negative/hard examples.

## Build

```bash
docker compose run --rm ml-worker python tools/build_audio_error_grounding_dataset.py \
  --run-roots \
    /app/lab_artifacts/autonomous_runs/stem_3_track002_conditional_route_001 \
    /app/lab_artifacts/autonomous_runs/stem_3_household_goods_conditional_route_001 \
    /app/lab_artifacts/autonomous_runs/preview_1_audio_render_contract_001 \
    /app/lab_artifacts/no_reference_meta_harness/runtime_validation_audio_preview_001 \
  --output-dir /app/lab_artifacts/audio_error_grounding/dataset_001 \
  --window-sec 12 \
  --hop-sec 12
```

This is a CPU/DSP extraction pass. Progress rows explicitly record
`gpu_inference_skipped=true` because no neural separator inference is performed.

## Validate

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_audio_error_grounding_dataset.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001
```

Validation fails if required row fields are missing, safe rows lack computable
maps, window boundaries are invalid, or any artifact sets `release_ready=true`,
`melody_transcription_ran=true`, or `recommendation_index_updated=true`.
Validation also rejects `unknown` placeholder values in provenance fields and any
safe row with missing `track_id`, `run_id`, `candidate_id`, `route_id`,
`judge_decision`, or `pair_compatibility_status`.
