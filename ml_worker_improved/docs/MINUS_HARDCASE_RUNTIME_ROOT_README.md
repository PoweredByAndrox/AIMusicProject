# Minus Hardcase Runtime Root

`tools/build_minus_hardcase_runtime_root.py` materializes scattered minus
candidate manifests or JSONL rows into the runtime-root contract consumed by:

- `tools/run_no_reference_windowed_judge.py --score-mode minus`
- `tools/build_audio_error_grounding_dataset.py`
- `tools/audit_audio_error_dataset_provenance.py`
- `tools/build_audio_error_training_views.py`

This is a provenance bridge only. It does not copy audio, render audio, train a
model, relax thresholds, publish a listen pack, or mark a candidate release
ready.

## Inputs

Use one or more:

```bash
python tools/build_minus_hardcase_runtime_root.py \
  --candidate-manifest /path/to/stem_manifest.json \
  --candidate-manifest /path/to/repair/manifest.json \
  --candidate-jsonl /path/to/candidate_rows.jsonl \
  --output-dir /path/to/runtime_root
```

The bridge accepts:

- stem manifests with `stems.vocals` and `stems.instrumental`;
- repair manifests with embedded `candidate_row`;
- JSONL rows with `mix_path`, `vocal_path`, and `minus_path`.

For macOS shared-workspace runs, default rewrites normalize old Mac-local paths
into the mounted Windows project path:

- `/Users/andrejviluha/aimusic_local_artifacts` -> `$AIMUSIC_MAC_PROJECT_ROOT/artifacts`
- `/host_artifacts` -> `$AIMUSIC_MAC_PROJECT_ROOT/artifacts`
- `D:\AIMusicProject` -> `$AIMUSIC_MAC_PROJECT_ROOT`
- `D:/AIMusicProject` -> `$AIMUSIC_MAC_PROJECT_ROOT`

## Outputs

The output directory contains:

```text
run_manifest.json
candidate_pool.jsonl
selected_candidates.jsonl
pair_compatibility.jsonl
preview_render_contract.json
summary.md
skipped.jsonl
progress.jsonl
```

Rows with missing audio files are written to `skipped.jsonl`; they are not
filled with synthetic paths.

## Safety

All generated manifests and rows keep:

- `delivery_allowed=false`
- `presentation_status=diagnostic_runtime_root_not_for_listen_pack`
- `release_ready=false`
- `training_started=false`
- `separator_training_started=false`
- `model_downloaded=false`
- `model_registry_edited=false`
- `thresholds_relaxed=false`
- `audio_copied=false`
- `audio_exported=false`

The runtime root becomes eligible only for downstream diagnostic scoring and
audio-error grounding. It is not a production delivery artifact.
