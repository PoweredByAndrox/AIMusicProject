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

Scalar evidence must include at least one real acoustic score, or an existing
error-map `.npz` must be available. Time/categorical fields such as `route_id`,
`stem_type`, `window_start_sec`, and `window_end_sec` do not count as acoustic
evidence by themselves. Rows that only contain those metadata fields remain
diagnostic and are marked unsafe with `missing_acoustic_feature_evidence`.

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

`overall_accept` requires selected-candidate evidence plus exact pair
compatibility evidence. `final_status` alone is never a supervised label, and a
selected minus/vocal candidate without pair evidence cannot become an overall
acceptance label.

Unsafe rows remain diagnostic rows with `safe_for_this_view=false` and
`label=null`; they are not training labels.

Every row's `feature_values` includes evidence flags:

- `acoustic_feature_evidence_present`
- `error_map_evidence_present`

When `--run-roots` imports autonomous runtime/hardcase judge artifacts, output
rows preserve structured source provenance in addition to
`provenance_source_files`:

- `runtime_source_row`
- `source_run_dir`
- `source_judge_file`
- `source_pair_file`
- `source_candidate_family`
- `candidate_family`
- `hardcase`
- `hardcase_holdout`

Runtime judge rows may be self-contained. If `candidate_pool.jsonl` contains a
matching `(track_id, candidate_id)` row, its candidate metadata is preferred.
Otherwise the builder imports paths, route aliases such as `minus_route_id`, and
candidate family fields from the judge row itself.

Accepted no-reference minus listen-preview manifests can be supplied with
`--listen-preview-manifests`. They are accepted only when the manifest says
`delivery_status=accepted`, `delivery_allowed=true`, and
`presentation_status=accepted_for_listen_pack`. The builder reads the referenced
post-render self-judge manifest, derives short exact-window evidence under
`accepted_preview_sources/lpNNN/no_reference_minus_windowed_judge.jsonl`, and
adds those windows only to `minus_critic`.

This route does not create pair or overall labels. Even when a selector file is
present, accepted listen-preview rows carry `target_views=["minus_critic"]`, so
`pair_critic` and `overall_accept` remain empty unless separate pair/overall
evidence exists. The derived source rows remain non-delivery metadata with
`release_ready=false` and no training/model side effects.

`hardcase=true` marks blocked/error windows for downstream mining. Imported
runtime rows keep `hardcase_holdout=false` unless the source row explicitly
marks them as holdout, so diagnostic hardcase evidence is not silently promoted
into locked-eval or listen-result status.

Target critic training excludes legacy or malformed safe rows whose
`feature_values` do not contain acoustic evidence, even if an older view file
marked them safe.

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
  --run-roots /app/lab_artifacts/audio_error_grounding/negative_harvest_batch_001 \
  --listen-preview-manifests /host_artifacts/loop/it076/deliv/r001/no_reference_minus_listen_preview_manifest.json
```

Omit `--run-roots` for repaired-dataset-only accounting. Include harvest/runtime
run roots when strict negative windows need to be added from real autonomous
runtime artifacts. Include `--listen-preview-manifests` only for accepted
listen-preview manifests that already passed post-render self-judge and selector
checks; blocked manifests add zero rows.

Validate:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_audio_error_training_views.py \
  --views-dir /app/lab_artifacts/audio_error_grounding/dataset_001_training_views
```

Strict negative harvest batch:

```bash
docker compose run --rm ml-worker python tools/run_strict_negative_harvest_batch.py \
  --input-audio-dir /host_testing_stuff_music \
  --output-dir /app/lab_artifacts/audio_error_grounding/neg_b001 \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --max-tracks 20 \
  --profile balanced \
  --candidate-diversity-profile balanced \
  --enable-conditional-hard-case-routes true \
  --enable-audio-preview-render true \
  --require-preview-audio-for-preview-ready true \
  --continue-on-failure true
```

Use a fresh `--output-dir` for every harvest batch. The wrapper rejects stale
`run_*` directories by default so old runtime roots cannot silently inflate
strict positive/negative counts. It also rejects `testing_flstudio_music` and,
unless a test-only bypass is explicitly enabled, requires the input corpus path
to be `testing_stuff_music` / `/host_testing_stuff_music`.
