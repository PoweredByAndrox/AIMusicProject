# No-Reference Stem Semantic Batch

`tools/run_no_reference_stem_semantic_batch.py` runs the non-vocal stem semantic
judge and semantic-aware stem score matrix per discovered `track_id`.

It consumes existing `no_reference_stem_listen_preview_manifest.json` files and
does not generate, copy, or export audio.

## Contract

- Docker execution only.
- Discovers listen-preview manifests under one or more `--listen-manifest-root`
  paths.
- Groups manifests by `track_id`.
- For each track, writes:
  - `stem_semantic_judge/no_reference_stem_semantic_judge.jsonl`
  - `stem_semantic_judge/no_reference_stem_semantic_judge_summary.json`
  - `stem_score_matrix/no_reference_stem_candidate_score_matrix.json`
- Writes batch summary and progress JSONL.
- Uses `acceleration_mode=external_only_cpu_dsp_scoring`.
- Records `gpu_inference_skipped=true` because it reads existing stems.
- Keeps `release_ready=false` and `can_publish_as_release=false`.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_semantic_batch.py \
  --listen-manifest-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --output-root /app/lab_artifacts/testing_stuff_music_nonvocal_stem_semantic_batch_20260612
```

Optional filter:

```bash
--track-id nine_thou_grant_mohrman_superstars_remix_320
```

This is a corpus orchestration layer for semantic proxy gating. It is not a
release-ready gate.
