# No-Reference Stem Semantic Batch

`tools/run_no_reference_stem_semantic_batch.py` runs the non-vocal stem semantic
judge and semantic-aware stem score matrix per discovered `track_id`.

It consumes either existing `no_reference_stem_listen_preview_manifest.json`
files or a short-name stem review `pack.json` from
`tools/build_music_intelligence_stem_listen_pack.py`. It does not generate,
copy, or export audio.

## Contract

- Docker execution only.
- Discovers presemantic stem manifests under one or more
  `--listen-manifest-root` paths. These inputs may be
  `semantic_judge_input=true` with
  `presentation_status=semantic_judge_input_not_for_listen_pack`; they are
  judge-consumable but not user-presentable listen/result artifacts.
- With `--input-pack-json`, materializes only short JSON views under `src/`
  and reads the existing audio referenced by the pack.
- Groups manifests by `track_id`.
- For each legacy track, writes:
  - `stem_semantic_judge/no_reference_stem_semantic_judge.jsonl`
  - `stem_semantic_judge/no_reference_stem_semantic_judge_summary.json`
  - `stem_score_matrix/no_reference_stem_candidate_score_matrix.json`
- For each short-pack track, writes short metadata:
  - `tNNN/judge/rows.jsonl`
  - `tNNN/judge/summary.json`
  - `tNNN/matrix/matrix.json`
  - `review/review.json`
  - `review/rows.jsonl`
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

## Short Pack Run

Use this after creating a semantic review pack. The output `review/review.json`
is the machine-readable semantic review artifact consumed by the listen coverage
scorecard. Unsupported stems are reported as blockers; they are not silently
accepted.

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_semantic_batch.py \
  --input-pack-json /app/lab_artifacts/lp_v45_001/sem001/pack.json \
  --short-output-names \
  --output-root /app/lab_artifacts/lp_v45_001/semj001
```
