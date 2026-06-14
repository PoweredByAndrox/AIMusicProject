# Testing Stuff Music Corpus Plan

`tools/build_testing_stuff_music_corpus_plan.py` builds a deterministic
no-reference evaluation queue from the Docker-mounted `testing_stuff_music`
audio corpus.

This is a planning/inventory route only. It does not read audio content, copy
audio, hash audio, export stems, run separation models, train, fine-tune, edit
the model registry, add cleanup/debleed/denoise, relax thresholds, or mark any
artifact `release_ready`.

## Contract

- Input source root defaults to `/host_testing_stuff_music`.
- The CLI entrypoint is Docker-only and rejects non-mounted source roots; the
  source root must be `/host_testing_stuff_music`.
- The expected Docker mount is read-only:
  `./testing_stuff_music:/host_testing_stuff_music:ro`.
- Each discovered audio file becomes one no-reference evaluation row.
- Rows target `vocal`, `minus`, `melody`, `drums`, `bass`, and `other` so later
  routes can generate mode-specific candidates for all requested stems.
- `training_allowed=false`, `fine_tune_allowed=false`, and
  `can_start_training_now=false` for every row.
- `reference_available=false` because raw MP3s are not verified ground-truth
  stems.
- Candidate generation is marked as GPU-required for the next route, while this
  planner itself is metadata-only.

## Outputs

- `testing_stuff_music_corpus_plan.jsonl`
- `testing_stuff_music_corpus_manifest.json`
- `testing_stuff_music_corpus_summary.md`

Every row includes:

- `track_id`
- `relative_path`
- `source_audio_path`
- `docker_audio_path`
- `host_audio_path_hint`
- `source_audio_suffix`
- `source_audio_bytes`
- `target_stems`
- `next_route`
- `expected_acceleration_mode`
- safety and forbidden-action flags

The manifest and summary include `docker_runtime_required=true`,
`docker_source_root_required=/host_testing_stuff_music`, and runtime validation
flags. CLI-generated artifacts set `docker_runtime_validated=true` and
`docker_source_root_validated=true` only after the Docker guard and source-root
contract pass.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_testing_stuff_music_corpus_plan.py \
  --source-root /host_testing_stuff_music \
  --docker-root-label /host_testing_stuff_music \
  --host-root-label D:/AIMusicProject/testing_stuff_music \
  --output-dir /app/lab_artifacts/testing_stuff_music_corpus_plan_20260611
```

The output is not a release result. It is the source-of-truth queue for the next
safe step: generate/evaluate candidates across the corpus, then mine blockers
for minus, vocal, melody, and other stems without pretending raw corpus files are
training ground truth.
