# No-Reference Non-Vocal Bounded Wave

`tools/run_no_reference_nonvocal_bounded_wave.py` automates a bounded corpus wave
for non-vocal stems.

It selects the next tracks from a `testing_stuff_music` job matrix, optionally
runs GPU candidate generation, builds stem listen-preview manifests, and runs
the semantic batch gate.

## Contract

- Docker execution only.
- Dry-run by default; GPU generation starts only with `--execute-generation`.
- `--max-tracks` bounds the generation wave; each selected track is bounded by
  `--max-jobs-per-track`.
- Selects only tracks with all target stems: `bass`, `drums`, `melody`, `other`.
- Excludes tracks already present in `--existing-listen-root`.
- Writes structured progress JSONL and summary JSON/Markdown, including
  generation/listen-preview/semantic-batch phase events.
- Semantic batch scope is cumulative by design:
  `cumulative_existing_plus_new_listen_roots`. This keeps a single corpus matrix
  over existing listen roots plus the newly generated listen root, while
  generation remains bounded to the selected tracks only.
- Keeps `release_ready=false` and `can_publish_as_release=false`.
- Does not copy or export audio. Audio generation is delegated only to the
  existing job-matrix executor when `--execute-generation` is set.

## Dry Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_nonvocal_bounded_wave.py \
  --job-matrix-jsonl /app/lab_artifacts/testing_stuff_music_track2_nonvocal_job_matrix_20260612/testing_stuff_music_corpus_job_matrix.jsonl \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track3_nonvocal_listen_preview_20260612 \
  --max-tracks 1 \
  --output-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_dry_run_20260612
```

## Execute

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_nonvocal_bounded_wave.py \
  --job-matrix-jsonl /app/lab_artifacts/testing_stuff_music_track2_nonvocal_job_matrix_20260612/testing_stuff_music_corpus_job_matrix.jsonl \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track3_nonvocal_listen_preview_20260612 \
  --max-tracks 1 \
  --max-jobs-per-track 4 \
  --execute-generation \
  --output-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_20260612
```

This is not a release-ready gate.
