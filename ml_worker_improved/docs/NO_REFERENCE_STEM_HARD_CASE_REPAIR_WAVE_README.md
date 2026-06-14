# No-Reference Stem Hard-Case Repair Wave

`tools/run_no_reference_stem_hard_case_repair_wave.py` turns semantic judge
blockers into a bounded repair wave.

## Contract

- Docker execution only.
- Reads an existing `no_reference_stem_semantic_batch_summary.json`.
- Plans repairs only for rows whose semantic judge decision is not
  `accept_candidate`.
- Uses the model registry to select the next compatible registered candidate.
- Excludes already evaluated labels inferred from the blocked listen manifest.
- Falls back from exact issue-label matching to compatible available candidates
  for the same stem.
- Executes only through `run_testing_stuff_music_corpus_job_matrix.execute_job_matrix`.
- Builds listen-preview manifests from completed repair candidates.
- Runs a new semantic batch over existing listen roots plus the repair listen
  root.
- Keeps `release_ready=false` and `can_publish_as_release=false`.
- Does not train, fine-tune, edit model weights, relax thresholds, or copy audio.

## Melody Pitch-Salience Repair

`pitch_salience_melody_v1` is kept as a historical registered candidate, but
the semantic judge now blocks bass-dominant melody false passes with
`melody_bass_dominant`.

`pitch_salience_melody_v2` is the midband-safe registered route for this
failure class. It uses the same GPU-backed no-reference base melody route and
CPU/DSP pitch-salience repair, but with an explicit route label, a 120 Hz
minimum dominant-frequency search, narrower harmonic bandwidth, lower base
weight, and stronger salience reinforcement. It addresses:

- `melody_low_pitch_confidence`
- `melody_bass_dominant`

The route remains listen-preview only. A pass through listen manifest, GPU
provenance, semantic judge, and stem score matrix still keeps
`release_ready=false`.

Evidence run on `ytmp3free_cc_adrenaline_nine_lashes_lyrics_youtubemp3free_org`:

- v13 rejudge: `pitch_salience_melody_v1` blocked with
  `melody_bass_dominant` after the semantic guard was added.
- v14 execute: `pitch_salience_melody_v2` accepted for listen pack by the stem
  matrix with semantic score `0.771605`, empty blockers, GPU provenance passed,
  and `release_ready=false`.

Evidence run on
`baby_me_here_comes_the_hotstepper_evian_version_radio_record_baby_me_here_comes_the_hotstepper_e`:

- v14 semantic batch blocked the base `htdemucs_6s` melody with
  `melody_bass_dominant`.
- the updated registered `pitch_salience_melody_v2` route accepted under the
  same semantic judge with score `0.990813`, empty blockers, GPU provenance
  passed, and `release_ready=false`.
- the resulting full stem pack manifest is accepted for listen preview only
  with `accepted_stem_count=6`, `delivery_allowed=true`, and
  `release_status=listen_preview_only`.

## Dry Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_hard_case_repair_wave.py \
  --semantic-batch-summary-json /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_4_20260612/semantic_batch/no_reference_stem_semantic_batch_summary.json \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_4_20260612/listen_preview \
  --max-repair-jobs 1 \
  --output-root /app/lab_artifacts/testing_stuff_music_stem_hard_case_repair_dry_run_20260612
```

## Execute

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_hard_case_repair_wave.py \
  --semantic-batch-summary-json /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_4_20260612/semantic_batch/no_reference_stem_semantic_batch_summary.json \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_track3_nonvocal_listen_preview_20260612 \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_20260612/listen_preview \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_2_20260612/listen_preview \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_3_20260612/listen_preview \
  --existing-listen-root /app/lab_artifacts/testing_stuff_music_nonvocal_bounded_wave_execute_4_20260612/listen_preview \
  --max-repair-jobs 1 \
  --execute-generation \
  --output-root /app/lab_artifacts/testing_stuff_music_stem_hard_case_repair_execute_20260612
```

This is a repair/evidence route, not a release-ready gate.
