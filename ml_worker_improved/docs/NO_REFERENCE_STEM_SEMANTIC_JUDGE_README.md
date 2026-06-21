# No-Reference Stem Semantic Judge

`tools/run_no_reference_stem_semantic_judge.py` runs a deterministic
no-reference semantic proxy judge over accepted non-vocal stem listen-preview
manifests.

Supported stems:

- `melody`
- `drums`
- `bass`
- `other`
- `guitar`
- `piano`

## Contract

- Docker execution only.
- Reads existing listen-preview manifests and selected audio files.
- Does not run model inference.
- Does not copy or export audio.
- Writes JSONL, JSON summary, Markdown summary, and structured progress JSONL.
- Uses `acceleration_mode=external_only_cpu_dsp_scoring`.
- Explicitly records `gpu_inference_skipped=true` because the judge reads
  already generated stems.
- Accepted rows mean `stem_semantic_decision=accept_candidate` for downstream
  selection only.
- Judge rows are not listen artifacts:
  `delivery_allowed=false`, `presentation_status=semantic_judge_not_for_listen_pack`,
  `release_status=not_release_ready`, `release_ready=false`.

## Stem Blockers

- common: `stem_duration_too_short`, `stem_too_silent`,
  `stem_peak_above_ceiling`
- melody: `melody_low_pitch_confidence`, `melody_bass_dominant`
- drums: `drums_low_transient_density`
- bass: `bass_low_band_energy_ratio`
- guitar/piano: `<stem>_low_tonal_confidence`,
  `<stem>_low_mid_band_energy_ratio`, `<stem>_bass_dominant`
- unsupported: `unsupported_stem_type`

`melody_bass_dominant` blocks melody candidates whose dominant frequency is
below 80 Hz while the low band dominates and the mid band is weak. This prevents
a candidate from passing the melody proxy only by concentrating energy into a
bass-like fundamental.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_stem_semantic_judge.py \
  --listen-manifest-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --output-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_stem_semantic_judge_20260612
```

## Outputs

- `no_reference_stem_semantic_judge.jsonl`
- `no_reference_stem_semantic_judge_summary.json`
- `no_reference_stem_semantic_judge_summary.md`
- `no_reference_stem_semantic_judge_progress.jsonl`

This is a semantic proxy gate, not a release-ready gate.
The `guitar` and `piano` routes are conservative tonal-instrument proxy checks
based on existing audio metrics; they are not source-separated ground truth or
melody transcription.
