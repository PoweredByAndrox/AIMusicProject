# No-Reference Phase Alignment Optimizer

`tools/run_no_reference_phase_alignment_optimizer.py` renders bounded
phase/stereo correction candidates from existing vocal/minus pairs. It is a
no-reference optimizer route for candidates whose dominant blocker is
phase/stereo inconsistency.

## Contract

- No reference vocal/minus stems are required.
- No human CSV is required.
- No model inference, fine-tune, training, dataset build, or model registry edit
  is performed.
- No cleanup, debleed, denoise, or threshold route is added.
- Original candidate files are never overwritten.
- Rejected variants are not written as audio.
- Selected correction files are preview/candidate artifacts only:
  `can_publish_as_result=false`, `can_mark_release_ready=false`, and
  `requires_post_correction_judge=true`.

## Correction Route

The tool reads existing `mix_path`, `vocal_path`, and `instrumental_path` /
`minus_path` rows. It evaluates the original pair with the existing
no-reference judge and score matrix. It can also consume windowed judge evidence
via `--windowed-judge-jsonl`; matching rows with `severe_phase_comb_filtering`,
`phase_or_stereo_inconsistency`, or `severe_phase_collapse` trigger the bounded
phase route even when the full-track proxy score looks acceptable. If
phase/stereo evidence is below the trigger or present in matching windowed
evidence, it tries bounded vocal stereo variants:

- `vocal_mid_mono_compensated`
- `vocal_flip_left_compensated`
- `vocal_flip_right_compensated`

When matching windowed judge rows also provide `window_start_sec` plus
`window_duration_sec` or `window_end_sec`, the tool first tries window-local
versions of the same variants. These apply the compensated correction only
inside phase-blocked windows with a bounded crossfade, then evaluate the whole
vocal/minus pair again. Full-track variants are still evaluated after the
window-local variants, but no gate is relaxed and no rejected variant is written
as audio.
For window-local variants, `--min-delta` is applied to the evidence-window
fitness delta instead of the diluted whole-track delta. The whole-track
candidate is still checked for blockers, regressions, and a minimum whole-track
fitness improvement floor before any preview candidate is written. The default
floor is `--min-window-local-whole-track-delta 0.001`; variants below it are
rejected with `window_local_whole_track_delta_below_floor` even if the local
window delta passes.

For every vocal correction, the paired minus is compensated with:

```text
corrected_minus = original_minus + (original_vocal - corrected_vocal)
```

This preserves the original mix reconstruction relationship as closely as the
input pair allowed. A variant is rejected if it introduces vocal/minus/pair
blockers, damages the opposite stem, or improves less than `--min-delta`.
Rows include `windowed_phase_evidence_count` and
`windowed_phase_windows_used` so the route is auditable when window-local
correction was considered. Selected rows also include
`whole_track_improvement_delta` and `local_window_improvement_delta`.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_phase_alignment_optimizer.py \
  --candidates-jsonl /app/lab_artifacts/no_reference_art_of_dying_20260605/mode_specific_v24_collector_with_previews/no_reference_candidate_pool.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_art_of_dying_20260605/no_reference_windowed_judge_vocal.jsonl \
  --output-dir /app/lab_artifacts/no_reference_art_of_dying_20260605/phase_alignment_optimizer_v1 \
  --min-delta 0.01 \
  --min-window-local-whole-track-delta 0.001
```

Then rejudge the selected candidate rows with
`tools/run_no_reference_windowed_judge.py` in `vocal`, `minus`, and `both` mode.
Do not promote automatically from this tool.
