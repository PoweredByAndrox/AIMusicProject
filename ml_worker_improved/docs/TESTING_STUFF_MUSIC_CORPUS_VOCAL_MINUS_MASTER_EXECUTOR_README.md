# Testing Stuff Music Corpus Vocal/Minus/Master Executor

`tools/run_testing_stuff_music_corpus_vocal_minus_master_plan.py` consumes the
fail-closed corpus vocal/minus/master plan and runs a bounded number of next
gates.

The executor is not a release route. It writes diagnostic execution artifacts,
keeps `delivery_allowed=false`, and keeps `release_ready=false` even when a
single per-track gate succeeds.

## Contract

- Docker-only CLI entrypoint.
- Bounded by `--max-steps`; default is one track/gate.
- `run_corpus_job_matrix_for_track` uses the existing corpus job matrix executor
  and records `gpu_inference_then_cpu_dsp_scoring`.
- `run_vocal_delivery_pipeline` requires both a vocal judge JSONL and
  `no_reference_vocal_candidate_score_matrix.json`. The vocal judge input may
  use either the legacy `no_reference_vocal_judge.jsonl` name or the current
  windowed judge output name `no_reference_windowed_judge_results.jsonl`.
- `run_minus_delivery_pipeline` requires self-judge manifests and
  `no_reference_minus_candidate_score_matrix.json`.
- Missing mode-specific matrices block execution before delivery runners are
  called.
- `run_pair_compatibility_and_master_manifest` requires accepted vocal/minus
  manifest paths from the plan and pair compatibility JSONL evidence.
- `complete_nonvocal_stem_score_matrices` requires
  `--stem-listen-manifest-root`, runs the existing non-vocal semantic batch for
  the selected track only, writes a matching
  `no_reference_stem_candidate_score_matrix.json`, and marks the step completed
  only when `bass`, `drums`, `melody`, and `other` all have accepted,
  semantic-verified listen-preview rows. This gate is
  `external_only_cpu_dsp_scoring`; it does not run hidden model inference.
- `build_full_stem_pack_manifest` requires an accepted master manifest path from
  the plan and a matching `no_reference_stem_candidate_score_matrix.json`.
  It calls the existing full-stem pack contract builder and writes only
  listen-preview manifest evidence; it does not copy/export audio or mark
  release-ready.

## Outputs

- `testing_stuff_music_corpus_vocal_minus_master_execution_manifest.json`
- `testing_stuff_music_corpus_vocal_minus_master_execution_summary.json`
- `testing_stuff_music_corpus_vocal_minus_master_execution_steps.jsonl`
- `testing_stuff_music_corpus_vocal_minus_master_execution_progress.jsonl`

The progress JSONL records start/end events for each step, selected track, gate,
expected acceleration mode, and whether GPU inference is required or skipped.

## Dry Run

```bash
python /app/tools/run_testing_stuff_music_corpus_vocal_minus_master_plan.py \
  --plan-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_plan_v3/testing_stuff_music_corpus_vocal_minus_master_plan.json \
  --output-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_execution_dry_run_v1 \
  --max-steps 1
```

## Execute One Bounded Gate

```bash
python /app/tools/run_testing_stuff_music_corpus_vocal_minus_master_plan.py \
  --plan-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_plan_v3/testing_stuff_music_corpus_vocal_minus_master_plan.json \
  --output-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_execution_v1 \
  --max-steps 1 \
  --execute \
  --job-matrix-jsonl /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/job_matrix/testing_stuff_music_corpus_job_matrix.jsonl \
  --vocal-judge-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --vocal-score-matrix-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --minus-self-judge-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --minus-score-matrix-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --pair-compatibility-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --stem-listen-manifest-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1 \
  --stem-score-matrix-root /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1
```

If the selected plan row is a vocal or minus delivery gate and the matching
mode-specific score matrix is missing, the executor records a blocked step
instead of falling back to heuristic base selection.

If the selected row is `complete_nonvocal_stem_score_matrices`, the executor
writes the semantic-batch outputs under the step output directory:

- `no_reference_stem_semantic_batch_summary.json`
- `no_reference_stem_semantic_batch_progress.jsonl`
- `<track_id>/stem_semantic_judge/no_reference_stem_semantic_judge.jsonl`
- `<track_id>/stem_score_matrix/no_reference_stem_candidate_score_matrix.json`

The step blocks if no listen manifests are available for the selected track or
if any required non-vocal stem is not accepted and semantic-verified.

If the selected row is `build_full_stem_pack_manifest`, the executor writes:

- `no_reference_full_stem_pack_manifest.json`
- `no_reference_full_stem_pack_summary.md`

The executor summary still keeps `delivery_allowed=false` and
`release_ready=false`; only the per-track full-stem pack manifest can be
`accepted_for_listen_pack`.
