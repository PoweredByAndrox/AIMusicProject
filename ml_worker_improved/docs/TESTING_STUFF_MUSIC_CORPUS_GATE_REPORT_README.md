# Testing Stuff Music Corpus Gate Report

`tools/build_testing_stuff_music_corpus_gate_report.py` builds a fail-closed
diagnostic report for bounded `testing_stuff_music` corpus waves.

The report does not render audio, does not change thresholds, does not edit the
model registry, and never marks artifacts `release_ready`.

## Inputs

- `--corpus-plan-jsonl`: rows from `build_testing_stuff_music_corpus_plan.py`.
- `--execution-results-jsonl`: one or more execution result JSONL files from
  `run_testing_stuff_music_corpus_job_matrix.py`.
- `--score-matrix-json`: optional accepted evidence from mode/stem-specific
  score matrices.

## Gate Semantics

- Accepted rows are only taken from score matrices that already carry
  `delivery_status=accepted`, `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack`.
- Completed `vocal` candidates without vocal judge + selector + listen-preview
  evidence remain `diagnostic_only_missing_mode_specific_gate`.
- Completed `minus` candidates without self-judge + result-judge + selector +
  listen-preview evidence remain `diagnostic_only_missing_mode_specific_gate`.
- Completed non-vocal stems without a stem listen/semantic score matrix remain
  `diagnostic_only_missing_stem_score_matrix`.

## Example

```bash
python /app/tools/build_testing_stuff_music_corpus_gate_report.py \
  --corpus-plan-jsonl /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/plan/testing_stuff_music_corpus_plan.jsonl \
  --execution-results-jsonl /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/execution_nine_vocal_htdemucs6s_v1/testing_stuff_music_corpus_job_execution_results.jsonl \
  --execution-results-jsonl /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/execution_nine_minus_instrumental_resurrection_v1/testing_stuff_music_corpus_job_execution_results.jsonl \
  --score-matrix-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/nine_melody_score_matrix_v1/no_reference_stem_candidate_score_matrix.json \
  --output-manifest-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/corpus_gate_report_v1/testing_stuff_music_corpus_gate_report_manifest.json \
  --summary-md /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/corpus_gate_report_v1/testing_stuff_music_corpus_gate_report_summary.md
```

The output is diagnostic-only and is intended to drive the next bounded corpus
wave instead of promoting raw candidate generation as a listen result.
