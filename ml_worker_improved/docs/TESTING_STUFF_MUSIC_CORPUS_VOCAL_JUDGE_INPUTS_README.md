# Testing Stuff Music Corpus Vocal Judge Inputs

`tools/build_testing_stuff_music_corpus_vocal_judge_inputs.py` builds
mode-specific no-reference candidate rows from completed corpus job execution
rows.

The bridge is diagnostic and fail-closed:

- consumes one or more `testing_stuff_music_corpus_job_execution_results.jsonl`
  files;
- uses only rows with `execution_status=completed`;
- pairs completed `vocal` and `minus` candidates by `track_id`;
- writes no audio and runs no inference;
- records `expected_acceleration_mode=external_only_cpu_dsp_scoring`;
- sets `gpu_inference_skipped=true` with an explicit skip reason;
- keeps `delivery_allowed=false` and `release_ready=false`;
- does not require references, human CSV, training, registry edits, or relaxed
  thresholds.

Outputs:

- `no_reference_vocal_windowed_judge_candidates.jsonl`
- `testing_stuff_music_corpus_vocal_judge_inputs_manifest.json`
- `testing_stuff_music_corpus_vocal_judge_inputs_summary.json`

The JSONL can be passed to:

```bash
python /app/tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/.../no_reference_vocal_windowed_judge_candidates.jsonl \
  --output-jsonl /app/lab_artifacts/.../no_reference_vocal_judge.jsonl \
  --summary-md /app/lab_artifacts/.../no_reference_vocal_judge_summary.md \
  --cache-jsonl /app/lab_artifacts/.../no_reference_vocal_judge_cache.jsonl \
  --score-mode vocal
```

For minus self-judge matrix construction, the same paired rows can be consumed by
`tools/run_no_reference_minus_self_judge_matrix.py`. Accepted delivery still
requires the downstream mode-specific score matrix and artifact status contract;
this bridge never produces a listen-pack artifact by itself.
