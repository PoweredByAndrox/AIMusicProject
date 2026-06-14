# Testing Stuff Music Corpus Minus Judge Inputs

`tools/build_testing_stuff_music_corpus_minus_judge_inputs.py` builds
mode-specific no-reference minus candidate rows from completed corpus job
execution rows.

The bridge is diagnostic and fail-closed:

- consumes one or more `testing_stuff_music_corpus_job_execution_results.jsonl`
  files;
- uses only rows with `execution_status=completed`;
- accepts completed `target_stem=minus`, `instrumental`, `vocal`, or `vocals`
  rows as minus candidates only when their `stem_manifest.json` exposes a
  `minus`/`instrumental` path plus a vocal companion path;
- reads the completed `stem_manifest.json` to preserve the exact minus path,
  vocal companion path, and optional stem paths such as `melody`;
- skips minus candidates whose manifest is missing, whose minus path is absent,
  or whose vocal companion path is absent;
- writes no audio and runs no inference;
- records `expected_acceleration_mode=external_only_cpu_dsp_scoring`;
- sets `gpu_inference_skipped=true` with an explicit skip reason;
- keeps `delivery_allowed=false` and `release_ready=false`;
- does not require references, human CSV, training, registry edits, or relaxed
  thresholds.

Outputs:

- `no_reference_minus_windowed_judge_candidates.jsonl`
- `no_reference_minus_windowed_judge_candidates_manifest.json`
- `no_reference_minus_windowed_judge_candidates_summary.md`

The JSONL can be passed to:

```bash
python /app/tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/.../no_reference_minus_windowed_judge_candidates.jsonl \
  --output-jsonl /app/lab_artifacts/.../no_reference_minus_judge.jsonl \
  --summary-md /app/lab_artifacts/.../no_reference_minus_judge_summary.md \
  --cache-jsonl /app/lab_artifacts/.../no_reference_minus_judge_cache.jsonl \
  --score-mode minus
```

Accepted delivery still requires the downstream mode-specific score matrix,
post-render/self-judge checks where applicable, and the artifact status
contract. This bridge never produces a listen-pack artifact by itself.
