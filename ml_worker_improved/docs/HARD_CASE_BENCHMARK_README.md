# Hard Case Benchmark

`tools/build_hard_case_benchmark.py` compresses the full control-plane hard-case corpus into a
small deterministic benchmark for review and regression planning.

## Inputs

- `hard_case_corpus.jsonl`
- `hard_case_corpus_manifest.json`
- `control_plane_dashboard.json`
- `golden_corpus.jsonl`

The golden corpus is used only as protected holdout exclusion metadata. It is not used for training,
threshold tuning, scoring calibration, or candidate selection.

## Outputs

- `hard_case_benchmark.jsonl`
- `hard_case_benchmark_manifest.json`
- `hard_case_benchmark_summary.md`

The tool does not copy or export audio. Benchmark rows keep manifest provenance only.

## Row Contract

Each benchmark row contains:

- `benchmark_case_id`
- `source_hard_case_id`
- `track_hash`
- `split`
- `stem_type`
- `dominant_failure_reason`
- `failure_labels`
- `failed_checks`
- `severity_score`
- `window_start_sec`
- `window_end_sec`
- `source_manifest`
- `recommended_review_type`

`split` is recomputed from `track_hash` with `stable_track_split(track_hash)` so the split is track
level, not window level.

## Sampling Contract

Default target size is 120 rows, constrained to 50-200 rows.

The sampler is deterministic and stratified:

- candidate-space exhaustion cases are selected when present;
- no-accepted-candidate cases are selected when present;
- unknown reasons are sampled separately;
- minus and vocal stems both receive minimum coverage when available;
- remaining rows are filled by dominant-failure strata and deterministic hash order.

Severity scoring is control-plane only:

- `candidate_space_exhausted`, `no_accepted_candidate`, and
  `selected_candidate_still_has_severe_issue` are high severity;
- unsafe diagnostic rows are high severity;
- unknown reasons are medium/high severity.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python -m tools.build_hard_case_benchmark \
  --hard-case-jsonl /app/lab_artifacts/control_plane_phase2_20260528/hard_case_corpus.jsonl \
  --hard-case-manifest /app/lab_artifacts/control_plane_phase2_20260528/hard_case_corpus_manifest.json \
  --dashboard-json /app/lab_artifacts/control_plane_phase2_20260528/control_plane_dashboard.json \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --output-manifest /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark_manifest.json \
  --output-summary-md /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark_summary.md
```
