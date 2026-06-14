# Hard Case Review Pack

`tools/build_hard_case_review_pack.py` converts the deterministic hard-case benchmark into a
manual/perceptual review workflow.

This is a control-plane artifact. It does not add models, checkpoints, cleanup, debleed, denoise,
threshold changes, or fine-tuning.

## Inputs

- `hard_case_benchmark.jsonl`
- `hard_case_benchmark_manifest.json`
- `hard_case_benchmark_summary.md`
- `golden_corpus.jsonl`
- `control_plane_dashboard.json`

The golden corpus is used only to exclude protected holdout track hashes from review-task generation.
Golden rows are not used for training, threshold tuning, route tuning, or scoring calibration.

## Outputs

- `hard_case_review_pack.jsonl`
- `hard_case_review_sheet.csv`
- `hard_case_review_summary.md`

No audio is copied or exported. Diagnostic audio is not promoted as a result.

## Review Row Contract

Each row contains:

- `benchmark_case_id`
- `track_hash`
- `stem_type`
- `dominant_failure_reason`
- `failure_labels`
- `severity_score`
- `recommended_review_type`
- `review_priority`
- `expected_audio_problem`
- `human_rating`
- `human_notes`
- `reviewer_decision`
- `allowed_next_step`

Human fields start as `null` in JSONL and empty cells in CSV.

## Review Priority

- `P0`: `no_accepted_candidate`, `candidate_space_exhausted`, or
  `selected_candidate_still_has_severe_issue`.
- `P1`: unstable vocal consistency, clustered issue windows, or `unknown_blocked_reason`.
- `P2`: thin/overcleaned, robotic/noise artifact, minus-specific, or remaining review cases.

## Reviewer Decisions

Allowed reviewer decisions:

- `true_failure`
- `proxy_false_positive`
- `acceptable_artifact`
- `needs_finetune`
- `needs_policy_fix`
- `needs_reference_fix`
- `needs_manual_ab`

## Allowed Next Steps

- `manual_ab_only`
- `taxonomy_relabel`
- `policy_fix_candidate`
- `finetune_candidate`
- `holdout_only`

Holdout rows are locked to `holdout_only`; they must not be used for route tuning.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python -m tools.build_hard_case_review_pack \
  --benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --benchmark-manifest /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark_manifest.json \
  --benchmark-summary-md /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark_summary.md \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --dashboard-json /app/lab_artifacts/control_plane_phase2_20260528/control_plane_dashboard.json \
  --output-jsonl /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_pack.jsonl \
  --output-csv /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_sheet.csv \
  --output-summary-md /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_summary.md
```
