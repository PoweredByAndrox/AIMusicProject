# No-Reference Segment Blend Planner

This tool creates a metadata-only segment blend plan from existing no-reference
windowed judge evidence. It does not execute a blend, copy audio, export audio,
train, add models, or change gates.

## Contract

- No stems or reference vocal/minus files are required.
- No human CSV is required.
- No fine-tune, training, model registry edit, cleanup, debleed, denoise, or
  threshold relaxation is performed.
- Output is a plan only: `can_execute_blend_now=false` and
  `requires_crossfade=true`.
- A window is proposed only when an alternate candidate improves over the base
  window by `min_segment_delta` in the selected source-stem matrix and has no
  severe window blocker.
- Vocal-only windows use vocal score movement; minus-only windows use minus
  score movement. A `both` window requires both stem scores to improve and pair
  compatibility to pass.
- Shared or pair `window_fitness` is not allowed to pre-filter vocal-only or
  minus-only plans, because that can hide a valid one-stem repair behind an
  opposite-stem regression. The emitted `improvement_delta` is vocal delta for
  `source_stem=vocal`, minus delta for `source_stem=minus`, and the smaller of
  vocal/minus deltas for `source_stem=both`.

## Inputs

- `no_reference_candidate_pool_300.jsonl`
- `no_reference_windowed_judge_300.jsonl`
- `no_reference_optimizer_windowed_300_result.jsonl`

The windowed judge rows must contain `window_metrics`; candidate-level aggregate
metrics are not enough for segment planning.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_no_reference_segment_blends.py \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --optimizer-result-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_summary.md
```
