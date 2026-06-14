# No-Reference Candidate Pool

This tool scans existing lab artifacts and builds a deterministic candidate pool
for the no-reference optimizer. It only collects paths and manifest metadata.

## Contract

- No audio is generated, copied, read, hashed, or exported.
- No models, weights, training, fine-tune, cleanup, debleed, denoise, or
  threshold routes are added.
- No reference vocal/minus files or human CSV are required.
- Only complete existing pairs are emitted to the optimizer pool:
  `mix_path + vocal_path + minus_path`.
- Existing no-reference preview comparison rows from
  `preview_vs_base_candidates.jsonl` and
  `no_reference_segment_blend_preview_judge_input.jsonl` are also collected as
  complete candidate pairs when their mix, vocal preview, and minus preview/base
  paths already exist. They remain ordinary candidates for judging; collecting
  them does not publish or mark them release-ready.
- Targeted segment-blend preview judge rows matching
  `*segment_blend_preview_window_judge*.jsonl` are collected only when
  `preview_decision=preview_improved_window`. Regressed, noop/self-blend, and
  unsafe promotion rows are skipped with explicit reasons. Defensive noop
  markers such as `preview_differs_from_base=false` or
  `allowed_next_step=fix_candidate_identity/path-distinctness` also block
  collection even if an upstream row is mislabeled as improved.
- Existing candidate-pool JSONL files can be supplied with
  `--candidate-pool-jsonl`. These seed rows are validated as complete
  `mix_path + vocal_path + minus_path` pairs and merged with newly discovered
  preview candidates before deterministic dedup/capping. This is used when a
  segment preview improves one stem but the optimizer still needs the original
  base candidates available for the other stem.
- Incomplete candidates are written to a skipped JSONL with an explicit reason.
- Identical vocal/minus pairs are deduplicated deterministically.
- To keep optimizer runs bounded, the collector emits up to 4 complete pairs per
  track by default. Use `--max-pairs-per-track` to change this deterministic cap.
- It also emits up to 12 complete pairs globally by default. Full discovered
  counts and capped counts remain in the summary.

## Run

```bash
python tools/collect_no_reference_candidate_pool.py \
  --lab-artifacts-root /app/lab_artifacts \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_summary.md \
  --skipped-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/skipped_no_reference_candidates.jsonl \
  --max-pairs-per-track 4 \
  --max-total-pairs 12
```

## Outputs

- `no_reference_candidate_pool.jsonl`
- `no_reference_candidate_pool_summary.md`
- `skipped_no_reference_candidates.jsonl`

Every complete row includes:

- `track_id`
- `candidate_id`
- `mix_path`
- `vocal_path`
- `minus_path`
- `instrumental_path`
- `source_manifest`
- `candidate_family`
- `candidate_origin`
- `score_hints`
- `is_complete_pair`
- `skip_reason`

`instrumental_path` mirrors `minus_path` for compatibility with the existing
no-reference judge and optimizer CLI.
