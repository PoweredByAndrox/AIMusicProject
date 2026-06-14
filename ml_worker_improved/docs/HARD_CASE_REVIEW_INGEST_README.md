# Hard Case Review Decision Ingest

`tools/ingest_hard_case_review_decisions.py` reads a filled `hard_case_review_sheet.csv` and
converts explicit human/perceptual decisions into control-plane queues.

This tool does not add models, checkpoints, cleanup, debleed, denoise, threshold changes, policy
changes, or training runs.

## Inputs

- `hard_case_review_pack.jsonl`
- `hard_case_review_sheet.csv`
- `hard_case_review_summary.md`
- `golden_corpus.jsonl`

Every non-empty `reviewer_decision` row in the CSV is validated against
`hard_case_review_pack.jsonl` by `benchmark_case_id` and stable metadata fields.

## Valid Reviewer Decisions

- `true_failure`
- `proxy_false_positive`
- `acceptable_artifact`
- `needs_finetune`
- `needs_policy_fix`
- `needs_reference_fix`
- `needs_manual_ab`

Unknown decisions and unknown `benchmark_case_id` values fail ingestion.

## Outputs

- `validated_review_decisions.jsonl`
- `policy_fix_queue.jsonl`
- `finetune_candidate_verified.jsonl`
- `reference_fix_queue.jsonl`
- `proxy_false_positive_queue.jsonl`
- `manual_ab_followup_queue.jsonl`
- `review_ingest_summary.md`

The manual-AB queue is emitted because `true_failure` and `needs_manual_ab` decisions can require
explicit follow-up without entering training or policy queues.

## Routing

- `needs_finetune` -> `finetune_candidate_verified.jsonl`, only when the row is not holdout-only.
- `needs_policy_fix` -> `policy_fix_queue.jsonl`.
- `needs_reference_fix` -> `reference_fix_queue.jsonl`.
- `proxy_false_positive` -> `proxy_false_positive_queue.jsonl`.
- `acceptable_artifact` -> no training queue.
- `true_failure` -> `finetune_candidate_verified.jsonl` only when `allowed_next_step=finetune_candidate`;
  otherwise `manual_ab_followup_queue.jsonl`.
- `needs_manual_ab` -> `manual_ab_followup_queue.jsonl`.

Rows with `allowed_next_step=holdout_only` are protected from fine-tune routing.

## Evidence

Queued rows preserve:

- `human_rating`
- `human_notes`
- `reviewer_decision`
- benchmark metadata and failure labels

No audio is copied or exported.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python -m tools.ingest_hard_case_review_decisions \
  --review-pack-jsonl /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_pack.jsonl \
  --review-sheet-csv /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_sheet.csv \
  --review-summary-md /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_summary.md \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase5_20260528
```
