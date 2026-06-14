# No-Reference Existing Vocal/Minus Cross Product

`tools/build_no_reference_existing_vocal_minus_cross_product.py` builds metadata
rows that pair existing vocal candidates with existing minus candidates for the
same source mix. It does not render, copy, publish, or accept audio.

## Contract

- Inputs are JSONL candidate rows with `candidate_id`, `track_id`, `mix_path` or
  `source_audio_path`, and either vocal or minus paths.
- Rows are paired only when they share the same source mix. Exact `track_id`
  match is preferred; shared normalized `mix_path` can be used when track IDs
  drift between artifact roots.
- Output rows are complete vocal/minus candidate pairs for later dense judge,
  selector, self-judge, or delivery routes.
- Every output row remains blocked:
  `delivery_allowed=false`, `can_publish_as_result=false`,
  `can_mark_release_ready=false`.
- The builder sets `requires_dense_post_cross_product_judge=true` and
  `requires_dense_post_reconciliation_judge=true`.

This route is a control-plane accelerator for comparing existing candidates. It
is not a release route and must not be used to bypass mode-specific gates.

## Safety

- No reference stems are required.
- No human CSV is required.
- No model inference, training, fine-tuning, cleanup, debleed, or denoise is run.
- No thresholds or gates are relaxed.

## Example

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/build_no_reference_existing_vocal_minus_cross_product.py \
  --vocal-candidates-jsonl /app/lab_artifacts/example/vocals.jsonl \
  --minus-candidates-jsonl /app/lab_artifacts/example/minus.jsonl \
  --output-jsonl /app/lab_artifacts/example/no_reference_existing_vocal_minus_cross_product.jsonl \
  --summary-md /app/lab_artifacts/example/no_reference_existing_vocal_minus_cross_product_summary.md
```
