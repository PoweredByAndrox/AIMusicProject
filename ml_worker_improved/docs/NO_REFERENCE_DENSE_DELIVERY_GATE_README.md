# No-Reference Dense Delivery Gate

`tools/build_no_reference_delivery_gate.py` builds a listen-preview manifest
only after an accepted optimizer row also passes dense no-reference judging in
all three modes:

- `score_mode=vocal`
- `score_mode=minus`
- `score_mode=both`

It does not run models, render audio, copy audio, train, fine-tune, add cleanup
routes, or relax thresholds.

## Contract

A row can become `accepted_for_listen_pack` only when:

- the optimizer row is accepted by the existing listen-preview contract;
- phase-corrected optimizer rows are eligible only when they carry `requires_post_correction_judge=true`;
- matching dense judge rows exist for `vocal`, `minus`, and `both`;
- every mode has `decision=accept_candidate`;
- each mode has at least `--min-window-count` windows;
- mode-specific blockers are empty;
- `both` mode is pair-compatible.

The output remains listen preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `can_mark_release_ready=false`

If any dense evidence is missing or blocked, the tool writes a blocked manifest
with `dominant_failure_reason=dense_delivery_gate_failed` and exits non-zero.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_delivery_gate.py \
  --optimizer-result-jsonl /app/path/no_reference_optimizer_mode_specific_result.jsonl \
  --vocal-judge-jsonl /app/path/no_reference_accepted_pair_dense_vocal.jsonl \
  --minus-judge-jsonl /app/path/no_reference_accepted_pair_dense_minus.jsonl \
  --both-judge-jsonl /app/path/no_reference_accepted_pair_dense_both.jsonl \
  --output-manifest-json /app/path/no_reference_dense_delivery_gate_manifest.json \
  --summary-md /app/path/no_reference_dense_delivery_gate_summary.md \
  --min-window-count 5
```
