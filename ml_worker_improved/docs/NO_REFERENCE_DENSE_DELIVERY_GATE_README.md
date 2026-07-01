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
- every mode has no blocking semantic labels in generic `decision/reasons`;
- each mode has at least `--min-window-count` windows;
- mode-specific blockers are empty;
- `both` mode is pair-compatible;
- accepted embedded gate views are mode-specific, not generic:
  `vocal_score`, `vocal_decision=accept_candidate`, and `vocal_blockers`
  for vocal; `minus_score`, `minus_decision=accept_candidate`, and
  `minus_blockers` for minus; `pair_compatibility_score`,
  `pair_decision=pair_compatible`, and `pair_blockers` for the pair.

Generic `decision=accept_candidate` plus `window_count` is diagnostic-only and
is not sufficient evidence for listen-preview promotion.
Generic `decision/reasons` still participate in semantic-clearance blocking:
labels such as `foreign_source_recovered:*`, vocal leak/bleed, residual, or
mix-conservation failures block the dense gate even when mode-specific scores
and blockers otherwise look accepted.
Optimizer metadata such as `pair_compatibility_decision` and generic
`overall_no_reference_score` also cannot substitute for dense
`pair_decision`, `pair_compatibility_score`, and `pair_blockers`.

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
