# No-Reference Score Matrix

`tools/no_reference_score_matrix.py` separates no-reference candidate scoring by
stem mode:

- `vocal`: evaluates vocal quality only.
- `minus`: evaluates instrumental/minus quality only.
- `both`: checks selected vocal/minus pair compatibility only.

The matrix consumes existing no-reference metrics. It does not read audio, copy
audio, run models, train, fine-tune, add cleanup/debleed/denoise routes, or mark
anything `release_ready`.

## Location

- Matrix implementation: `tools/no_reference_score_matrix.py`
- Unit tests: `tests/test_no_reference_score_matrix.py`
- Windowed judge integration: `tools/run_no_reference_windowed_judge.py`
- Mode-specific optimizer integration: `tools/run_no_reference_optimizer.py`
- Candidate selector integration: `tools/run_no_reference_candidate_selector.py`
- Vocal candidate matrix artifact: `tools/build_no_reference_vocal_candidate_score_matrix.py`
- Minus candidate matrix artifact: `tools/build_no_reference_minus_candidate_score_matrix.py`
- Segment planner integration: `tools/plan_no_reference_segment_blends.py`
- Preview judge integration: `tools/judge_no_reference_segment_blend_preview.py`

## Contract

- `score_candidate(metrics, "vocal")` returns `vocal_score`,
  `vocal_blockers`, `vocal_dominant_defect`, and `vocal_decision`.
- `score_candidate(metrics, "minus")` returns `minus_score`,
  `minus_blockers`, `minus_dominant_defect`, and `minus_decision`.
- Minus scoring fails closed when required evidence is absent or non-finite.
  Missing/invalid `vocal_leakage_in_minus_score` records
  `missing_minus_vocal_leakage_evidence`; missing/invalid
  `mixture_consistency_score` records
  `missing_minus_instrumental_integrity_evidence`. Missing evidence is never
  treated as a clean minus.
- Present numeric scoring metrics and pair identity metadata must be finite
  numbers. `None`, empty strings, booleans, `NaN`, and `Inf` values record
  `invalid_no_reference_metric:<metric_name>` blockers instead of being clipped
  into a good score component. Missing optional metrics may still use the
  scorer's documented defaults for compatibility with partial legacy payloads.
- `pair_compatibility_score(vocal_candidate, minus_candidate)` returns
  `pair_compatibility_score`, `pair_blockers`, and `pair_decision`.
- Invalid modes raise `ValueError`.
- The optimizer must not use `both` mode as a default selector. It selects vocal
  and minus independently, then runs pair compatibility.
- The candidate selector can run with `--score-mode vocal|minus|both`; vocal and
  minus modes rank by their own score fields and blockers rather than legacy
  `overall_no_reference_score`.
- The minus candidate matrix builder aggregates existing result-judge,
  self-judge, selector, listen-preview, and GPU provenance evidence into a
  ranked route-selection artifact. It does not rescore audio or relax gates; it
  chooses the next safe action, such as `run_segment_router_for_self_blockers`,
  `candidate_ready_for_listen_preview_contract`, or
  `do_not_release_consider_leakage_repair_or_new_model_route`.
- The vocal candidate matrix builder aggregates existing vocal judge, selector,
  listen-preview, and available GPU provenance evidence into a ranked
  route-selection artifact. It keeps vocal selection mode-specific and chooses
  next actions such as `candidate_ready_for_listen_preview_contract`,
  `run_selector_and_listen_preview_contract`, or
  `accepted_for_listen_pack_not_release`.
- The segment planner must not use shared or pair `window_fitness` to pre-filter
  vocal-only or minus-only repairs. Vocal windows use vocal-score delta, minus
  windows use minus-score delta, and `both` windows require both stem deltas plus
  pair compatibility.

## Container Run

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_no_reference_score_matrix.py \
  tests/test_run_no_reference_windowed_judge.py \
  tests/test_run_no_reference_optimizer.py \
  tests/test_plan_no_reference_segment_blends.py \
  tests/test_judge_no_reference_segment_blend_preview.py \
  tests/test_build_no_reference_vocal_candidate_score_matrix.py \
  tests/test_build_no_reference_minus_candidate_score_matrix.py \
  -q
```

On Mac targets, run the same project tests through the approved Apple
`container` route instead of Docker Desktop, Docker daemon, or `docker compose`.
