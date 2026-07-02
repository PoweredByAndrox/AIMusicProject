# No-Reference Minus Hard-Case Router

`tools/no_reference_minus_hard_case_router.py` is a control-plane router for
minus candidates that fail no-reference judging after candidate generation has
already succeeded.

It does not score audio as safe, relax thresholds, run cleanup/debleed/denoise,
download models, train separators, or mark `release_ready`.

## Input

The router reads a runtime output directory containing:

```text
run_manifest.json
candidate_pool.jsonl
minus_windowed_judge.jsonl
selected_candidates.jsonl
pair_compatibility.jsonl
```

## Output

```text
minus_hard_case_router.jsonl
minus_hard_case_router_summary.md
```

Each JSONL row records:

- `track_id`
- `run_id`
- `blocker_type`
- `affected_candidate_id`
- `minus_candidate_count`
- `minus_candidate_families`
- `recommended_route`
- `route_reason`
- `can_run_without_new_models`
- `available_route_configs`
- `forbidden_actions`
- safety flags with `release_ready=false`

## Detected Blockers

- `severe_vocal_bleed_in_minus`
- `severe_phase_collapse`
- `severe_phase_comb_filtering`
- `pitch_guard_full_region_stalled`
- `scalar_score_only_pitch_guard_progress`
- `trained_adapter_full_region_still_blocked`
- `model_gap_requires_separator_training_objective`
- `pitch_guard_damage_regression`
- `missing_safe_mode_candidate`
- `only_one_minus_family`
- `low_minus_candidate_diversity`
- `pair_compatibility_not_reached`

## Routes

- `train_separator_attribution_objective`
- `generate_minus_focused_variants`
- `generate_phase_safe_variants`
- `generate_artifact_safe_variants`
- `segment_blend_needed`
- `blocked_hard_case`

The router only recommends routes backed by existing registry configs. If the
registry cannot run a recommended route, it reports the route as unavailable
instead of faking a candidate.

`train_separator_attribution_objective` is a diagnostic planning route for
cases where pitch-guard evidence shows scalar-score improvement but the full
failing region count stays blocked, trained adapter runs still leave all
regions blocked, or a Mac model-gap manifest explicitly requires a separator
training objective. It does not start training, publish audio, or bypass the
normal `release_ready=false` safety contract.

`generate_artifact_safe_variants` is preferred over minus-focused expansion
when guard rows report damage regression, because a louder or more aggressive
minus route is not useful if it clears leakage by damaging the instrumental.

The router can derive these pitch-guard blockers from `candidate_pool.jsonl` or
`minus_windowed_judge.jsonl` row metrics even when an upstream writer has not
emitted an explicit blocker string. If before/after pitch-guard region counts
are equal and non-zero, the row is treated as `pitch_guard_full_region_stalled`;
if the best scalar score also decreased, it is treated as
`scalar_score_only_pitch_guard_progress`.
Rows with `damage_regressed=true` or positive `damage_regression_rows` are
treated as `pitch_guard_damage_regression`. Run manifests with
`status=trained_adapter_improved_still_blocked` are normalized to
`trained_adapter_full_region_still_blocked`; manifests with
`schema=mac_melband_pitch_refiner_model_gap.v1` are normalized to
`model_gap_requires_separator_training_objective`.

## CLI

```bash
docker compose run --rm ml-worker python tools/no_reference_minus_hard_case_router.py \
  --run-dir /app/lab_artifacts/autonomous_runs/runtime_6_track_002_minus_hard_case
```

The autonomous controller can run this router when explicitly enabled:

```bash
--enable-minus-hard-case-routing true \
--max-hard-case-extra-jobs 4 \
--hard-case-runtime-sec 1800
```

The default is disabled for smoke runs.
