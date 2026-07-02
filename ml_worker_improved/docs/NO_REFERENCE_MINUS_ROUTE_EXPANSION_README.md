# No-Reference Minus Route Expansion

STEM-2 expands and audits minus candidate route space without adding models,
downloads, training, cleanup/debleed/denoise, threshold relaxation, or release
publication.

## Discovery

```bash
docker compose run --rm ml-worker python tools/discover_no_reference_minus_routes.py \
  --output-dir /app/lab_artifacts/no_reference_minus_route_expansion
```

Outputs:

- `minus_route_discovery.json`
- `minus_route_discovery.jsonl`
- `minus_route_discovery_summary.md`

Discovery reads existing source-of-truth files only:

- `worker/model_registry.py`
- `tools/no_reference_candidate_job_registry.py`
- `tools/run_registered_model_candidate.py`

It does not mutate `worker/model_registry.py`. Missing model files,
unsupported CLI parameters, missing entrypoints, or forbidden cleanup/download
routes are marked unavailable. If the available minus route families are only
`htdemucs_ft` / `htdemucs_6s`, the summary reports
`route_space_limited=true`.

## Expansion

```bash
docker compose run --rm ml-worker python tools/run_minus_route_expansion.py \
  --input-audio "/host_testing_stuff_music/ytmp3free.cc_sub-zero-project-never-surrender-official-hardstyle-video-youtubemp3free.org.mp3" \
  --output-dir /app/lab_artifacts/no_reference_minus_route_expansion/track002_stem2_001 \
  --profile hard_case \
  --max-routes 6 \
  --max-runtime-sec 2400
```

Outputs:

- `minus_route_expansion_manifest.json`
- `minus_route_expansion_progress.jsonl`
- `minus_route_jobs.jsonl`
- `minus_route_candidate_pool.jsonl`
- `minus_route_judge_minus.jsonl`
- `minus_route_pair_compatibility.jsonl`
- `minus_route_failures.jsonl`
- `minus_route_expansion_summary.md`

Each candidate row records `route_id`, `candidate_config_id`, `target_mode`,
`minus_path`, optional `vocal_path`, source model/tool, generation status,
shape-normalization provenance, minus judge decision, blockers,
`accepted_minus`, and `release_ready=false`.

The progress JSONL is part of the contract. It records start/finish events for
route execution and judge stages, expected acceleration mode, selected route
ids, and safety flags. Registered model execution is GPU-required; the
windowed judge is explicitly marked as existing-audio CPU/DSP scoring with
`gpu_inference_skipped=true`.

Exploration/hard-case discovery may include an already registered Becruily
instrumental/vocals route:

```text
local_registered__becruily_instrumental_vocals__both__minus_hard_case
```

This route is not in the safe default profile. It is provenance-labeled as
`other_existing`, runs only when explicitly selected by hard-case exploration,
and still must pass the same minus result judge, windowed judge, pair
compatibility, and warning-window comparison gates before any listen handoff.

## Matrix-Aware Hard-Case Policy

`tools/no_reference_hard_case_route_policy.py` can consume one or more
`--candidate-score-matrix-json` files. Matrix evidence is control-plane only:
it never marks audio as release-ready. The policy derives hard-case triggers
from blocked matrix `next_action`, result judge blockers, and self-judge proxy
issues, then selects a locally available route from the registered job registry.
It also consumes `minus_hard_case_router.jsonl` when present, so router-level
blockers such as `pitch_guard_full_region_stalled`,
`scalar_score_only_pitch_guard_progress`,
`trained_adapter_full_region_still_blocked`,
`model_gap_requires_separator_training_objective`, and
`pitch_guard_damage_regression` stay machine-readable downstream instead of
being flattened into scalar score progress.

Full-region/model-gap blockers require a separator training objective and do
not select another conditional route:

```text
training_objective_required=true
route_allowed=false
route_block_reason=hard_case_requires_separator_training_objective
release_ready=false
```

Damage-regression blockers only select a route registered for
`generate_artifact_safe_variants`; if none is available the policy blocks
instead of falling back to a more aggressive minus-focused route.

Downstream controller gates must consume this artifact explicitly via
`--hard-case-route-policy-json`. When `training_objective_required=true`,
generic controller selector staging, runtime-smoke replay, and selector
shadow-batch fail closed with
`blocked_hard_case_requires_separator_training_objective`; when
`artifact_safe_route_required=true`, they only allow or aggregate routes listed
in `artifact_safe_candidate_route_ids`.

When a matrix-selected route is minus-only but a same-model `both` route is
locally available, the policy selects the `both` route so the next candidate
wave can produce a vocal+minus pair for the required gates. The policy also
tracks failed `model_name`/`source_model_id` values across all supplied matrices
and refuses to reselect exhausted route families. If no unfailed local route is
available, it writes:

```text
route_allowed=false
route_block_reason=matrix_hard_case_routes_already_failed_or_unavailable
release_ready=false
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_minus_route_expansion.py \
  --discovery-json /app/lab_artifacts/no_reference_minus_route_expansion/minus_route_discovery.json \
  --expansion-dir /app/lab_artifacts/no_reference_minus_route_expansion/track002_stem2_001 \
  --output-json /app/lab_artifacts/no_reference_minus_route_expansion/stem2_validation.json \
  --summary-md /app/lab_artifacts/no_reference_minus_route_expansion/stem2_validation_summary.md
```

Validation fails closed on `release_ready=true`, forbidden cleanup/debleed/
denoise routes, threshold relaxation, or training flags. A route can be marked
quality-improved only after the minus judge accepts it without hard blockers.
The validator scans all emitted machine-readable route artifacts:
`minus_route_candidate_pool.jsonl`, `minus_route_jobs.jsonl`,
`minus_route_pair_compatibility.jsonl`, and `minus_route_failures.jsonl`.

## STEM-3 Conditional Runtime Integration

STEM-3 does not promote any route to the default runtime profile. The validated
quality-improved route id:

```text
local_registered__htdemucs_6s__both__shared
```

can be used only by `tools/no_reference_hard_case_route_policy.py` as a
hard-case scoped measurement route. The autonomous runtime enables that route
only when all of these are true:

- `--enable-conditional-hard-case-routes true` was passed;
- `--conditional-route-policy hard_case_only` was passed;
- a hard-case trigger is present, such as `missing_safe_mode_candidate`,
  `severe_vocal_bleed_in_minus`, `severe_phase_collapse`,
  `severe_phase_comb_filtering`, no accepted minus candidates, one observed
  minus route family, or `route_space_limited=true`;
- the selected route is already locally available through the existing registry
  and can run without model download or registry edit.

The runtime writes:

```text
hard_case_route_policy.json
hard_case_route_policy_summary.md
```

If the conditional route does not produce an accepted minus candidate, the run
must remain `needs_more_candidates` or another blocked diagnostic status. It
must not be marked solved, published, or `release_ready`.

## Safety Contract

- `reference_required=false`
- `human_csv_required=false`
- `training_started=false`
- `separator_training_started=false`
- `model_downloaded=false`
- `model_registry_edited=false`
- `thresholds_relaxed=false`
- `cleanup_routes_added=false`
- `melody_transcription_ran=false`
- `recommendation_index_updated=false`
- `release_ready=false`
