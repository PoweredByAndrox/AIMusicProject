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
- `minus_route_jobs.jsonl`
- `minus_route_candidate_pool.jsonl`
- `minus_route_judge_minus.jsonl`
- `minus_route_pair_compatibility.jsonl`
- `minus_route_expansion_summary.md`

Each candidate row records `route_id`, `candidate_config_id`, `target_mode`,
`minus_path`, optional `vocal_path`, source model/tool, generation status,
shape-normalization provenance, minus judge decision, blockers,
`accepted_minus`, and `release_ready=false`.

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
