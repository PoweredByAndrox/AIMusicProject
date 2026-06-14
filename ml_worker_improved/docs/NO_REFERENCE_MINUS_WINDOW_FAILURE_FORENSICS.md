# No-Reference Minus Window Failure Forensics

STEM-4 adds metadata-only forensics for no-reference minus hard cases. It does
not change the default runtime profile, train models, download models, relax
thresholds, add cleanup/debleed/denoise chains, run melody transcription, update
recommendation indexes, publish results, or mark anything `release_ready`.

## Inputs

Primary run:

```bash
/app/lab_artifacts/autonomous_runs/stem_3_track002_conditional_route_001
```

The tools read existing runtime artifacts only:

- `run_manifest.json`
- `runtime_summary.md`
- `candidate_pool.jsonl`
- `candidate_wave_jobs.jsonl`
- `candidate_job_plan.jsonl` when present
- `hard_case_route_policy.json`
- `hard_case_route_policy_summary.md`
- `minus_windowed_judge.jsonl`
- `vocal_windowed_judge.jsonl`
- `selected_candidates.jsonl`
- `pair_compatibility.jsonl`
- `preview_manifest.json` when present
- `music_intelligence_input_manifest.json`

Optional comparison runs:

- `/app/lab_artifacts/no_reference_minus_route_expansion/track002_stem2_001`
- `/app/lab_artifacts/autonomous_runs/stem_3_household_goods_conditional_route_001`

## Window Blocker Profiler

```bash
docker compose run --rm ml-worker python tools/analyze_minus_window_blockers.py \
  --run-dir /app/lab_artifacts/autonomous_runs/stem_3_track002_conditional_route_001 \
  --output-dir /app/lab_artifacts/no_reference_minus_forensics/track002_stem4_001
```

Outputs:

- `minus_window_blocker_profile.jsonl`
- `minus_window_blocker_summary.md`
- `minus_window_blocker_clusters.json`

Each profile row represents one candidate/window and includes route provenance,
window timing, minus judge decision, blockers, severe blocker booleans, blocker
signature, score components, raw/normalized minus paths, shape-normalization
metadata, and `release_ready=false`.

The cluster artifact groups evidence by candidate, route, and window. It also
marks whether failures are global across all windows, localized to some windows,
shared across routes, route-specific, source-model correlated, or correlated
with shape normalization.

## Route Window Comparison

```bash
docker compose run --rm ml-worker python tools/compare_minus_route_windows.py \
  --run-dir /app/lab_artifacts/autonomous_runs/stem_3_track002_conditional_route_001 \
  --control-run-dir /app/lab_artifacts/autonomous_runs/stem_3_household_goods_conditional_route_001 \
  --extra-run-dir /app/lab_artifacts/no_reference_minus_route_expansion/track002_stem2_001 \
  --output-dir /app/lab_artifacts/no_reference_minus_forensics/track002_stem4_001
```

Outputs:

- `minus_route_window_comparison.jsonl`
- `minus_route_window_comparison_summary.md`
- `minus_route_window_comparison_summary.json`

The comparison computes route-level window counts, accepted/blocked windows,
severe phase collapse, severe vocal bleed, phase-comb filtering, score range,
median score, blocker signatures, and whether a route improves any window.
`--extra-run-dir` is read as a comparison summary only; it does not add new
candidates or rerun audio.

Decision flags:

- `all_routes_same_blocker_signature`
- `any_route_has_clean_window`
- `any_route_reduces_phase_collapse`
- `any_route_reduces_vocal_bleed`
- `segment_blend_possible`
- `segment_blend_all_windows_covered`
- `route_space_exhausted`

`route_space_exhausted=true` requires all routes to share the same blocker
signature and no clean windows to exist. Localized clean windows therefore do
not become a release result; they only support a later bounded segment-repair
phase.

## Track002 Failure Report

```bash
docker compose run --rm ml-worker python tools/build_track002_minus_failure_report.py \
  --forensics-dir /app/lab_artifacts/no_reference_minus_forensics/track002_stem4_001 \
  --output-dir /app/lab_artifacts/no_reference_minus_forensics/track002_stem4_001
```

Outputs:

- `track002_minus_failure_report.json`
- `track002_minus_failure_report.md`
- `track002_next_route_decision.json`

Decision categories:

- `segment_blend_candidate`
- `local_route_candidate_needed`
- `route_space_exhausted`
- `normalization_issue_suspected`
- `judge_threshold_issue_suspected`
- `input_track_hard_case`
- `unsupported_new_model_needed`
- `blocked_no_safe_action`

Rules:

- Threshold relaxation is never recommended.
- Cleanup, debleed, and denoise chains are never recommended.
- New model download is never executed in this phase.
- `unsupported_new_model_needed` is only a decision flag.
- `release_ready=false`.
- `can_publish_as_result=false`.
- `melody_transcription_ran=false`.
- `recommendation_index_updated=false`.
