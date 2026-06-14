# No-Reference Meta-Harness Optimizer

`tools/build_no_reference_replay_bundle.py`,
`tools/replay_no_reference_harness_candidate.py`, and
`tools/optimize_no_reference_harness.py` implement a metadata-only replay
optimizer for the autonomous no-reference runtime harness.
`tools/generate_no_reference_failure_reflections.py` adds deterministic
failure reflections from replay events.

The optimizer mutates harness configuration, not separator code or separator
weights. It does not require reference stems, human CSV, Codex at runtime,
model downloads, threshold relaxation, cleanup/debleed/denoise chains, or
`release_ready`.

## Replay Bundle

Build a bundle from existing autonomous run folders:

```bash
docker compose run --rm ml-worker python tools/build_no_reference_replay_bundle.py \
  --run-roots \
    /app/lab_artifacts/autonomous_runs/runtime_6_track_002_minus_hard_case_v3 \
    /app/lab_artifacts/autonomous_runs/runtime_5_real_track_002_review_fix \
    /app/lab_artifacts/autonomous_runs/real_track_batch_001 \
    /app/lab_artifacts/autonomous_runs/real_track_batch_002 \
    /app/lab_artifacts/autonomous_runs/real_track_batch_003 \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_001
```

Output:

```text
replay_bundle_manifest.json
replay_bundle_events.jsonl
replay_bundle_artifacts_index.jsonl
```

The bundle stores paths, hashes, row counts, JSON keys, and replay events only.
Audio paths are indexed as `metadata_only`; audio is not copied into the bundle.

Required replay artifacts:

- `run_manifest.json`
- `runtime_summary.md`
- `candidate_pool.jsonl`
- `candidates/candidate_wave_summary.json`
- `candidates/candidate_wave_jobs.jsonl`
- `candidates/candidate_wave_failures.jsonl`
- `vocal_windowed_judge.jsonl`
- `minus_windowed_judge.jsonl`
- `learned_critic_scores.jsonl`
- `selected_candidates.jsonl`
- `pair_compatibility.jsonl`
- `preview_manifest.json`
- `post_render_judge.jsonl`
- `learning_outcome.jsonl`

Optional replay artifacts:

- `candidates/candidate_wave_manifest.json`
- `candidates_minus_hard_case/*`
- `minus_hard_case_router.jsonl`
- `learned_critic_disagreements.jsonl`
- `optimizer_mode_specific_result.jsonl`
- per-job `job_manifest.json`
- per-job `job_progress.jsonl`

Some files may be present but empty, especially `selected_candidates.jsonl`,
`pair_compatibility.jsonl`, and `post_render_judge.jsonl`. Empty files do not
mean a usable result exists.

If required artifacts are missing, the run is marked `replay_incomplete`.

## Harness Candidate

A harness candidate is a JSON config with:

- `harness_candidate_id`
- `parent_id`
- `mutation_type`
- `candidate_profile`
- `candidate_diversity_profile`
- `job_priority_policy`
- `shared_reuse_policy`
- `hard_case_route_order`
- `minus_selector_policy`
- `vocal_selector_policy`
- `pair_compatibility_policy`
- `window_judge_policy`
- `learned_script_hybrid_weight`
- `timeout_budget_policy`
- `max_candidates`
- `max_hard_case_extra_jobs`
- `forbidden_actions`
- `module_config`
- `stage_graph`

Allowed mutation fields are harness-policy fields only:

- job priority order
- hard-case route order
- safe/balanced/exploration profile within existing allowed configs
- timeout budget allocation
- max candidate count
- learned/script hybrid weight in `[0, 1]`
- selector preference order
- window selection policy

Forbidden mutation fields are rejected:

- safety blockers
- `release_ready` rules
- threshold relaxation
- cleanup/debleed/denoise
- model downloads
- unsupported models
- source-code generation
- model registry edits

`module_config` must include planning, reasoning, tool-use, and memory groups.
`stage_graph` must preserve safety, pair compatibility, dry-run, and
post-render-judge stages. Validation checks allowed nodes, forbidden mutations,
edge endpoints, required path reachability, unsafe direct preview/post-render
edges, and cycles.

## Optimizer

Run deterministic replay optimization:

```bash
docker compose run --rm ml-worker python tools/optimize_no_reference_harness.py \
  --replay-bundle-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_001 \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/optimizer_001 \
  --generations 5 \
  --population-size 12 \
  --use-llm-proposer false
```

Output:

```text
harness_candidate_eval.jsonl
harness_pareto_frontier.jsonl
harness_rejected_candidates.jsonl
failure_reflections/failure_reflections.jsonl
failure_reflections/failure_reflections_summary.md
harness_optimizer_summary.json
harness_optimizer_summary.md
```

Pareto selection does not choose by mean score alone. It considers:

- maximize `preview_ready_count`
- maximize `minus_accepted_count`
- maximize `pair_compatible_count`
- minimize severe blockers
- minimize failed runs
- minimize estimated runtime
- minimize complexity

A candidate that improves one track but hurts another is recorded as
`candidate_specialist_route_only` instead of replacing the baseline globally.

`--use-llm-proposer` is disabled by default. The current implementation is
deterministic and does not require LLM access.

## Failure Reflections

```bash
docker compose run --rm ml-worker python tools/generate_no_reference_failure_reflections.py \
  --replay-bundle-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_001 \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/failure_reflections_001
```

Output:

```text
failure_reflections.jsonl
failure_reflections_summary.md
```

Reflections are diagnostics and harness-policy recommendations only. They do
not relax thresholds, add cleanup/debleed/denoise, download models, edit the
registry, edit Python source, bypass blockers, or enable `release_ready`.
