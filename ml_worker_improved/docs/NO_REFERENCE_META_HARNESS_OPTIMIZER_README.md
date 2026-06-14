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

### Preview-contract-complete replay set

For optimizer runs that must exclude ambiguous legacy preview artifacts, first
filter runtime folders to PREVIEW-1 contract-complete runs:

```bash
docker compose run --rm ml-worker python tools/filter_no_reference_replay_runs.py \
  --run-roots \
    /app/lab_artifacts/autonomous_runs/runtime_6_track_002_minus_hard_case_v3 \
    /app/lab_artifacts/autonomous_runs/runtime_5_real_track_002_review_fix \
  --output-dir /app/lab_artifacts/no_reference_meta_harness
```

Output:

```text
preview_contract_complete_runs.jsonl
preview_contract_incomplete_runs.jsonl
preview_contract_filter_summary.json
preview_contract_filter_summary.md
```

A run is complete only when it has `run_manifest.json`,
`preview_manifest.json`, `preview_render_contract.json`, and either validated
`rendered_preview_audio.jsonl` rows or an explicit
`preview_render_skipped`/`preview_render_failed`/`preview_blocked_safety_contract`
status with a reason. Legacy `preview_ready` without PREVIEW-1 fields is
incomplete. Any `release_ready=true`, `can_mark_release_ready=true`, or
`can_publish_as_result=true` anywhere under the run disqualifies it.

Legacy runs can be backfilled from already selected stems only:

```bash
docker compose run --rm ml-worker python tools/backfill_no_reference_preview_contract.py \
  --run-dir /app/lab_artifacts/autonomous_runs/<LEGACY_RUN_DIR> \
  --preview-duration-sec 30 \
  --preview-render-mode bounded
```

Backfill writes `preview_manifest.json`, `rendered_preview_audio.jsonl`,
`preview_render_contract.json`, and `preview_render_summary.md`. It does not
rerun separator jobs, does not modify `selected_candidates.jsonl`, does not
overwrite source stems, and keeps `can_publish_as_result=false`,
`can_mark_release_ready=false`, `melody_transcription_ran=false`,
`recommendation_index_updated=false`, and `release_ready=false`.

Build the clean replay bundle from the filtered JSONL:

```bash
docker compose run --rm ml-worker python tools/build_no_reference_replay_bundle.py \
  --run-roots-from-jsonl /app/lab_artifacts/no_reference_meta_harness/preview_contract_complete_runs.jsonl \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_audio_preview_complete_001
```

The clean bundle is expected to have `preview_contract_missing_count=0`.
Optimizer candidates should no longer be rejected only because the replay set
contains missing preview contracts.

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

`--run-roots` may point either to autonomous runtime run folders or to a
harness validation output root containing `runs/<candidate>/<track>/`.
The bundle expands those nested runtime folders and records the resolved list in
`expanded_run_roots`.

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
- `rendered_preview_audio.jsonl`
- `preview_render_contract.json`
- `post_render_judge.jsonl`
- `learning_outcome.jsonl`

Optional replay artifacts:

- `candidates/candidate_wave_manifest.json`
- `candidates_minus_hard_case/*`
- `minus_hard_case_router.jsonl`
- `learned_critic_disagreements.jsonl`
- `optimizer_mode_specific_result.jsonl`
- `preview_render_summary.md`
- per-job `job_manifest.json`
- per-job `job_progress.jsonl`

Some files may be present but empty, especially `selected_candidates.jsonl`,
`pair_compatibility.jsonl`, and `post_render_judge.jsonl`. Empty files do not
mean a usable result exists.

If required artifacts are missing, the run is marked `replay_incomplete`.

Preview replay fields use PREVIEW-1 semantics:

- `accepted_pair_ready` means a pair was selected/compatible, not listenable
  audio.
- `manifest_preview_ready` is informational and never counts as preview
  success by itself.
- `audio_preview_ready` requires `preview_render_contract.json`, validated
  `rendered_preview_audio.jsonl`, and `rendered_preview_audio_count > 0`.
- legacy runs without `preview_render_contract.json` are marked
  `preview_contract_missing=true`.
- `preview_render_failed`, `preview_validation_failed`, and
  `preview_contract_missing` are negative replay metrics.

Replay events and the bundle manifest keep `can_publish_as_result=false`,
`can_mark_release_ready=false`, `melody_transcription_ran=false`,
`recommendation_index_updated=false`, and `release_ready=false`.

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
efficiency_only_candidates.jsonl
no_op_candidates.jsonl
promotion_gate_summary.md
failure_reflections/failure_reflections.jsonl
failure_reflections/failure_reflections_summary.md
harness_optimizer_summary.json
harness_optimizer_summary.md
```

Pareto selection does not choose by mean score alone. Manifest-only preview
signals are informational and do not count as preview success. It considers:

- maximize `audio_preview_ready_count`
- maximize `rendered_preview_audio_count`
- maximize `accepted_minus_count` / `minus_accepted_count`
- maximize `pair_compatible_count`
- maximize `analysis_ready_stems_count`
- minimize `blocked_stems_count`
- minimize `preview_render_failed_count`
- minimize `preview_validation_failed_count`
- minimize `preview_contract_missing_count`
- minimize severe blockers (`severe_phase_collapse`,
  `severe_vocal_bleed_in_minus`)
- minimize failed runs
- minimize estimated runtime
- minimize complexity

Promotion is gated by quality first:

- `candidate_quality_improved` is the only global-promotion decision class.
- `best_global_candidate` is empty unless at least one quality metric improves.
- lower `estimated_runtime_sec` without a quality gain is
  `candidate_efficiency_only`, written to `efficiency_only_candidates.jsonl`.
- identical quality and non-improved runtime is `candidate_no_op`, written to
  `no_op_candidates.jsonl`.
- runtime/estimated-runtime improvements are tie-breaks and diagnostics, not
  baseline-promotion evidence by themselves.
- `manifest_preview_ready_count` is informational and never creates quality
  improvement without rendered `audio_preview_ready` evidence.
- any safety regression, `release_ready_count > 0`, or `release_ready=true`
  blocks promotion.

A candidate that improves one track but hurts another is recorded as
`candidate_specialist_only` instead of replacing the baseline globally.
More complete pairs without validated `audio_preview_ready` evidence remains
inconclusive. Candidates with preview render failures are not promoted by the
runtime validation layer. A candidate with `preview_contract_missing_count > 0`
is rejected from global Pareto promotion as
`candidate_rejected_preview_contract_missing`.

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
