# No-Reference Systems-Inspired Harness Optimizer

This layer adapts optimization-system ideas to the local no-reference
vocal/minus runtime without copying those systems blindly and without changing
separator models.

It optimizes harness policies and JSON configs around existing local runtime
artifacts. It does not require Codex at runtime, reference stems, human CSV,
separator training, model downloads, unsupported models, threshold relaxation,
cleanup/debleed/denoise chains, `release_ready`, or final publishing.

## Systems Mapped

- OPRO: optional config proposer only; disabled by default and never writes
  Python source.
- TextGrad: deterministic failure reflections from replay events.
- GEPA: deterministic mutation over JSON harness candidates plus Pareto
  selection.
- AutoRAG: modular pipeline selection only, not document retrieval for audio.
- AFlow: safe stage graph over allowed runtime nodes.
- ADAS: config-space design search over existing safe modules.
- Meta-Harness: metadata-only replay archive and replay evaluation.
- AgentSquare: explicit module groups for planning, reasoning, tool use, and
  memory.

## Module Schema

Every harness candidate includes `module_config` with:

- Planning: `candidate_job_scheduler`, `profile_escalation`,
  `hard_case_router`, `timeout_budget_policy`
- Reasoning: `no_reference_windowed_judge`, `mode_specific_score_matrix`,
  `learned_critic`, `failure_reflection`
- Tool Use: `local_separator_job_runner`, `shape_normalizer`,
  `preview_renderer`, `post_render_judge`
- Memory: `replay_bundle`, `learning_outcomes`, `model_manifest`,
  `route_outcome_history`

## Stage Graph

Allowed nodes:

- `candidate_wave`
- `windowed_judge_vocal`
- `windowed_judge_minus`
- `learned_critic`
- `selector`
- `pair_compatibility`
- `hard_case_router`
- `segment_blend_planner`
- `dry_run`
- `preview_render`
- `post_render_judge`

Allowed graph mutations:

- enable/disable optional hard-case route
- reorder hard-case routes
- adjust fallback edge
- adjust profile escalation
- adjust retry count

Forbidden graph mutations:

- remove safety gate
- skip pair compatibility
- skip post-render judge after preview
- enable `release_ready`
- bypass blockers

The validator also checks edge endpoints, reachability through required safety
stages, unsafe direct edges into preview/post-render nodes, and cycles.

## Replay Bundle

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

- `replay_bundle_manifest.json`
- `replay_bundle_events.jsonl`
- `replay_bundle_artifacts_index.jsonl`

The bundle stores paths, hashes, metadata, and replay events only. Audio is not
copied.

## Optimizer

```bash
docker compose run --rm ml-worker python tools/optimize_no_reference_harness.py \
  --replay-bundle-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_001 \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/optimizer_001 \
  --generations 5 \
  --population-size 12 \
  --use-llm-proposer false
```

Output:

- `harness_candidate_eval.jsonl`
- `harness_pareto_frontier.jsonl`
- `harness_rejected_candidates.jsonl`
- `failure_reflections/failure_reflections.jsonl`
- `failure_reflections/failure_reflections_summary.md`
- `harness_optimizer_summary.json`
- `harness_optimizer_summary.md`

## Failure Reflections

```bash
docker compose run --rm ml-worker python tools/generate_no_reference_failure_reflections.py \
  --replay-bundle-dir /app/lab_artifacts/no_reference_meta_harness/replay_bundle_001 \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/failure_reflections_001
```

Output:

- `failure_reflections.jsonl`
- `failure_reflections_summary.md`

Each reflection records:

- `failure_id`
- `blocker_type`
- `failed_route`
- `evidence`
- `natural_language_diagnosis`
- `recommended_harness_change`
- `forbidden_change_check`

The recommended changes are harness-policy recommendations only. They cannot
relax thresholds, add cleanup/debleed/denoise, download models, edit registry,
edit Python source, bypass blockers, or enable `release_ready`.
