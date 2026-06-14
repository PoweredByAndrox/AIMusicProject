# Autonomous No-Reference Runtime

`tools/run_autonomous_no_reference_pipeline.py` is a local one-command
controller for the no-reference vocal/minus runtime. It does not require Codex
at runtime and it does not train or change separator models.

## Runtime Command

```bash
docker compose run --rm ml-worker python tools/run_autonomous_no_reference_pipeline.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001 \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --profile safe \
  --generate-candidates true \
  --candidate-profile safe \
  --candidate-execution-mode auto \
  --allow-shape-normalization true \
  --candidate-diversity-profile safe \
  --auto-escalate-profile false \
  --max-candidates 6 \
  --max-candidate-runtime-sec 600 \
  --continue-on-candidate-failure true
```

By default the controller runs `tools/run_no_reference_candidate_wave.py` and
writes:

```text
<output-dir>/candidate_pool.jsonl
```

If `<output-dir>/candidate_wave_sources.jsonl` exists, the controller passes it
to the candidate wave as local separator output metadata. With
`--candidate-execution-mode auto`, the wave first collects those existing rows
and then runs bounded local separator jobs if there are not enough complete
pairs. If jobs are unavailable or yield zero usable candidates, the runtime
stops safely with `needs_more_candidates`.

`--allow-shape-normalization true` is the default. It allows generated
separator candidates to be resampled, channel-matched, and tiny-duration
aligned to the input mix shape before judging, while preserving raw output
paths in provenance. It does not run cleanup, debleed, denoise, EQ, mastering,
or threshold relaxation.

`--candidate-diversity-profile` selects the local candidate registry profile:

- `safe`: 2-4 default local configs, no aggressive variants;
- `balanced`: adds supported local model/config variants and records
  unsupported parameter variants as unavailable;
- `exploration`: wider local registry surface, still no release-ready output.

Safe remains the default for smoke and quick tests. For real hard tracks, the
measured profile is `balanced`:

```bash
docker compose run --rm ml-worker python tools/run_autonomous_no_reference_pipeline.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_balanced \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --profile balanced \
  --generate-candidates true \
  --candidate-profile balanced \
  --candidate-execution-mode auto \
  --allow-shape-normalization true \
  --candidate-diversity-profile balanced \
  --auto-escalate-profile false \
  --max-candidates 8 \
  --max-candidate-runtime-sec 1800 \
  --continue-on-candidate-failure true
```

`--auto-escalate-profile true` lets a safe run launch a second balanced
candidate wave when safe produces zero or one complete pair, or when timeout
diagnostics show the safe pass is under-covered. Escalation only adds candidate
diversity; it does not change thresholds, judges, cleanup, training, or
release status.

Use `--candidate-pool-jsonl /app/path/candidate_pool.jsonl
--generate-candidates false` when you intentionally want to use a provided
pool. If the candidate wave or provided pool yields zero usable complete pairs,
the controller stops safely with `output_status=needs_more_candidates` and
still writes the required artifacts.

## Contract

- Codex is not required at runtime.
- Manual phase-by-phase execution is not required.
- Reference stems and human CSV labels are not required.
- Audio separator training, fine-tune, new separator models, registry edits,
  cleanup route additions, and threshold relaxation are not performed.
- Learned critic scoring is shadow/hybrid only. It cannot bypass script
  blockers.
- Script blockers are hard overrides.
- `release_ready` is always false.
- Original input/candidate files are never overwritten.
- Model manifest/version metadata is recorded in `run_manifest.json`.
- Local separator jobs use only existing project entrypoints and
  `worker/model_registry.py` entries. Unavailable configs are skipped, not
  faked.

## Stages

1. validate input audio
2. run candidate generation/collection wave, unless disabled for a provided pool
3. run vocal-mode windowed no-reference judge
4. run minus-mode windowed no-reference judge
5. run learned critic shadow/hybrid scoring
6. select best vocal candidate
7. select best minus candidate
8. run pair compatibility
9. if compatible, write preview/listen manifest for the selected existing stems
10. if incompatible, run segment blend planner
11. run dry-run render contract for segment plans
12. render bounded segment-blend preview only when a safe `render_ready`
    contract exists
13. run targeted post-render judge when preview render exists
14. write final runtime report
15. write learning outcome row

Allowed final statuses:

- `preview_ready`
- `accepted_for_listen_pack`
- `blocked_hard_case`
- `needs_more_candidates`
- `needs_pair_reconciliation`
- `segment_blend_needed`
- `failed_safety_contract`

## Output Files

```text
run_manifest.json
runtime_summary.md
selected_candidates.jsonl
learned_critic_scores.jsonl
pair_compatibility.jsonl
preview_manifest.json
post_render_judge.jsonl
learning_outcome.jsonl
```

Every run writes machine-readable artifacts. If a stage fails, downstream
artifacts are written as empty/blocked diagnostics rather than silently omitted.

Candidate-wave diagnostics are written next to the runtime artifacts:

```text
candidate_pool_invalid_rows.jsonl
candidates/candidate_wave_failures.jsonl
candidates/candidate_wave_jobs.jsonl
candidates/candidate_job_plan.jsonl
candidates/candidate_wave_materialized_assets.jsonl
candidates/candidate_wave_progress.jsonl
candidates/candidate_wave_summary.md
candidates/candidate_wave_summary.json
candidates/candidate_wave_manifest.json
```

Candidate-wave summaries include:

- `allow_shape_normalization`
- `candidate_diversity_profile`
- `registry_configs_total`
- `registry_configs_available`
- `registry_configs_unavailable`
- `shared_jobs_planned`
- `duplicate_jobs_avoided`
- `scheduler_skipped_jobs`
- `jobs_attempted`
- `jobs_generated`
- `jobs_generated_raw`
- `jobs_generated_with_shape_normalization`
- `jobs_generated_with_late_timeout`
- `jobs_failed_timeout`
- `partial_outputs_recovered`
- `jobs_failed_shape_unrecoverable`
- `materialized_candidate_rows`
- `complete_pairs`
- `vocal_usable_candidates`
- `minus_usable_candidates`
- `pair_usable_candidates`
- `candidate_families_generated`
- `target_modes_generated`
- `release_ready_count`

Candidate execution modes:

- `collect_existing`: use existing local separator output metadata only.
- `run_jobs`: launch local separator jobs from the read-only candidate job
  registry.
- `auto`: collect existing rows first, then launch jobs when more candidates
  are needed.

## Learning Cycle

```bash
docker compose run --rm ml-worker python tools/run_no_reference_learning_cycle.py \
  --outcomes-root /app/lab_artifacts/autonomous_runs \
  --current-model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --output-model-dir /app/lab_artifacts/no_reference_ml_critic/models_next \
  --eval-output-dir /app/lab_artifacts/no_reference_ml_critic/eval_next
```

The learning cycle reads `learning_outcome.jsonl` files, excludes raw model
predictions, and trains only from rows with:

```json
{"verified_outcome": true, "raw_prediction_only": false}
```

Rows written by the autonomous runtime are raw diagnostic outcomes by default
and are not trainable until an external verification process marks them as
verified.

When verified rows exist, the learning cycle may train the local metadata
critic (`critic_training_started=true`). It still keeps
`separator_training_started=false`, does not fine-tune audio separators, and
does not add separator models.

## Promotion

```bash
docker compose run --rm ml-worker python tools/promote_no_reference_critic_model.py \
  --comparison-json /app/lab_artifacts/no_reference_ml_critic/eval_next/evaluation_comparison.json \
  --candidate-model-dir /app/lab_artifacts/no_reference_ml_critic/models_next \
  --active-model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --output-dir /app/lab_artifacts/no_reference_ml_critic/promotion
```

Promotion is refused if:

- false accept rate increases;
- blocker safety regresses;
- pairwise ranking accuracy regresses;
- disagreement outcome accuracy regresses;
- the comparison is not approved.

The promotion tool writes `promotion_manifest.json` and copies model files only
when all gates pass.

## Docker Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_run_autonomous_no_reference_pipeline.py \
  tests/test_run_no_reference_learning_cycle.py -q
```
