# No-Reference Harness Runtime Validation

`tools/run_no_reference_harness_runtime_validation.py` validates HARNESS-1
Pareto candidates on the real autonomous no-reference runtime. This is a
measurement layer only: it does not train, download models, edit the separator
registry, relax thresholds, add cleanup/debleed/denoise chains, run melody
transcription, update recommendation indexes, publish results, or mark
`release_ready`.

## Purpose

HARNESS-1 optimizer output is replay-only. This runner turns selected Pareto
configs into autonomous runtime CLI arguments, runs them on real mixed tracks,
collects runtime artifacts, and compares the resulting stem-quality counters
against the baseline candidate from `harness_candidate_eval.jsonl`.

Only Docker runtime evidence counts. The runner itself is intended to run
inside the `ml-worker` container and invokes:

```bash
python tools/run_autonomous_no_reference_pipeline.py ...
```

It does not require Codex at runtime.

## Command

```bash
docker compose run --rm ml-worker python tools/run_no_reference_harness_runtime_validation.py \
  --pareto-frontier-jsonl /app/lab_artifacts/no_reference_meta_harness/optimizer_001/harness_pareto_frontier.jsonl \
  --tracks \
    "/app/testing_stuff_music/ytmp3free.cc_be-nicegreenhorn-single-version-youtubemp3free.org.mp3" \
    "/app/testing_stuff_music/ytmp3free.cc_sub-zero-project-never-surrender-official-hardstyle-video-youtubemp3free.org.mp3" \
    "/app/testing_stuff_music/Totally Enormous Extinct Dinosaurs - Household Goods.mp3" \
    "/app/testing_flstudio_music/Chris Bulman & Sarju Jagai - No Escape.flac" \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --output-dir /app/lab_artifacts/no_reference_meta_harness/runtime_validation_001 \
  --max-pareto-candidates 3 \
  --max-runtime-sec-per-track 1800 \
  --measurement-only true \
  --require-audio-preview-ready true
```

The baseline is loaded from a sibling `harness_candidate_eval.jsonl` when
available. If no baseline exists, non-baseline candidate decisions are marked
`candidate_inconclusive`.

## Candidate Validation

Every candidate is validated with the existing
`validate_harness_candidate()` contract before any runtime call. Candidates are
rejected as `candidate_rejected_safety` if they attempt forbidden mutations:

- release-ready rules
- threshold relaxation
- cleanup/debleed/denoise
- model downloads or unsupported models
- model registry edits
- source-code generation
- safety blocker bypass

The learned critic can influence scoring only through the existing runtime; it
cannot bypass script blockers.

## Runtime Mapping

Harness fields map to autonomous runtime CLI fields:

- `candidate_profile` -> `--candidate-profile`
- `candidate_diversity_profile` -> `--candidate-diversity-profile`
- `max_candidates` -> `--max-candidates`
- `timeout_budget_policy.max_runtime_sec` -> bounded
  `--max-candidate-runtime-sec`
- `max_hard_case_extra_jobs` -> `--max-hard-case-extra-jobs`
- optional stage mutation `enable_optional_hard_case_route` ->
  `--enable-minus-hard-case-routing true`
- `--require-audio-preview-ready true` -> autonomous runtime
  `--require-preview-audio-for-preview-ready true`

The runtime profile is `safe` for safe candidates and `balanced` for balanced
or exploration candidate profiles, because the autonomous runtime currently
only accepts `safe|balanced` as top-level profile values.

## Outputs

The validation directory contains:

```text
runtime_validation_manifest.json
runtime_validation_summary.md
harness_runtime_candidate_results.jsonl
harness_runtime_track_results.jsonl
harness_runtime_progress.jsonl
runs/<candidate_id>/<track>/*
```

Each per-track runtime directory is expected to contain normal autonomous
runtime artifacts such as:

```text
run_manifest.json
runtime_summary.md
candidate_pool.jsonl
candidates/candidate_wave_summary.json
candidates/candidate_wave_jobs.jsonl
candidates/candidate_wave_failures.jsonl
vocal_windowed_judge.jsonl
minus_windowed_judge.jsonl
learned_critic_scores.jsonl
selected_candidates.jsonl
pair_compatibility.jsonl
preview_manifest.json
rendered_preview_audio.jsonl
preview_render_contract.json
preview_render_summary.md
post_render_judge.jsonl
learning_outcome.jsonl
music_intelligence_input_manifest.json
```

## Decisions

Allowed validation decisions are:

- `keep_baseline`
- `candidate_runtime_improved`
- `candidate_specialist_only`
- `candidate_rejected_runtime`
- `candidate_rejected_safety`
- `candidate_inconclusive`

A candidate improves stem quality when it has at least one of these changes
without safety regression:

- more validated rendered audio previews
- more accepted minus candidates
- fewer `severe_phase_collapse` blockers
- fewer `severe_vocal_bleed_in_minus` blockers
- more pair-compatible candidates

One improved track is a specialist-only route. Two or more improved tracks is a
runtime-improved candidate. Track002/sub-zero specialist candidates are reported
separately and must not become a global baseline automatically.

An increased `preview_render_failed_count` blocks runtime promotion. Existing
hard-case failures remain negative counters; more complete pairs without
`audio_preview_ready` evidence remains inconclusive unless another quality
metric improves without regressions.

## Preview Counters

`accepted_pair_ready=true` means selector + pair compatibility reached a safe
pair, but this is not a listenable preview.

`manifest_preview_ready=true` means the runtime manifest or preview manifest
reported a manifest/listen-preview status. It is informational only and is not
counted as preview success.

`audio_preview_ready=true` additionally requires at least one rendered preview
audio path in `rendered_preview_audio.jsonl`, a present
`preview_render_contract.json`, and validated preview status. This prevents
manifest-ready states from being counted as real audio-preview-ready results.

Legacy replay/runtime folders that do not contain the PREVIEW-1 contract are
marked `preview_contract_missing=true`; this is not faked as preview success.

The runtime and aggregate outputs include:

```text
accepted_pair_ready_count
manifest_preview_ready_count
audio_preview_ready_count
rendered_preview_audio_count
preview_render_failed_count
preview_render_skipped_count
preview_validation_failed_count
preview_contract_missing_count
```

`audio_preview_ready_count` is the primary preview success metric.
`manifest_preview_ready_count` is informational. `preview_render_failed_count`
`preview_validation_failed_count`, and `preview_contract_missing_count` are
negative metrics.

## Safety

The validation manifest always records:

```text
release_ready=false
reference_required=false
human_csv_required=false
training_started=false
separator_training_started=false
model_downloaded=false
model_registry_edited=false
thresholds_relaxed=false
cleanup_routes_added=false
melody_transcription_ran=false
recommendation_index_updated=false
```

If any runtime artifact reports `release_ready=true`, the validation keeps the
top-level `release_ready=false` and reports a non-zero `release_ready_count`.
