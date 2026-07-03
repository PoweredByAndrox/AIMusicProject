# No-Reference Local Separator Job Wave

`tools/run_no_reference_candidate_wave.py` can now launch bounded local
separator jobs through existing project entrypoints. It does not require Codex,
reference stems, human CSV review, separator training, model downloads, model
registry edits, cleanup chains, threshold relaxation, or `release_ready`.

## Registry

Supported job configs are described by:

```text
tools/no_reference_candidate_job_registry.py
```

Each config records:

- `candidate_config_id`
- `candidate_family`
- `target_mode`
- `source_model_id`
- `source_tool`
- `entrypoint_type`
- `command_template`
- `expected_outputs`
- `gpu_required`
- `timeout_sec`
- `max_retries`
- `profile_allowed`
- `safety_notes`
- `supported_parameters`
- `shared_execution_group_id`
- `produces_vocal`
- `produces_minus`
- `produces_both`
- `materialized_candidate_modes`
- `reuse_outputs_across_modes`
- `source_config_hash`

The registry is read-only over `worker/model_registry.py`. If a local model or
entrypoint cannot run, the config is kept with
`entrypoint_type=unavailable`; runtime skips it instead of faking a candidate.
Chunk/overlap variants are listed only when they are supported by the existing
entrypoint. Unsupported requested parameters are recorded as
`unsupported_parameter:<name>` safety notes and the config stays unavailable.
Production minus configs must expose an explicit `minus` or `instrumental`
artifact in the registered stem manifest. Multi-stem Demucs exports that only
write `drums`/`bass`/`other`/`vocals` stay unavailable for the minus lane with
`forbidden_reason=multistem_output_without_explicit_minus_or_instrumental_manifest`;
they are not converted into a minus by summing non-vocal stems. The safe minus
candidate lane starts from dedicated vocal/instrumental routes such as
`audio_separator_bs_roformer_viperx_1297_vendor_config_vocals`.

Configs that can share one separator execution use the same
`shared_execution_group_id`. The scheduler runs one preferred config for that
group and records the skipped duplicates in `candidate_job_plan.jsonl`.

## One Job

```bash
docker compose run --rm ml-worker python tools/run_local_separator_candidate_job.py \
  --input-audio /app/input/song.wav \
  --candidate-config-id local_registered__htdemucs_ft__vocal \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001/candidates/jobs/local_registered__htdemucs_ft__vocal \
  --job-manifest-json /app/lab_artifacts/autonomous_runs/song_001/candidates/jobs/local_registered__htdemucs_ft__vocal/job_manifest.json \
  --job-progress-jsonl /app/lab_artifacts/autonomous_runs/song_001/candidates/jobs/local_registered__htdemucs_ft__vocal/job_progress.jsonl \
  --profile safe \
  --allow-shape-normalization true
```

The job runner writes:

- `job_manifest.json`
- `job_progress.jsonl`

Allowed job statuses:

- `generated`
- `generated_with_shape_normalization`
- `generated_with_late_timeout`
- `partial_output_timeout`
- `failed_timeout_before_any_output`
- `skipped_unavailable_entrypoint`
- `failed_missing_output`
- `failed_invalid_audio_shape_unrecoverable`
- `failed_runtime_error`

Generated audio is accepted only as a candidate input to later no-reference
judges. It is never marked `release_ready`.

Shape normalization is limited to resampling, channel matching, and tiny
duration trim/pad. Raw separator outputs are never overwritten; normalized
outputs are written separately and recorded in `normalized_vocal_path` /
`normalized_minus_path`.

## Candidate Wave

```bash
docker compose run --rm ml-worker python tools/run_no_reference_candidate_wave.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001/candidates \
  --candidate-pool-jsonl /app/lab_artifacts/autonomous_runs/song_001/candidate_pool.jsonl \
  --profile safe \
  --execution-mode auto \
  --max-candidates 4 \
  --max-runtime-sec 600 \
  --continue-on-failure true \
  --allow-shape-normalization true \
  --candidate-diversity-profile safe
```

Execution modes:

- `collect_existing`: materialize existing source rows only.
- `run_jobs`: run local separator jobs from the registry.
- `auto`: collect existing rows first, then run jobs if not enough complete
  pairs exist.

Candidate-wave outputs:

- `candidate_job_plan.jsonl`
- `candidate_pool.jsonl`
- `candidate_wave_materialized_assets.jsonl`
- `candidate_wave_summary.md`
- `candidate_wave_summary.json`
- `candidate_wave_jobs.jsonl`
- `candidate_wave_failures.jsonl`
- `candidate_wave_progress.jsonl`
- `candidate_wave_manifest.json`

Each candidate row records `candidate_config_id`, `candidate_family`,
`target_mode`, `source_model_id`, `source_tool`, `source_config_hash`,
`job_manifest_path`, `job_progress_path`, `audio_rendered`, raw/normalized
candidate paths, `shape_normalization_applied`, and `release_ready=false`.

Summary counters include:

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

## Safety

- Original input audio is never overwritten.
- Candidate outputs are written only under the current run directory.
- Failed and skipped jobs are diagnostic rows, not listen results.
- Partial job failures are recorded and may continue only when
  `--continue-on-failure true`.
- `release_ready_count` must remain `0`.
