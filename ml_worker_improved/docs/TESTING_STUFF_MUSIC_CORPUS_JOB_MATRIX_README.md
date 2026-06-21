# Testing Stuff Music Corpus Job Matrix

`tools/build_testing_stuff_music_corpus_job_matrix.py` converts the
Docker-validated `testing_stuff_music` corpus plan into deterministic
candidate-generation jobs for:

- `vocal`
- `minus`
- `melody`
- `drums`
- `bass`
- `other`

This is a planning/provenance route only. It does not execute model inference,
read audio content, copy audio, export stems, train, fine-tune, edit the model
registry, relax thresholds, or mark anything `release_ready`.

## Contract

- CLI execution is Docker-only.
- Input rows should come from
  `testing_stuff_music_corpus_plan.jsonl`, whose audio paths are Docker-visible.
- `minus` uses the documented clean-minus baseline entrypoint
  `tools/export_instrumental_resurrection_minus.py`; it does not route through
  `build_vocal_guided_minus.py`.
- Registered model routes are planned through
  `tools/run_registered_model_candidate.py` so candidate manifests and execution
  manifests remain reproducible. Matrix rows pass `--target-stem` to the
  registered runner; completed registered jobs must expose the requested target
  stem in the candidate manifest.
- The default registry breadth is intentionally broad (`50` models per requested
  stem) so the selection matrix does not silently hide available vocal/minus/
  melody/stem candidates after one or two bad routes. Runtime remains bounded by
  the executor's required `--max-jobs`. `--max-models-per-stem` can intentionally
  narrow planning for a diagnostic run.
- Every job is `planned_only=true` and `execution_started=false`.
- Every job is blocked if the Docker audio path is missing or if required model
  dependencies are unavailable.
- Some registry entries can require an explicit runtime-probe success marker in
  `required_paths`. For example, `bs_mamba2_vocals` must remain unavailable
  until a bounded Docker probe creates its `runtime_probe_passed.json`; package
  imports alone are not enough because previous BS-Mamba2 stacks reached model
  load and then crashed before `stem_manifest.json`.
- Registry entries can also expose same-checkpoint config variants when the
  local assets contain a distinct reproducible sidecar config. For example,
  `audio_separator_bs_roformer_viperx_1297_vendor_config_vocals` uses the same
  ViperX-1297 checkpoint as `audio_separator_bs_roformer_vocals` but the vendor
  YAML under `/app/vendor/bs_roformer_viperx_1297`; this is planned and judged
  as a separate candidate route, not reported as a separate model-family score.
- Instrumental-focused registered routes can be planned for `minus` when they
  expose a benchmark-compatible `instrumental` stem. For example,
  `becruily_instrumental_vocals` uses the shared MelBand full-track exporter
  with `/app/vendor/becruily_instrumental` checkpoint/config paths. It is a
  reproducible minus/instrumental candidate only; it still requires downstream
  judge, selector, artifact contract, and stem QC gates before any listen-pack
  decision.
- Optional route priors from `build_no_reference_corpus_route_priors.py` can
  block routes that repeatedly failed mode-specific no-reference judging. These
  jobs stay in the matrix as `job_status=blocked_route_prior`; they are not
  executed and they carry the matching prior evidence.
- If a blocked minus prior also has `source_pool_eligible=true`, the matrix
  preserves that evidence in the blocked row's `route_prior` and emits a
  `source_pool_phase_repair_candidate` row for
  `tools/run_no_reference_minus_phase_repair.py`. This is an executable
  CPU-DSP candidate-generation route, not neural inference. It generates
  bounded stereo phase variants and still requires downstream minus-mode judge,
  selector, artifact contract, and listen/delivery gates before any result can
  be accepted.
- When route priors block the direct clean-minus baseline, the matrix also
  plans registered `instrumental` candidates for `minus`. Route-prior-blocked
  registered candidates are treated as already evaluated, so the planner chooses
  the next unblocked registry candidate instead of re-queueing a known bad
  route. Four-stem models that expose `drums`, `bass`, and `other` outputs are
  eligible for `minus` planning even when their registry target list does not
  contain `instrumental`, because `run_registered_model_candidate.py`
  materializes `minus.flac` from non-vocal stems and records that provenance. If
  the requested target stem cannot be exposed, execution fails closed instead of
  presenting the manifest as a valid candidate.
- Every job requires downstream gates before any listen or delivery decision:
  candidate manifest contract, mode-specific score matrix, artifact status
  contract, and stem-specific QC gates.

## Outputs

- `testing_stuff_music_corpus_job_matrix.jsonl`
- `testing_stuff_music_corpus_job_matrix_manifest.json`
- `testing_stuff_music_corpus_job_matrix_summary.md`
- `testing_stuff_music_corpus_job_matrix_progress.jsonl`

The manifest is diagnostic only:

- `delivery_allowed=false`
- `release_status=not_release_ready`
- `release_ready=false`
- `training_started=false`
- `fine_tune_started=false`

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_testing_stuff_music_corpus_job_matrix.py \
  --corpus-plan-jsonl /app/lab_artifacts/testing_stuff_music_corpus_plan_20260611/testing_stuff_music_corpus_plan.jsonl \
  --output-root /app/lab_artifacts/testing_stuff_music_corpus_job_matrix_20260611 \
  --route-priors-jsonl /app/lab_artifacts/testing_stuff_music_corpus_route_priors_20260611/no_reference_corpus_route_priors.jsonl \
  --max-models-per-stem 50 \
  --device cuda
```

The next production step is a bounded executor that consumes this job matrix,
runs only `execution_ready=true` jobs with structured progress logs, then feeds
created manifests into mode-specific no-reference judge/optimizer/delivery
gates.

## Bounded execution

`tools/run_testing_stuff_music_corpus_job_matrix.py` consumes the matrix and
runs a deliberately limited subset of `execution_ready=true` jobs. It is still
candidate generation only; it does not accept listen artifacts and it does not
mark release readiness.

Executor contract:

- CLI execution is Docker-only.
- `--max-jobs` is required so corpus execution cannot silently run the whole
  matrix.
- `--stem`, `--track-id`, and `--model-id` can narrow the subset.
- When `--stem` is provided, bounded selection first chooses at least one
  execution-ready job for each requested stem, in requested order, before using
  remaining capacity for extra model candidates. This prevents early multi-model
  rows for one stem from starving vocal/minus/melody/drums/bass/other coverage.
- `--dry-run` records the bounded selection without launching commands.
- real execution writes stdout/stderr logs per job.
- every completed job must create its expected candidate manifest.
- blocked jobs are skipped fail-closed and are not executed.
- before subprocess execution, the executor revalidates each selected job:
  route contract, allowlisted entrypoint, source/output/manifest paths,
  registered model availability, and direct minus baseline availability.
- real execution also fail-fast blocks `gpu_inference_required=true` jobs when
  CUDA is unavailable inside the Docker runtime. Dry-runs and injected test
  command runners do not require a live GPU, but production candidate
  generation must not silently fall back to CPU for GPU-capable routes.
- stale or tampered rows are blocked as `blocked_preflight_contract` or
  `blocked_unavailable_at_execution`.
- `--fail-fast` records remaining selected jobs as `not_run_fail_fast` instead
  of dropping them from provenance.
- downstream gates remain mandatory before delivery:
  no-reference judge, mode-specific score matrix, vocal/minus candidate score
  matrices for route selection, optimizer/selector, artifact contract, and
  stem-specific QC.

## Minus judge input pairing

`tools/build_testing_stuff_music_corpus_minus_judge_inputs.py` builds
diagnostic rows for `run_no_reference_windowed_judge --score-mode minus` from
completed execution results.

Pairing contract:

- a `minus`/`instrumental` execution provides the `minus_path`;
- a separate `vocal`/`vocals` execution for the same `track_id` may provide the
  vocal companion path;
- a vocal execution is companion-only unless its manifest contains an explicit
  `minus` or `instrumental` stem path;
- the builder must never use a vocal target audio path as `minus_path`;
- rows where `minus_path == vocal_path` are skipped fail-closed;
- output rows remain diagnostic only with `delivery_allowed=false` and
  `release_ready=false`.

The minus candidate score matrix consumes the windowed judge output after this
pairing step. A candidate with `minus_decision=accept_candidate` is still
blocked when the pair gate reports `pair_decision=blocked_*` or non-empty
`pair_blockers`; this prevents matrix selection from promoting a good isolated
minus that is incompatible with its vocal companion.

Cross-product candidate rows may carry a JSONL candidate-pool path as
`candidate_manifest` provenance. The minus score matrix treats `.jsonl`
manifests as provenance only, not as candidate identity: `canonical_candidate_id`
stays on the row's `candidate_id`, and output rows expose
`manifest_kind=jsonl_candidate_pool`. This prevents multiple cross-pairs from
collapsing into one matrix record when they share the same source pool.

Executor outputs:

- `rows.jsonl`
- `manifest.json`
- `summary.md`
- `progress.jsonl`
- `skipped.jsonl` when blocked jobs are encountered

Diagnostic-only status:

- `delivery_allowed=false`
- `presentation_status=diagnostic_execution_not_for_listen_pack`
- `release_status=not_release_ready`
- `release_ready=false`
- `training_started=false`
- `fine_tune_started=false`

Dry-run manifests use `execution_status=dry_run`; real execution manifests use
`completed`, `failed`, `partial`, `blocked_preflight`, or
`blocked_no_selected_jobs`.

Dry-run example:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_testing_stuff_music_corpus_job_matrix.py \
  --job-matrix-jsonl /app/lab_artifacts/testing_stuff_music_corpus_job_matrix_20260611/testing_stuff_music_corpus_job_matrix.jsonl \
  --output-root /app/lab_artifacts/testing_stuff_music_corpus_job_execution_dry_run_20260611 \
  --stem vocal \
  --max-jobs 1 \
  --dry-run
```

Real bounded example:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_testing_stuff_music_corpus_job_matrix.py \
  --job-matrix-jsonl /app/lab_artifacts/testing_stuff_music_corpus_job_matrix_20260611/testing_stuff_music_corpus_job_matrix.jsonl \
  --output-root /app/lab_artifacts/testing_stuff_music_corpus_job_execution_20260611 \
  --stem vocal \
  --max-jobs 1 \
  --fail-fast
```
