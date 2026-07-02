# Generic Fine-Tuning Corpus

This route implements the first production-safe layer from
`docs/generic_finetuning_playbook`.

## What It Does

`tools/build_generic_finetuning_corpus_from_lab_artifacts.py` scans existing
`lab_artifacts` manifests and auto-discovers feedback JSONL files whose names
contain `feedback`, `human`, `review`, or `validated_review_decisions`. It then
writes a reproducible fine-tuning corpus:

- `inventory.jsonl`
- `controller_decision_examples.jsonl`
- `audio_training_examples.jsonl`
- `hard_eval_cases.jsonl`
- `excluded_examples.jsonl`
- `generic_finetuning_corpus_manifest.json`

The builder is metadata-only. It does not run model inference, does not copy
audio, does not start training, and does not mark release-ready artifacts.

Accepted listen-preview artifacts are not automatically audio-training
targets. Human-positive feedback is sufficient only for controller metadata
training unless the source manifest is an accepted
`machine_stem_acceptance_gate.v1` with the aggregate validation, required-stem
coverage, machine-quality, pair-compatibility, and minus-audit gates present.
Reference-verified manifests and verified `ground_truth_manifest.json` bundles
remain eligible through the reference path. Audio examples become trainable only
when that post-gate/reference provenance exists and the row has a cleared
`permission_status` such as `owned` or `licensed`; otherwise the row is written
to `excluded_examples.jsonl` as
`accepted_but_missing_accepted_machine_gate_provenance`.

Machine failure labels win over accepted-looking status fields. Nested
`dominant_failure_reason`, `failure_reason`, `reasons`, `failure_reasons`,
`blocking_issue_labels`, `issue_labels`, `blockers`, `blocking_reasons`, and
`failed_checks` values are normalized into `failure_labels`; labels such as
`foreign_source_recovered:*`, `mix_conservation_failed`, vocal leak/bleed,
contamination, residual, delivery pair, separator-server, or GPU-memory
blockers route the artifact to `hard_eval_cases.jsonl` and force controller
targets toward `avoid_route_and_generate_alternative` instead of
audio-positive training. Human negative feedback without machine labels gets a
synthetic `human_negative_feedback:<rating>` hard-eval reason so hard-eval rows
never lose their rejection cause.

Verified `ground_truth_manifest.json` bundles are imported as audio-training
examples only when `ground_truth_contract.verified=true`, the rights/permission
status is trainable, and `full_mix_reconstruction_contract.status=accepted`
with `can_claim_source_mix_1_to_1=true`. `user_owned` is normalized to
`project_owned`. If the source mix cannot be claimed as a one-to-one sum of
imported stems, the bundle remains diagnostic/reference evidence only and is
written to `excluded_examples.jsonl` with
`ground_truth_reconstruction_not_trainable`.
Reference intake and raw-reference recovery material builders also reject
non-production artifact roots such as `dbg_loop`, `dbg_loop_tmp`, `test_*`,
smoke, debug, or fixture paths under `lab_artifacts`; those rows stay forensic
evidence with `candidate_root_non_production_artifact` and cannot become
trainable ground truth.

Auto-discovered feedback paths are recorded in
`generic_finetuning_corpus_manifest.json` under `inputs`. Use
`--no-auto-discover-human-feedback` for forensic runs that must consume only
explicit `--human-feedback-jsonl` files.

## Controller Training

`tools/train_generic_finetuning_controller_model.py` trains a metadata route
controller from `controller_decision_examples.jsonl`. The current model is a
Bayesian-smoothed route prior controller over `(stem_type, route_id)`.

It writes:

- `generic_finetuning_controller_model.json`
- `generic_finetuning_controller_eval_report.json`
- `generic_finetuning_controller_training_manifest.json`

This is a training artifact, not an audio result. Promotion is blocked until an
external eval/smoke route proves it improves route selection without regressions.

`tools/eval_generic_finetuning_controller_model.py` is that first external eval
gate. It replays a controller model on a separate controller-example JSONL,
writes an external eval manifest/report, and blocks promotion if the eval JSONL
is the same file used for training, if too few examples are present, if routes
are missing from the model, if eval examples overlap training manifest paths, or
if accuracy is below threshold. Passing this gate still does not edit model
registries; it only marks the controller artifact as ready for the next runtime
smoke-replay step.

`tools/build_qwen_controller_sft_dataset.py` converts
`controller_decision_examples.jsonl` plus an optional external eval JSONL into a
Qwen chat SFT dataset for a future controller fine-tune. It writes short
Windows-safe artifact names (`train.jsonl`, `eval.jsonl`, `excluded.jsonl`,
`manifest.json`, `progress.jsonl`) and remains metadata-only:
`training_started=false`, `fine_tune_started=false`,
`delivery_allowed=false`, and `release_ready=false`. The builder blocks
train/eval manifest overlap by normalized manifest path before any Qwen/QLoRA
training can start, excludes source rows with an unexpected controller schema or
with top-level/nested `delivery_allowed=true` / `release_ready=true` claims, and
preserves manifest provenance in each example.

`tools/cache_qwen_controller_model.py` is the reproducible Docker-side cache
gate for `Qwen/Qwen3-4B-Thinking-2507`. By default it performs an offline
local-cache check only. Passing `--allow-network` explicitly allows the tool to
override `HF_HUB_OFFLINE` for that process, call Hugging Face
`snapshot_download`, and validate the resulting snapshot before training is
eligible to start. It writes short artifact names (`manifest.json`,
`progress.jsonl`), records network/offline state, emits download heartbeats
during network runs, validates `config.json`, tokenizer files, and weight files,
and remains metadata-only: `training_started=false`,
`model_registry_edited=false`, `delivery_allowed=false`, and
`release_ready=false`.

`tools/qwen_controller_qlora_preflight.py` is the fail-closed runtime gate for
that future Qwen controller fine-tune. It reads the Qwen SFT dataset manifest,
recomputes train/eval manifest-path overlap, checks required QLoRA Python
modules (`transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`,
`trl`, `sentencepiece`, `protobuf`, and `torch`), verifies the PyTorch-only
Transformers/TRL API surface needed by the trainer, and checks CUDA
availability plus a configurable free-memory guard. The Qwen route sets
`USE_TF=0` and `TRANSFORMERS_NO_TF=1` before importing Transformers/TRL so the
shared TensorFlow/Keras worker environment does not break a PyTorch-only
controller fine-tune. It also verifies that the target Qwen model snapshot is
already available in the local Hugging Face cache with offline/local-files-only
semantics, so training cannot start and then fail on a missing model download
while `HF_HUB_OFFLINE=1`. Non-CUDA requested devices and out-of-range `cuda:N`
requests block with a structured manifest instead of falling through to device 0
or crashing. It writes short artifact names
(`manifest.json`, `preflight.json`, `progress.jsonl`) and remains preflight-only:
`training_started=false`, `fine_tune_started=false`,
`models_changed=false`, `delivery_allowed=false`, and `release_ready=false`.
If the Docker image has not been rebuilt with the QLoRA requirements, the
preflight must return `preflight_status=blocked` instead of starting training or
falling back to CPU. If the Qwen snapshot is not cached, it must return
`model_cache_unavailable` instead of allowing training to attempt a network
download. If the installed QLoRA libraries drift away from the expected trainer
API, it must return `qlora_runtime_api_incompatible`.

The minimal 12GB-class GPU smoke route uses `--min-cuda-free-memory-mb 10000`,
`--max-length 256`, `--lora-r 8`, `--lora-alpha 16`, and `--max-steps 1` to
prove the Docker/GPU QLoRA loop is runnable before any longer training run. A
smoke-trained adapter is still accepted only if the external eval gate beats the
base-model baseline; otherwise it remains a blocked training artifact.

`tools/qwen_controller_qlora_train.py` is the gated QLoRA training entrypoint
for the Qwen controller adapter. It requires both the Qwen SFT dataset manifest
and a `qwen_controller_qlora_preflight` manifest with `preflight_status=passed`.
If either gate fails, it writes short artifact names (`manifest.json`,
`metrics.json`, `progress.jsonl`, `adapter/`, `trainer/`) with
`training_status=blocked` and does not import the QLoRA runtime. A completed run
still remains a training artifact only: `model_registry_edited=false`,
`delivery_allowed=false`, `release_ready=false`, and promotion stays blocked
until an external eval gate accepts the adapter.

`tools/qwen_controller_qlora_eval.py` is that external eval gate. It requires a
`qwen_controller_qlora_training` manifest with `training_status=trained`, an
existing `adapter/` directory, and non-overlapping eval rows from the Qwen SFT
dataset. The gate evaluates the base model and the adapter on the same eval
rows, writes `manifest.json`, `rows.jsonl`, and `progress.jsonl`, and only
passes when adapter accuracy meets threshold and is strictly better than the
base-model baseline. Passing eval still does not edit model registries or mark
anything release/listen-ready.

`tools/build_generic_controller_external_eval_from_hard_cases.py` converts
`hard_eval_cases.jsonl` into a controller external-eval JSONL. It includes only
`split=hard_eval` rows by default, skips rows with no route/model identity, and
writes a skipped-row report instead of turning unknown routes into useful eval
data. `--include-unknown-route` exists only for explicit diagnostic runs.
The resulting eval set is metadata-only and must still be checked by
`eval_generic_finetuning_controller_model.py` with the controller training
manifest so manifest overlap blocks fake external promotion.

If a clean controller has no coverage for hard-case routes,
`tools/build_generic_controller_hard_case_train_eval_split.py` creates a
deterministic disjoint train/holdout split from hard cases with route identity.
Singleton routes go to train only; routes with enough examples keep one holdout
example and train on the rest. It writes an augmented controller-training JSONL
and a holdout external-eval JSONL, with a manifest gate proving that training
and holdout manifest paths do not overlap.

`tools/run_generic_finetuning_controller_runtime_smoke_replay.py` is the next
metadata-only gate after external eval. It replays a controller model on
controller examples, writes per-example predicted actions, and blocks staging if
routes are missing, too few examples are replayed, or action accuracy is below
threshold. A pass still does not edit the model registry; it only produces a
runtime-smoke artifact for controlled integration review.

`tools/stage_generic_finetuning_controller_with_no_reference_selector.py` is the
first controlled integration layer for the existing no-reference selector. It
loads no-reference judge rows, runs the existing selector, overlays the trained
controller priors on selector-chosen route IDs, and writes proposed
allow/block/manual-review actions. It is still metadata-only: it does not render
audio, does not edit the registry, and does not promote a controller as default.
Exact controller route matches are preferred. When the selector emits a base
route but the controller only has composite route evidence, the runner may use a
provenance-visible alias aggregate from controller priors whose route IDs end in
`__<base_route>`. The decision row records `controller_route_match_kind`,
`controller_matched_route_id`, and `controller_source_route_ids`; this is not a
silent fallback.
If the selector's top acceptable candidate is blocked by the controller, the
runner may rerank only within the existing selector-acceptable candidates and
choose the first candidate whose controller prior passes the accept threshold.
It records `selector_original_selected_candidate_id`,
`controller_rerank_applied`, `controller_rerank_from_candidate_id`, and
`controller_rerank_evaluated_candidates`. It does not make a blocked
no-reference candidate usable and does not relax score, confidence, blocker, or
segment-blend gates.

`tools/run_generic_controller_selector_shadow_batch.py` scales that integration
gate to a batch. It accepts repeated `--judge-jsonl` inputs or discovers judge
files under `--judge-root` with `--judge-glob`, runs the single-file staging
runner for each input, and aggregates allow/block/missing-route outcomes into a
shadow-batch manifest. The batch is metadata-only, writes a structured progress
trail, and remains fail-closed: missing controller routes or too few checked
candidates block staging. It still does not render audio, edit registries,
start fine-tuning, or create listen/release artifacts.

`tools/plan_generic_controller_selector_route_gaps.py` consumes a shadow-batch
manifest and all referenced single-case integration decisions. It aggregates
controller-blocked, missing-prior, and allowed route observations, then writes a
route-gap plan for the next candidate/training wave. Repeated controller-blocked
routes become `generate_new_candidate_wave`; missing priors become
`collect_controller_prior_evidence`; allowed routes are kept available. The plan
is metadata-only and cannot start training, edit registries, relax thresholds,
or mark any artifact release-ready.

`tools/plan_generic_controller_selector_candidate_wave.py` consumes that
route-gap plan and writes execution-ready candidate-wave job rows for the
existing Docker routes. Registered model routes use
`/app/tools/run_registered_model_candidate.py`; the clean Instrumental
Resurrection minus baseline uses
`/app/tools/export_instrumental_resurrection_minus.py`. The planner validates
Docker-visible source audio paths, records GPU inference intent for the later
execution step, and stays metadata-only: it does not execute the commands, start
training, edit registries, or create listen/release artifacts.

`tools/run_generic_controller_selector_candidate_wave_jobs.py` is the bounded
execution wrapper for that plan. It normalizes `planned` candidate-wave rows
into the existing corpus job-matrix execution contract, then reuses the shared
preflight, CUDA, command, log, and candidate-manifest checks. Use `--dry-run`
first to prove the queue shape, then run a small `--max-jobs` batch. Execution
artifacts remain diagnostic until downstream score-matrix, selector, and
artifact-status gates accept them.

## Audio Fine-Tune Planning

`tools/plan_generic_audio_finetune_jobs.py` reads
`audio_training_examples.jsonl` and plans GPU jobs for the existing
`tools/train_teacher_mask_separator.py` route.

The planner does not start training. It requires a minimum number of unique
verified tracks per scope before a job can become `ready_for_gpu_training`;
otherwise it writes `blocked_insufficient_unique_tracks`. This prevents
overfitting a separator to one ground-truth bundle and calling it a production
model.

Ready job commands call `train_teacher_mask_separator.py` with the real trainer
CLI (`--source-mix`, `--teacher-target`, `--output-dir`, `--stem`,
`--device cuda`). The trainer writes `teacher_mask_separator_progress.jsonl`
with start, CUDA memory guard, explicit CUDA sync phase markers, pre-compute
epoch/permutation/batch markers, batch compute subphase markers, batch
heartbeat, per-epoch, apply, artifact, and finish events. It also records
`training.sample_rate_alignment`; if a teacher target has a different sample
rate than the source mix, the target is resampled to the source rate via
`scipy.signal.resample_poly` and the manifest explicitly marks
`teacher_target_resampled=true`.

`tools/run_generic_audio_finetune_jobs.py` is the reproducible execution wrapper
for ready fine-tune jobs. It reads the job plan, runs only
`ready_for_gpu_training` jobs, validates `device_used=1.0`, validates the
per-job progress trail, and writes `generic_audio_finetune_run_manifest.json`.
Use `--skip-existing-complete` to validate already-trained outputs without
rerunning GPU training.

`tools/validate_generic_audio_finetune_holdout.py` is the first objective
generalization gate for these training artifacts. It reads
`generic_audio_finetune_run_rows.jsonl`, applies each selected checkpoint to
cross-track examples from `merged_audio_training_examples.jsonl`, and compares
the checkpoint output against the target stem and the raw source baseline. The
gate requires both an absolute SDR threshold and an improvement over source.
It writes `generic_audio_finetune_holdout_manifest.json`,
`generic_audio_finetune_holdout_rows.jsonl`, and a structured progress log with
`gpu_inference_holdout` / `cpu_diagnostic_holdout` mode. The holdout artifacts
remain blocked from listen/release delivery:
`delivery_allowed=false`, `presentation_status=holdout_training_artifact_not_for_listen_pack`,
and `release_ready=false`.

Checkpoint application is observable inside the holdout progress log. The
validator records checkpoint application start, audio load, checkpoint load,
feature extraction, source-feature-memory search, chunk progress, blocked
guards, and finish/failure events. Source-feature-memory KNN application is
fail-closed by `--max-source-feature-memory-distance-ops` to prevent a
cross-track holdout from running an unbounded `torch.cdist` search without a
phase trail. If the guard trips, the holdout row fails with
`source_feature_memory_cost_guard`; no candidate audio is published, no fallback
is used, and `release_ready=false` remains unchanged.

The shared teacher-mask adapter also supports bounded source-feature memory for
matrix experiments:

- `mask_output_mode=magnitude_residual` uses an identity-centered magnitude
  mask head. The output starts at no-op (`mask=1.0`) and is bounded by
  `mask_ceiling` around identity, so a config can test small corrective deltas
  without the sigmoid attenuation bias of the default `magnitude` mode. It is
  still a training/evaluation candidate only; holdout improvement gates remain
  unchanged.
- `source_preservation_weight` adds an explicit loss term against moving the
  learned mask away from source identity. It is intended for candidate-
  conditioned stems that are already strong baselines and must not be damaged by
  a shared adapter. The default is `0.0`; enabling it does not lower any holdout
  gate and does not publish fallback/no-op audio.
- `source_feature_memory_projection_dim` stores a deterministic seed-derived
  projection matrix in the checkpoint and applies KNN in that projected space.
- `source_feature_memory_max_frames` stores only a deterministic frame subset
  for source-feature memory.

These fields are part of the adapter fingerprint and manifest config. They do
not bypass holdout gates; they only make a memory route operationally bounded
enough to evaluate.

The first vocal holdout run on the recovered candidate-map corpus is:

`/app/lab_artifacts/generic_finetuning_playbook_20260613/vocal_holdout_eval_candidate_map_recovered_v2_device_traced/generic_audio_finetune_holdout_manifest.json`

It failed promotion correctly: 2/2 evaluated per-example vocal checkpoints
improved over source but stayed below the absolute SDR threshold on cross-track
examples. This means the current per-example teacher-mask checkpoints are not a
production vocal separator. The next training route must learn one shared
multi-example vocal adapter before any promotion or listen-pack claim.

`tools/train_generic_audio_teacher_mask_adapter.py` trains that shared
multi-example adapter. It consumes `merged_audio_training_examples.jsonl`,
filters by stem, excludes explicit holdout track IDs, optionally caps examples
per track and STFT frames per pair for balanced training, and writes a single
checkpoint plus compatible `generic_audio_finetune_run_rows.jsonl` for the
holdout gate. The route can enable KNN source-feature memory with
`--source-feature-memory-neighbors`; complex source memory remains disabled in
the shared trainer because it is reconstructive per-source state.

The shared adapter can also train `--mask-output-mode complex`. In that mode the
network predicts a bounded complex ratio mask and the training loss includes
complex STFT reconstruction error instead of only magnitude-mask error. This is
a training/evaluation route only: it still writes `delivery_allowed=false`,
requires the same locked holdout gate, and cannot mark `release_ready`.

Complex mask output can be combined with explicit source-feature memory by
setting `--source-feature-memory-neighbors`. In that route the checkpoint stores
bounded frame-level complex masks under
`source_feature_memory_complex_frame_mask_real/imag`, optionally projected by
`source_feature_memory_projection_dim` and capped by
`source_feature_memory_max_frames`. Application records
`source_feature_memory_complex_frame_mask_available/used` and remains guarded by
`--max-source-feature-memory-distance-ops`. This is an explicit matrix
candidate, not a silent fallback or release path.

`tools/run_generic_audio_finetune_matrix.py` is the reproducible config-matrix
wrapper for that shared adapter. It reads a JSON matrix, trains each config into
its own subdirectory, immediately runs the cross-track holdout gate for that
config, and writes `generic_audio_finetune_matrix_manifest.json`,
`generic_audio_finetune_matrix_rows.jsonl`, and a structured progress log. Each
nested training run emits dataset feature-stack, chunk-release, model-init
subphase (network construction, move-to-device, optimizer), training-index,
`dataset_train_loop_start`, pre-compute epoch/permutation/batch start rows,
explicit pre-loop CUDA sync rows, batch feature-gather/forward/loss/backward/
optimizer subphase rows, post-step batch heartbeat rows, and `dataset_train_epoch`
for every epoch, not just a percentage stride, so long GPU runs keep a
reproducible phase trail.
The matrix wrapper also emits `cuda_phase_cleanup_start` and
`cuda_phase_cleanup_finish` between training, holdout, and the next config. The
cleanup synchronizes CUDA work, releases cached allocator blocks, runs IPC
cleanup when available, records free/total CUDA memory, and stays fail-closed:
if cleanup fails during a required GPU matrix run, the current config fails
instead of falling back to CPU or hiding the allocator state from the next
config.
Index-order permutation and batch index select are CPU-resident orchestration
metadata. Large precomputed feature, STFT, mask, and weight tensors are also
kept CPU-resident. CUDA dataset runs store the precomputed magnitude/features in
compact `float16` and convert the current batch back to `float32` on the
requested training device. Magnitude-mode training does not store full STFT or
complex-mask tensors; those are only kept for explicit complex-mask training.
Dataset feature normalization is chunked: the trainer computes feature
mean/std across CPU chunks, normalizes each chunk separately, and only then
stacks the compact normalized features. Progress rows include
`dataset_feature_stats_*`, `dataset_feature_normalize_*`, and
`normalization_strategy=chunked_cpu_low_peak` so large matrix runs do not hide
multi-GB feature-stack phases.
Each batch emits `dataset_train_batch_to_device_start` /
`dataset_train_batch_to_device_finish` while moving only the current batch to
the requested training device. Forward, loss, backward, and optimizer steps
still run on that requested device. This prevents hidden multi-minute CUDA sync
work before the first batch and reduces CPU/GPU allocation pressure while
keeping long runs observable. The shared adapter
initializes and moves the model to the requested device before training, then
releases per-pair chunk references after stacking to reduce allocation pressure.
Magnitude-mode networks scale their sigmoid output by `mask_ceiling`, matching
the existing `oracle_mask = target_mag / source_mag` training target when
`mask_ceiling > 1.0`; the application gate still clamps masks to the same
configured ceiling.
If CUDA free memory is below the configured `min_cuda_free_memory_mb` guard,
training fails closed before allocation instead of falling back to CPU. The
matrix chooses a best config only from configs whose holdout manifest passed; if
none pass, `matrix_validation_status=failed` and `best_config_id=""`. Matrix
also recomputes the row-level holdout contract from `holdout_rows_jsonl` instead
of trusting a single aggregate status field. The matrix runner passes the same
source-feature-memory application cost guard through to each nested holdout so a
single high-KNN config cannot become an unobservable long-running run. Matrix
artifacts are still
training/evaluation evidence only:
`delivery_allowed=false`,
`presentation_status=training_matrix_artifact_not_for_listen_pack`, and
`release_ready=false`.

`tools/build_candidate_conditioned_audio_training_view.py` creates a training
view where a completed registered candidate stem becomes the source and the
verified corpus stem remains the target. The candidate source is explicitly
marked `source_candidate_is_ground_truth=false`, missing candidates are blocked
instead of replaced with the raw mix, and the route is metadata-only.

`tools/build_multi_candidate_conditioned_audio_training_view.py` extends that
contract to multiple registered source families. Each `--candidate-root
LABEL=PATH` emits at most one source row per verified example, with ids such as
`candidate_conditioned__htd6__<example_id>`. This lets a matrix train on several
registered model families without pretending any candidate is a target stem.
The output manifest records `family_counts`, keeps
`delivery_allowed=false`, and forces `release_ready=false`.
When the source rows are accepted no-reference teacher targets rather than
verified references, use `--output-pseudo-teacher-rows`. That mode writes
`generic_finetuning_pseudo_teacher_audio_training_example.v1` rows with
`pseudo_teacher=true`, `pseudo_teacher_targets_are_ground_truth=false`,
`human_verified=false`, and `reference_verified=false`, so the rows can be used
only through `--pseudo-teacher-examples-jsonl` and cannot become holdout
references.

`tools/build_pseudo_teacher_audio_training_examples.py` is the train-safe route
for using accepted no-reference runtime stems as weak training targets without
calling them ground truth. It reads run roots containing:

- `run_manifest.json`
- `selected_candidates.jsonl`
- `music_intelligence_input_manifest.json`

and writes:

- `pseudo_teacher_audio_training_examples.jsonl`
- `pseudo_teacher_audio_training_blocked_candidates.jsonl`
- `pseudo_teacher_audio_training_manifest.json`
- `pseudo_teacher_audio_training_progress.jsonl`

Rows use schema
`generic_finetuning_pseudo_teacher_audio_training_example.v1` and must carry
`pseudo_teacher=true`, `human_verified=false`, `reference_verified=false`,
`permission_status=pseudo_teacher_runtime_artifact`,
`pseudo_teacher_targets_are_ground_truth=false`, `delivery_allowed=false`, and
`release_ready=false`. The builder only includes selected stems that also appear
as `quality_status=accepted`, `analysis_allowed=true` Music Intelligence stems
and whose source/target audio files exist. Publishable or release-ready runtime
candidates are blocked instead of reused.

`tools/train_generic_audio_teacher_mask_adapter.py` accepts pseudo rows only
through the explicit `--pseudo-teacher-examples-jsonl` argument. The verified
training JSONL remains separate, and the adapter manifest records
`pseudo_teacher_training_example_count`, `pseudo_teacher_track_ids`, and a
contract stating `dirty_stems_never_ground_truth=true`.
For candidate-conditioned specialists, both the trainer and the matrix support
`source_candidate_family_filter`. It filters training rows by
`source_candidate_family`, carries the filter into the holdout gate, and records
the selected source families in the adapter manifest. This prevents a specialist
config from training on one registered model family while being evaluated on a
different family.

`tools/run_generic_audio_finetune_matrix.py` forwards
`--pseudo-teacher-examples-jsonl` only to the training stage. Holdout validation
continues to read the verified `--audio-training-examples-jsonl`, so pseudo
teacher rows cannot become the reference target for locked eval. This route is
not a fallback and cannot lower thresholds, edit runtime gates, promote a model,
or mark `release_ready`.

`tools/run_candidate_family_blend_matrix.py` is a trainable no-reference adapter
for candidate-conditioned views with two or more source families for the same
verified example, such as `scnet` and `melband`. It learns convex blend weights
only from non-holdout verified groups, writes a JSON adapter checkpoint, renders
bounded holdout evaluation audio, and emits the same matrix/row contract shape
used by `build_train_ready_audio_model_decision.py`. Holdout improvement is
measured against `baseline_policy=best_family`, meaning the blend must beat the
best individual candidate family for that holdout group; it cannot pass by
choosing a weaker source as baseline. The route is CPU/DSP by design because it
does no neural inference or separator training. Its progress log still records
`gpu_inference_skipped=true`, `gpu_training_skipped=true`, and explicit skip
reasons. It never uses pseudo-teacher rows, never relaxes thresholds, never
edits runtime gates or model registries, and always writes `release_ready=false`.
Use `tools/build_candidate_family_blend_locked_split.py` before the train-ready
decision when the split is assigned downstream rather than encoded in the input
JSONL. The split builder writes a locked leakage manifest with train and holdout
track ids, requires complete source-family groups, rejects dirty/release-ready
input rows, and keeps `release_ready=false`.

The first shared vocal adapter waves excluded `nothing-can-wait` from training
and evaluated it as holdout:

- `shared_vocal_adapter_exclude_ncw_v1`: failed, candidate SDR `-0.060670`,
  improvement over source `+7.713702 dB`.
- `shared_vocal_adapter_exclude_ncw_v2_active_balanced`: failed, candidate SDR
  `-0.840131`, improvement `+6.934241 dB`.
- `shared_vocal_adapter_exclude_ncw_v3_frame_balanced`: failed, candidate SDR
  `0.034064`, improvement `+7.808436 dB`.
- `shared_vocal_adapter_exclude_ncw_v4b_knn_memory`: failed, KNN memory used on
  CUDA, candidate SDR `-0.032964`, improvement `+7.741407 dB`.
- `shared_vocal_adapter_matrix_v1`: failed, Docker/CUDA matrix smoke with two
  configs. `frame_balanced_smoke` reached candidate SDR `0.157136`,
  improvement `+7.931508 dB`; `knn_memory_smoke` reached candidate SDR
  `-0.609945`, improvement `+7.164426 dB`. Both stayed below the absolute
  `min_sdr_db=3.0` holdout gate, so `best_config_id=""`.

None of these artifacts are release-ready. The evidence says the current
verified vocal corpus is still too narrow or too mismatched for a production
NCW-domain vocal separator; the next data step is to add more verified vocal
targets from distinct real tracks or train/evaluate a stronger separator-family
adapter rather than lowering holdout thresholds.

## Audio Training Recovery Queue

`tools/build_generic_audio_training_recovery_queue.py` scans the corpus feedback
inputs plus `ground_truth_manifest.json` files and writes source/target audio
rows with explicit blockers:

- missing or non-trainable permission status;
- missing human/reference verification;
- missing accepted machine-gate provenance for non-reference rows;
- missing Docker-visible source or target audio path;
- negative or non-positive listener rating.

The queue is metadata-only. Rows with no blockers can be imported into the next
corpus version only when they are either reference-verified or carry accepted
`machine_stem_acceptance_gate.v1` provenance; blocked rows tell exactly what
must be fixed before training.
It also writes `generic_audio_training_recovery_decision_template.jsonl`, a
ready-to-fill review sheet for blocked rows.
Rows are tagged with `source_domain` and `recovery_priority`; synthetic fixture
ground-truth rows are separated from user-supplied music so they cannot be
mistaken for production fine-tune data.

If a human-feedback row points at an accepted listen-preview manifest via
`manifest_path` but does not repeat the source/target audio paths, the recovery
queue resolves the manifest and recovers Docker-visible `mix_path` plus the
stem-specific selected target path (`selected_vocal_path`, `selected_minus_path`,
etc.). Path recovery applies only to manifests with `delivery_status=accepted`,
`delivery_allowed=true`, and `presentation_status=accepted_for_listen_pack`,
but training readiness still requires accepted `machine_stem_acceptance_gate.v1`
provenance. Plain listen-preview/source-ready manifests are kept as
non-training recovery evidence with
`missing_accepted_machine_gate_provenance`.

If a human-feedback row points at a listen-pack artifact and the listen-pack
directory contains `candidate_map.json`, the recovery queue can recover the
source mix from the mapped candidate's sibling `stem_manifest.json`. When
`source_audio_sha256` is present in the feedback row and in the stem manifest,
they must match. This is a general provenance recovery rule; it does not
hardcode track names, timestamps, or candidate labels.

`tools/ingest_generic_audio_training_recovery_decisions.py` applies explicit
review decisions to that queue. A blocked row becomes an audio-training example
only when the decision says `approve_for_training`, provides a trainable
`permission_status`, and no source/target/verification/provenance blocker
remains. The ingest layer repeats the accepted-machine-gate/reference check so
older or manually created `ready_for_audio_training=true` rows cannot bypass the
queue builder.

`tools/build_generic_audio_training_approval_pack.py` is the safe review
handoff for expanding the training set. It selects only P0/P1 permission-only
recovery rows with existing source/target audio and human/reference
verification after accepted machine-gate/reference provenance has already been
cleared, writes `generic_audio_training_approval_candidates.jsonl`, and
also writes `generic_audio_training_draft_decisions.jsonl` with
`decision=needs_more_evidence`. It never auto-approves permission and excludes
synthetic fixture rows. Rows with `missing_accepted_machine_gate_provenance` are
not permission-only candidates and cannot enter the approval pack.
Both `missing_permission_status` and `permission_not_trainable` are treated as
permission-only review blockers when they are the only blocker. This allows
rows with `permission_status=requires_review` to enter the approval pack after
their source/target audio paths have been recovered, while still requiring an
explicit `approve_for_training` decision and a trainable permission status
before ingestion.

`tools/build_project_owned_audio_permission_decisions.py` is the non-CSV
attestation route for local project-owned ground-truth bundles. It only writes
`approve_for_training` decisions for recovery rows that are
`source_domain=user_supplied_flstudio_music`,
`source_type=ground_truth_manifest`, `reference_verified=true`, have existing
source and target audio, and are blocked only by a permission-status blocker.
Human feedback audio pairs and synthetic fixtures are not auto-approved. The
output is still metadata-only and must be passed through
`tools/ingest_generic_audio_training_recovery_decisions.py` before merge and
validation.

## Merged Audio Training Dataset

`tools/merge_generic_audio_training_examples.py` bridges recovery decisions
back into the trainer path. It merges one or more base
`audio_training_examples.jsonl` files with one or more
`approved_audio_training_examples.jsonl` files, recomputes a missing target
SHA-256 from the existing target audio, deduplicates identical source/target
pairs, and rejects rows that are not trainable or whose audio paths are missing.

The output `merged_audio_training_examples.jsonl` is the single input that
`tools/plan_generic_audio_finetune_jobs.py` should consume. The merge stage is
metadata-only: it does not start training, does not edit model registries, and
does not create listen or release artifacts.

Before a merged corpus enters GPU training, validate it with
`tools/validate_verified_audio_training_corpus.py`. This gate requires every
training row to have existing source/target audio, trainable permission, and
either accepted-machine-gated human verification or reference verification with
`source_gate.trainable_ground_truth=true` or an accepted exact-reference
waveform audit recorded as `source_gate.exact_reference_waveform_audit_accepted=true`.
Reference metadata such as `reference_contract.verified=true` is not enough by
itself. It also blocks track leakage across split labels and never marks any
artifact release-ready.

After a shared adapter matrix finishes, use
`tools/build_train_ready_audio_model_decision.py` to decide whether the TRAIN
result is actually verified. The decision is strict: the best matrix config
must pass safety and holdout checks and its success rate must be greater than
`0.90`. The decision artifact is still not a promotion and keeps
`release_ready=false`.

## Docker Commands

```bash
python /app/tools/build_generic_finetuning_corpus_from_lab_artifacts.py \
  --artifact-root /app/lab_artifacts \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v1 \
  --human-feedback-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/human_feedback_seed_user_reported_v1.jsonl

python /app/tools/train_generic_finetuning_controller_model.py \
  --controller-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v1/controller_decision_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_v1 \
  --min-examples-per-route 2

python /app/tools/build_qwen_controller_sft_dataset.py \
  --controller-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v1/controller_decision_examples.jsonl \
  --eval-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_eval_v1/controller_decision_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_sft_v1

python /app/tools/cache_qwen_controller_model.py \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_model_cache_v1 \
  --allow-network \
  --max-workers 4

python /app/tools/qwen_controller_qlora_preflight.py \
  --sft-dataset-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_sft_v1/manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_qlora_preflight_v1 \
  --requested-device cuda:0 \
  --min-cuda-free-memory-mb 10000

python /app/tools/qwen_controller_qlora_train.py \
  --sft-dataset-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_sft_v1/manifest.json \
  --preflight-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_qlora_preflight_v1/manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_qlora_train_v1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 256 \
  --lora-r 8 \
  --lora-alpha 16 \
  --max-steps 1

python /app/tools/qwen_controller_qlora_eval.py \
  --sft-dataset-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_sft_v1/manifest.json \
  --training-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_qlora_train_v1/manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/qwen_qlora_eval_v1 \
  --min-eval-examples 20 \
  --min-adapter-accuracy 0.7 \
  --min-accuracy-improvement 0.0

python /app/tools/validate_generic_audio_finetune_holdout.py \
  --run-rows-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/vocal_gpu_training_run_candidate_map_recovered_v3_runner_validated/generic_audio_finetune_run_rows.jsonl \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_examples_candidate_map_recovered_v4/merged_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/vocal_holdout_eval_candidate_map_recovered_v2_device_traced \
  --scope vocal_teacher_mask_adapter \
  --stem vocal \
  --device cuda \
  --max-checkpoints 2 \
  --max-holdouts-per-checkpoint 1 \
  --min-sdr-db 3.0 \
  --min-improvement-db 0.5

python /app/tools/train_generic_audio_teacher_mask_adapter.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_examples_candidate_map_recovered_v4/merged_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/shared_vocal_adapter_exclude_ncw_v4b_knn_memory \
  --stem vocal \
  --scope vocal_shared_teacher_mask_adapter \
  --exclude-holdout-track-id nothing-can-wait \
  --min-unique-tracks 2 \
  --max-examples-per-track 1 \
  --max-frames-per-pair 6000 \
  --n-fft 1024 \
  --hop-length 256 \
  --hidden-size 384 \
  --epochs 120 \
  --batch-size 1024 \
  --learning-rate 0.0025 \
  --mask-ceiling 1.0 \
  --activity-weight 22.0 \
  --leakage-suppression-weight 0.5 \
  --magnitude-loss-weight 0.1 \
  --active-frame-oversample 2 \
  --context-frame-radius 2 \
  --source-feature-memory-neighbors 3 \
  --source-feature-memory-chunk-size 256 \
  --seed 20260613 \
  --device cuda

python /app/tools/run_generic_audio_finetune_matrix.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_examples_candidate_map_recovered_v4/merged_audio_training_examples.jsonl \
  --config-matrix-json /app/lab_artifacts/generic_finetuning_playbook_20260613/shared_vocal_adapter_matrix_v1/config_matrix.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/shared_vocal_adapter_matrix_v1 \
  --stem vocal \
  --scope vocal_shared_teacher_mask_adapter_matrix \
  --exclude-holdout-track-id nothing-can-wait \
  --min-unique-tracks 2 \
  --device cuda \
  --min-sdr-db 3.0 \
  --min-improvement-db 0.5 \
  --max-holdouts-per-checkpoint 1

python /app/tools/run_candidate_family_blend_matrix.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/multi_candidate_conditioned_vocal_training_view_v13_strong_scnet_melband_all_verified_no_melody/candidate_conditioned_audio_training_examples.jsonl \
  --config-matrix-json /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_candidate_family_blend_matrix_v32_configs.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_candidate_family_blend_matrix_v32_001 \
  --stem vocal \
  --scope vocal_candidate_family_blend_matrix_gt90_cycle_001_v32 \
  --exclude-holdout-track-id nothing-can-wait \
  --min-sdr-db 3.0 \
  --min-improvement-db 0.5 \
  --max-holdouts-per-config 2

python /app/tools/build_candidate_family_blend_locked_split.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/multi_candidate_conditioned_vocal_training_view_v13_strong_scnet_melband_all_verified_no_melody/candidate_conditioned_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/candidate_family_blend_locked_split_v33_001 \
  --stem vocal \
  --family scnet \
  --family melband \
  --holdout-track-id nothing-can-wait \
  --holdout-track-id 3404a02f61d017a1ea0a3776f7db6f3abf7cd1969133e9bc87e7f879f6f71273

python /app/tools/eval_generic_finetuning_controller_model.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_v1/generic_finetuning_controller_model.json \
  --eval-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_eval_v1/controller_decision_examples.jsonl \
  --training-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_v1/generic_finetuning_controller_training_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_external_eval_v1 \
  --min-eval-examples 20

python /app/tools/build_generic_controller_external_eval_from_hard_cases.py \
  --hard-eval-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/hard_eval_cases.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_external_eval_set_v1

python /app/tools/eval_generic_finetuning_controller_model.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_v1/generic_finetuning_controller_model.json \
  --eval-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_external_eval_set_v1/generic_controller_external_eval_examples.jsonl \
  --training-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_v1/generic_finetuning_controller_training_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_external_eval_from_hard_cases_v1 \
  --min-eval-examples 20

python /app/tools/build_generic_controller_hard_case_train_eval_split.py \
  --base-controller-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/controller_decision_examples.jsonl \
  --hard-eval-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/hard_eval_cases.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_hard_case_split_v1

python /app/tools/train_generic_finetuning_controller_model.py \
  --controller-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_hard_case_split_v1/augmented_controller_train_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1 \
  --min-examples-per-route 2

python /app/tools/eval_generic_finetuning_controller_model.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_model.json \
  --eval-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_hard_case_split_v1/hard_case_holdout_external_eval_examples.jsonl \
  --training-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_training_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_external_eval_from_hard_case_split_v1 \
  --min-eval-examples 20

python /app/tools/run_generic_finetuning_controller_runtime_smoke_replay.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_model.json \
  --replay-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_hard_case_split_v1/hard_case_holdout_external_eval_examples.jsonl \
  --hard-case-route-policy-json /app/lab_artifacts/no_reference_minus_route_expansion/hard_case_route_policy.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_runtime_smoke_from_hard_case_split_v1 \
  --min-replay-examples 20

python /app/tools/stage_generic_finetuning_controller_with_no_reference_selector.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_model.json \
  --judge-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/example_no_reference_score_matrix/no_reference_minus_candidate_score_matrix.jsonl \
  --hard-case-route-policy-json /app/lab_artifacts/no_reference_minus_route_expansion/hard_case_route_policy.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_integration_v1 \
  --score-mode minus \
  --min-selector-usable 1

python /app/tools/run_generic_controller_selector_shadow_batch.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_model.json \
  --judge-root /app/lab_artifacts/generic_finetuning_playbook_20260613 \
  --judge-glob "**/no_reference_*_windowed_judge_results.jsonl" \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_shadow_batch_v1 \
  --score-mode minus \
  --min-selector-usable-per-input 1 \
  --min-batch-checked-candidates 20

python /app/tools/plan_generic_controller_selector_route_gaps.py \
  --shadow-batch-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_shadow_batch_v1/generic_controller_selector_shadow_batch_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_route_gap_plan_v1 \
  --min-repeated-blocked 2

python /app/tools/plan_generic_controller_selector_candidate_wave.py \
  --route-gap-plan-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_route_gap_plan_v1/generic_controller_selector_route_gap_plan_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_plan_v1 \
  --device cuda \
  --resurrection-device cuda:0 \
  --max-sources-per-route 1

python /app/tools/run_generic_controller_selector_candidate_wave_jobs.py \
  --candidate-wave-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_plan_v1/plan.json \
  --output-root /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_execution_v1_dry_run \
  --max-jobs 3 \
  --dry-run

python /app/tools/run_generic_controller_selector_candidate_wave_jobs.py \
  --candidate-wave-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_plan_v1/plan.json \
  --output-root /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_execution_v1 \
  --max-jobs 1 \
  --fail-fast

python /app/tools/plan_generic_audio_finetune_jobs.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_plan_v1 \
  --min-unique-tracks 3

python /app/tools/build_generic_audio_training_recovery_queue.py \
  --artifact-root /app/lab_artifacts \
  --corpus-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/generic_finetuning_corpus_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_recovery_v1

python /app/tools/build_generic_audio_training_approval_pack.py \
  --recovery-queue-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_recovery_v1/generic_audio_training_recovery_queue.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_approval_pack_v1

python /app/tools/ingest_generic_audio_training_recovery_decisions.py \
  --recovery-queue-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_recovery_v1/generic_audio_training_recovery_queue.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_recovery_decision_v1

python /app/tools/merge_generic_audio_training_examples.py \
  --base-audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/corpus_v3/audio_training_examples.jsonl \
  --approved-audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_training_recovery_decision_v1/approved_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_v1

python /app/tools/plan_generic_audio_finetune_jobs.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/merged_audio_training_v1/merged_audio_training_examples.jsonl \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_plan_from_merged_v1 \
  --min-unique-tracks 3

python /app/tools/run_generic_audio_finetune_jobs.py \
  --job-plan-json /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_plan_from_merged_v1/generic_audio_finetune_job_plan.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/audio_finetune_run_v1 \
  --scope vocal_teacher_mask_adapter
```

These commands must run inside Docker. GPU inference is skipped for corpus,
recovery, decision ingest, and merge stages because they curate metadata rather
than audio tensors. Ready training jobs produced by the planner use
`--device cuda` and must fail closed if CUDA is unavailable.
