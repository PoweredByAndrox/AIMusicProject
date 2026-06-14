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
targets. Audio examples become trainable only when they are human-verified or
reference-verified and have a cleared `permission_status` such as `owned` or
`licensed`.

Verified `ground_truth_manifest.json` bundles are imported as audio-training
examples when `ground_truth_contract.verified=true` and the rights/permission
status is trainable. `user_owned` is normalized to `project_owned`. If the
ground-truth contract says the source mix cannot be claimed as a one-to-one sum
of imported stems, the example remains trainable but carries
`source_mix_1_to_1_claim_allowed=false` for downstream evaluators.

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
with start, audio-load, epoch, apply, artifact, and finish events. It also
records `training.sample_rate_alignment`; if a teacher target has a different
sample rate than the source mix, the target is resampled to the source rate via
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

`tools/run_generic_audio_finetune_matrix.py` is the reproducible config-matrix
wrapper for that shared adapter. It reads a JSON matrix, trains each config into
its own subdirectory, immediately runs the cross-track holdout gate for that
config, and writes `generic_audio_finetune_matrix_manifest.json`,
`generic_audio_finetune_matrix_rows.jsonl`, and a structured progress log. The
matrix chooses a best config only from configs whose holdout manifest passed; if
none pass, `matrix_validation_status=failed` and `best_config_id=""`. Matrix
also recomputes the row-level holdout contract from `holdout_rows_jsonl` instead
of trusting a single aggregate status field. Matrix artifacts are still
training/evaluation evidence only:
`delivery_allowed=false`,
`presentation_status=training_matrix_artifact_not_for_listen_pack`, and
`release_ready=false`.

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
- missing Docker-visible source or target audio path;
- negative or non-positive listener rating.

The queue is metadata-only. Rows with no blockers can be imported into the next
corpus version; blocked rows tell exactly what must be fixed before training.
It also writes `generic_audio_training_recovery_decision_template.jsonl`, a
ready-to-fill review sheet for blocked rows.
Rows are tagged with `source_domain` and `recovery_priority`; synthetic fixture
ground-truth rows are separated from user-supplied music so they cannot be
mistaken for production fine-tune data.

If a human-feedback row points at an accepted listen-preview manifest via
`manifest_path` but does not repeat the source/target audio paths, the recovery
queue resolves the manifest and recovers Docker-visible `mix_path` plus the
stem-specific selected target path (`selected_vocal_path`, `selected_minus_path`,
etc.). This applies only to manifests with `delivery_status=accepted`,
`delivery_allowed=true`, and `presentation_status=accepted_for_listen_pack`.
It does not make the row trainable by itself; permission and verification
blockers still apply.

If a human-feedback row points at a listen-pack artifact and the listen-pack
directory contains `candidate_map.json`, the recovery queue can recover the
source mix from the mapped candidate's sibling `stem_manifest.json`. When
`source_audio_sha256` is present in the feedback row and in the stem manifest,
they must match. This is a general provenance recovery rule; it does not
hardcode track names, timestamps, or candidate labels.

`tools/ingest_generic_audio_training_recovery_decisions.py` applies explicit
review decisions to that queue. A blocked row becomes an audio-training example
only when the decision says `approve_for_training`, provides a trainable
`permission_status`, and no source/target/verification blocker remains.
Rows that are already ready can be imported without a decisions file.

`tools/build_generic_audio_training_approval_pack.py` is the safe review
handoff for expanding the training set. It selects only P0/P1 permission-only
recovery rows with existing source/target audio and human/reference
verification, writes `generic_audio_training_approval_candidates.jsonl`, and
also writes `generic_audio_training_draft_decisions.jsonl` with
`decision=needs_more_evidence`. It never auto-approves permission and excludes
synthetic fixture rows.
Both `missing_permission_status` and `permission_not_trainable` are treated as
permission-only review blockers when they are the only blocker. This allows
rows with `permission_status=requires_review` to enter the approval pack after
their source/target audio paths have been recovered, while still requiring an
explicit `approve_for_training` decision and a trainable permission status
before ingestion.

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
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_runtime_smoke_from_hard_case_split_v1 \
  --min-replay-examples 20

python /app/tools/stage_generic_finetuning_controller_with_no_reference_selector.py \
  --controller-model-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_model_from_hard_case_split_v1/generic_finetuning_controller_model.json \
  --judge-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260613/example_no_reference_score_matrix/no_reference_minus_candidate_score_matrix.jsonl \
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
  --candidate-wave-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_plan_v1/generic_controller_selector_candidate_wave_manifest.json \
  --output-root /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_execution_v1_dry_run \
  --max-jobs 3 \
  --dry-run

python /app/tools/run_generic_controller_selector_candidate_wave_jobs.py \
  --candidate-wave-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260613/controller_selector_candidate_wave_plan_v1/generic_controller_selector_candidate_wave_manifest.json \
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
