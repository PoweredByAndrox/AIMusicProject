# ARIS Train >90 Cycle 15 Plan

Date: 2026-06-16
Scope: vocal teacher-mask adapter train/eval loop
Status: prepared, waiting for CUDA memory availability

## Source-of-Truth Checkpoint

Cycle 14 / v20 completed; it did not hang.

Evidence:

- Matrix manifest:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v20_detail_repair_only_holdout_ncw_compact_cpu_resident/generic_audio_finetune_matrix_manifest.json`
- Matrix rows:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v20_detail_repair_only_holdout_ncw_compact_cpu_resident/generic_audio_finetune_matrix_rows.jsonl`
- Train progress:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v20_detail_repair_only_holdout_ncw_compact_cpu_resident/configs/001_strong_family_magnitude_detail_repair/train/generic_audio_teacher_mask_adapter_progress.jsonl`
- Holdout manifest:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v20_detail_repair_only_holdout_ncw_compact_cpu_resident/configs/001_strong_family_magnitude_detail_repair/holdout/generic_audio_finetune_holdout_manifest.json`
- Train-ready decision:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/train_ready_decision_vocal_matrix_v20_detail_repair_only_ncw_compact_cpu_resident/train_ready_audio_model_decision_manifest.json`

Confirmed facts:

- Training reached epoch 200/200 on CUDA.
- Last train progress event was `finish`.
- `gpu_training_required=true`.
- `gpu_training_skipped=false`.
- Holdout ran on CUDA.
- Holdout validation failed: 0/2 holdout rows passed.
- Holdout improvements on `nothing-can-wait` were negative:
  - `candidate_conditioned__scnet__9c29000c5a66578f`: `-0.474735 dB`
  - `candidate_conditioned__melband__9c29000c5a66578f`: `-0.079139 dB`
- Train-ready decision:
  - `decision=train_result_not_ready`
  - `success_rate=0.0`
  - `train_ready=false`
  - `release_ready=false`

## Failure Interpretation

The v20 failure is not an observability hang. It is a quality failure.

The current verified training view is too narrow for a general vocal adapter:

- `candidate_conditioned_audio_training_examples.jsonl`: 16 rows, 3 track ids.
- With `nothing-can-wait` excluded for holdout, training used 14 rows on only 2 track ids.
- The verified rows are dominated by the Square Oxygen family.

Do not repeat v20 unchanged.

## New Safe Data Signal

Built explicit pseudo-teacher dataset from accepted no-reference runtime stems:

`lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/pseudo_teacher_vocal_runtime_accepted_v21/pseudo_teacher_audio_training_manifest.json`

Confirmed:

- `status=passed`
- `pseudo_teacher_example_count=4`
- `blocked_candidate_count=0`
- `unique_track_count=2`
- track ids:
  - `Chris_Bulman___Sarju_Jagai_-_No_Escape`
  - `Totally_Enormous_Extinct_Dinosaurs_-_Household_Goods`
- `pseudo_teacher_targets_are_ground_truth=false`
- `human_verified=false`
- `reference_verified=false`
- `delivery_allowed=false`
- `release_ready=false`
- `dirty_stems_never_ground_truth=true`

Contract:

- Pseudo-teacher rows are allowed only as weak train targets.
- They must never be used as holdout references.
- Holdout remains verified-only.
- No threshold relaxation.
- No CPU fallback.
- No release-ready.

Verified corpus pre-v21 validation:

- Manifest:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/verified_vocal_training_corpus_v13_pre_v21/verified_audio_training_corpus_manifest.json`
- Status: `passed`
- `verified_trainable_examples=16`
- `unique_track_count=3`
- row breakdown:
  - `Square a Saw - Oxygen (Tevlo Vocals Resynthesized)`: 12 rows
  - `nothing-can-wait`: 2 rows
  - `3404a02f61d017a1ea0a3776f7db6f3abf7cd1969133e9bc87e7f879f6f71273`: 2 rows
- `release_ready=false`
- `delivery_allowed=false`
- `training_started=false`

Interpretation:

- The verified corpus is valid but narrow and Square-Oxygen dominated.
- v21 keeps the verified holdout strict while adding explicit weak
  pseudo-teacher rows only to training.

## GPU Blocker

CUDA training is currently blocked by another container:

- Container: `trading-view-llm-lora-trainer-run-045d1c338b75`
- GPU state observed through Docker `nvidia-smi`:
  - RTX 4080 Laptop GPU
  - about 10.3 GiB VRAM used
  - about 1.7 GiB VRAM free
  - GPU util about 97%

The v21 train route must not start until free VRAM is at least 4096 MiB.

Fresh continuation check at 2026-06-16T15:13-15:14+04:00:

- Holder still running:
  `4719fd2a3c55 trading-view-llm-lora-trainer-run-045d1c338b75`
- GPU checks:
  - `memory.used=10189 MiB`, `memory.free=1809 MiB`, `utilization.gpu=100%`
  - `memory.used=10189 MiB`, `memory.free=1809 MiB`, `utilization.gpu=35%`
  - `memory.used=10189 MiB`, `memory.free=1809 MiB`, `utilization.gpu=100%`
- Read-only log inspection showed the holder is still mid-training, not a
  finished/stale AIMusic job.

Decision:

- Do not launch v21 while this blocker holds.
- Do not use CPU fallback.
- Do not stop the unrelated container without explicit user approval.

Automation prepared:

- Watcher script:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/run_v21_when_gpu_free.ps1`
- Active watcher PID at launch check: `30896`
- Monitor log:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v21_launch_monitor.log`
- Matrix stdout log:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v21_matrix_stdout.log`
- Decision stdout log:
  `lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/v21_train_ready_decision_stdout.log`
- Last observed monitor entries still show `free_mb=1839`, so v21 had not
  launched yet.

## Cycle 15 Experiment

Hypothesis:

Adding accepted runtime pseudo-teacher rows from distinct real tracks can improve generalization against the verified `nothing-can-wait` holdout without pretending those runtime stems are ground truth.

Run type:

- Docker-only.
- CUDA-required.
- Same verified training view as v20.
- Add `--pseudo-teacher-examples-jsonl`.
- Holdout excludes `nothing-can-wait` from training and validates against verified `nothing-can-wait`.
- Use the existing detail-repair config first to isolate the data change.

Command:

```bash
docker compose run --rm --no-deps -T ml-worker python tools/run_generic_audio_finetune_matrix.py \
  --audio-training-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/multi_candidate_conditioned_vocal_training_view_v13_strong_scnet_melband_all_verified_no_melody/candidate_conditioned_audio_training_examples.jsonl \
  --pseudo-teacher-examples-jsonl /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/pseudo_teacher_vocal_runtime_accepted_v21/pseudo_teacher_audio_training_examples.jsonl \
  --config-matrix-json /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_adapter_matrix_v18_detail_repair_only_configs.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v21_pseudo_teacher_holdout_ncw \
  --scope vocal_candidate_conditioned_teacher_mask_adapter_matrix_gt90_cycle_001_v21_pseudo_teacher_ncw \
  --stem vocal \
  --exclude-holdout-track-id nothing-can-wait \
  --min-unique-tracks 2 \
  --device cuda \
  --min-sdr-db 3.0 \
  --min-improvement-db 0.5 \
  --min-checkpoint-pass-rate 1.0 \
  --max-holdouts-per-checkpoint 2 \
  --max-source-feature-memory-distance-ops 50000000000
```

After matrix:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_train_ready_audio_model_decision.py \
  --matrix-manifest-json /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v21_pseudo_teacher_holdout_ncw/generic_audio_finetune_matrix_manifest.json \
  --output-dir /app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/train_ready_decision_vocal_matrix_v21_pseudo_teacher_ncw \
  --min-success-rate 0.90
```

## Stop / Go Rules

Continue only if:

- CUDA free memory is at least 4096 MiB before training starts.
- Progress JSONL updates during each long phase.
- Train progress includes batch-level CUDA events.
- Holdout writes rows and manifest.

Stop as failed, not successful, if:

- Progress age stalls beyond the phase timeout.
- CUDA guard fails.
- Training falls back to CPU.
- Holdout uses pseudo-teacher rows as references.
- `release_ready=true` appears anywhere.

Success only if:

- Train-ready decision writes `train_ready=true`.
- `success_rate > 0.90`.
- No leakage/safety blockers.
- `release_ready=false`.

If v21 fails:

- Do not repeat same run.
- Build the next ARIS cycle around verified-corpus expansion or tensor/critic feedback bridge.
