# ARIS Train >90 Cycle 12 Plan

## Checkpoint

v11 `htdemucs_6s` candidate-conditioned matrix completed with Docker/CUDA progress logs.

- matrix: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/vocal_shared_adapter_matrix_v11_htdemucs_6s_candidate_conditioned_holdout_ncw`
- decision: `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/train_ready_decision_vocal_matrix_v11_htdemucs_6s_candidate_conditioned_ncw`
- decision: `train_result_not_ready`
- success_rate: `0.0`
- release_ready: `false`

Observed rows:

- `htd6s_magnitude_detail_repair`: mean_improvement_db `-0.011380`
- `htd6s_magnitude_high_detail`: mean_improvement_db `-0.011661`
- `htd6s_complex_projected_memory`: mean_improvement_db `-9.481463`

## Reconciliation

Confirmed:

- v11 used 8 verified vocal rows and excluded `nothing-can-wait` from training.
- v11 ran on Docker/CUDA with matrix, train, holdout, and per-epoch progress artifacts.
- v11 did not relax thresholds and did not mark `release_ready`.
- v11 did not produce a train-ready model.

Unconfirmed:

- whether a single source family is sufficient for cross-track transfer;
- whether source-family diversity improves the adapter;
- whether the current adapter can learn generic vocal repair from only two train track identities.

## Cycle 12 Idea

Build a multi-family candidate-conditioned training view.

Instead of training from only one source candidate per verified target, create one training example per `(verified target, registered source family)`.

This keeps the same target contract:

- candidate source is never ground truth;
- verified target is preserved;
- missing candidate blocks, never falls back to raw mix;
- output artifacts stay training-only;
- `release_ready=false`.

## First Experiment

Use the already available all-verified candidate roots:

- `htdemucs_6s`
- `htdemucs_ft`

Build a 16-row multi-family view with `htdemucs_6s` emitted first so the NCW holdout source has repair headroom instead of an already target-like source.

Then run a small v12 matrix and require the same holdout gates:

- `min_sdr_db=3.0`
- `min_improvement_db=0.5`
- `min_checkpoint_pass_rate=1.0`
- `release_ready=false`

## Stop / Go

Pass:

- decision `train_ready`
- success_rate strictly above `0.90`
- no leakage / no fallback / no release-ready

Fail:

- if success_rate remains `0.0`, expand the multi-family view to additional registered source families (`scnet_xl_ihf_musdb18`, `melband_roformer_vocals`, `mdx23c_vocals`) after generating missing verified-row candidates.

