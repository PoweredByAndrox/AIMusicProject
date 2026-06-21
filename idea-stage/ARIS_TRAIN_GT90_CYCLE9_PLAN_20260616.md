# ARIS Train >90 Cycle 9 Plan

## Checkpoint

- Goal remains active: verified TRAIN success rate must be strictly greater than `0.90`.
- v9 candidate-conditioned route used `htdemucs_ft` dirty vocal candidates as source and verified vocal stems as target.
- v9 fixed registered-candidate observability and disabled unnecessary melody export for `target_stem=vocal`.
- v9 matrix status: `failed`.
- v9 train-ready decision: `train_result_not_ready`, `success_rate=0.0`.
- Best v9 holdout row:
  - baseline SDR: `12.254840 dB`
  - adapter candidate SDR: `12.251416 dB`
  - improvement: `-0.003424 dB`
  - failure: `improvement_below_threshold`

## Reconciliation

The v9 route did not fail because the holdout source was unusable. The dirty
`htdemucs_ft` NCW vocal source was already strong against the verified NCW target.
The adapter failed because it could not learn a positive repair delta from only two
non-NCW unique training tracks and only one Square example was used due to
`max_examples_per_track=1`.

The merged verified vocal corpus currently has 8 trainable verified vocal rows:

- 1 Hotstepper/Evian row
- 6 Square/Oxygen rows
- 1 Nothing Can Wait holdout row

v9 used only 3 candidate-conditioned rows. v10 must use all verified rows as
candidate-conditioned source/target pairs before changing model architecture.

## Ranked Ideas

### 1. Bulk Candidate-Conditioned Verified Vocal View

- Generate `htdemucs_ft` candidate vocal sources for every verified vocal row.
- Build a candidate-conditioned view over all 8 verified vocal rows.
- Exclude `nothing-can-wait` from training and hold it out for evaluation.
- Set `max_examples_per_track` high enough to keep all Square variants.
- Dirty separator candidates remain source only; verified vocal stems remain target.
- Success criterion: NCW holdout passes `min_sdr_db=3.0`, `min_improvement_db=0.5`, and strict decision success rate `>0.90`.

### 2. Delta-Clamped Candidate Repair

- Add a candidate-repair adapter mode that predicts a bounded residual delta over
  the dirty candidate rather than a full mask replacement.
- Use only after Idea 1 proves more verified rows are insufficient.
- Success criterion remains unchanged.

### 3. Candidate Ensemble Source Selection

- Generate multiple dirty candidate source families per verified row and train on
  candidate family metadata or route-specific views.
- More expensive; use only after single-family bulk candidate-conditioned view fails.

## Chosen Next Experiment

Implement Idea 1 first because it uses existing production tools and increases
verified train diversity without touching thresholds, runtime gates, model registry,
or release status.

## Required Artifacts

- `candidate_conditioned_vocal_sources_v3_all_verified_no_melody/`
- `candidate_conditioned_vocal_training_view_v3_all_verified_no_melody/`
- `vocal_adapter_matrix_v10_candidate_conditioned_all_verified_configs.json`
- `vocal_shared_adapter_matrix_v10_candidate_conditioned_all_verified_holdout_ncw/`
- `train_ready_decision_vocal_matrix_v10_candidate_conditioned_all_verified_ncw/`

## Contract

- No fallback from missing candidate source to raw mix.
- No dirty source treated as ground truth.
- No pseudo targets in holdout.
- No threshold relaxation.
- No runtime gate change.
- No model registry edit.
- No release-ready.
- Docker-only execution.
- Progress logs required for every long run.
