# ARIS Train >90 Cycle 10 Plan

## Checkpoint

- v10 all-verified candidate-conditioned training view included all 8 verified vocal rows.
- v10 matrix still failed strict train-ready gate.
- Best v10 row:
  - config: `all_verified_magnitude_detail_repair`
  - baseline SDR: about `12.254840 dB`
  - candidate SDR: `12.256870 dB`
  - improvement: `+0.002030 dB`
- v10 train-ready decision: `train_result_not_ready`, `success_rate=0.0`.
- release_ready remains false.

## Reconciliation

The `htdemucs_ft` dirty vocal source is already close to the verified NCW target.
The adapter can learn near-identity, but the strict gate requires at least
`+0.5 dB` improvement. More verified rows did not solve the bottleneck.

The next safe step is not threshold relaxation. It is to choose a registered
source family with measurable repair headroom and then train the same strict
candidate-conditioned repair route on that family.

## Ranked Ideas

### 1. Holdout Source-Family Audit Before Full Training

- Generate NCW vocal candidates from existing registered source families.
- Compute baseline SDR against the verified NCW target.
- Select a repairable family that is not already near-perfect and is not
  catastrophically unrelated to the target.
- Then generate the same family for all 8 verified vocal rows and run strict
  candidate-conditioned matrix.

### 2. Candidate-Family Ensemble Repair View

- Use multiple registered vocal source families as separate source rows while
  preserving the same verified target.
- Train a repair adapter over family-diverse dirty sources.
- Risk: more mixed source distribution can overfit without explicit family
  conditioning.

### 3. Bounded Delta Adapter

- Add a new adapter output mode that learns a bounded residual delta over the
  dirty candidate instead of replacing the candidate with a learned mask.
- Use only if family audit shows all existing source families either have no
  repair headroom or degrade under mask learning.

## Chosen Next Experiment

Run Idea 1:

1. Generate NCW-only candidates for:
   - `htdemucs_6s`
   - `mdx23c_vocals`
   - `melband_roformer_vocals`
   - `scnet_xl_ihf_musdb18`
2. Compute baseline SDR/improvement headroom against verified NCW target.
3. Pick one source family for full 8-row candidate-conditioned training view.
4. Run v11 matrix under unchanged thresholds.
5. Build train-ready decision manifest.

## Contract

- Existing registered models only.
- No model download.
- No threshold relaxation.
- No runtime gate change.
- No model registry edit.
- Dirty candidates are source only.
- Verified stems remain target and holdout reference.
- No fallback to raw mix.
- No release-ready.
- Docker-only execution.
- Progress logs required for every candidate generation and matrix run.
