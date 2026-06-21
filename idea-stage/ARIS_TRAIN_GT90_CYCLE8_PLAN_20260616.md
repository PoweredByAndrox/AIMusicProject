# ARIS Train >90 Cycle 8 Plan

## Checkpoint

- Goal remains active: verified TRAIN success rate must be strictly greater than `0.90`.
- v6 complex-mask best NCW holdout SDR: `0.303320 dB`, failed `min_sdr_db=3.0`.
- v7 complex source-feature memory best NCW holdout SDR: `-0.314669 dB`, failed.
- v8 pseudo-teacher expansion built 3 pseudo vocal rows from accepted runtime stems and trained 3 GPU configs.
- v8 best NCW holdout SDR: `0.380684 dB`, improvement `+8.155055 dB`, failed `min_sdr_db=3.0`.
- v8 train-ready decision: `train_result_not_ready`, `success_rate=0.0`, `release_ready=false`.

## Reconciliation

The pseudo-teacher route increased training diversity from 2 verified non-NCW
tracks to 5 training tracks, but it did not cross the absolute SDR gate. The
bottleneck is now the raw-mix-to-vocal adapter formulation: the model must learn
separation and repair at once from too little verified target data.

The next train-safe route should not lower thresholds, not claim pseudo stems as
ground truth, and not create a fallback. It should change the supervised task:

```text
dirty separator vocal candidate -> verified vocal target
```

Dirty separator stems are allowed only as model inputs. Verified/human/reference
stems remain the only targets and the only holdout references.

## Ranked Ideas

### 1. Candidate-Conditioned Vocal Repair View

- Build a training/eval view where `source_audio_path` is a reproducible
  no-reference vocal candidate for the same track and `target_audio_path` is the
  existing verified vocal target.
- Train the same teacher-mask adapter as a repair adapter over candidate vocals
  instead of raw mixes.
- Holdout NCW must also use a candidate vocal as source and the verified NCW
  target as target.
- Success criterion: locked holdout row passes `min_sdr_db=3.0` and
  `min_improvement_db=0.5`; aggregate decision remains strict `>0.90`.
- Risk: matching candidate paths to verified rows can be sparse. Missing
  candidate source must block, not fall back to raw mix.

### 2. Candidate-Plus-Mix Dual Feature Adapter

- Extend the model input to include both raw mix STFT and candidate-vocal STFT.
- Target remains verified vocal.
- Success criterion same as above.
- Risk: requires larger model/code changes than idea 1.

### 3. Multi-Stem Consistency Repair

- Train vocal repair and minus repair jointly with mixture consistency loss.
- Target remains verified stems only where available.
- Risk: current verified corpus is too narrow and code change is larger.

## Chosen Next Experiment

Implement Idea 1 first because it reuses the existing adapter, holdout gate, and
matrix runner with the smallest new contract surface.

## Required Artifacts

- `tools/build_candidate_conditioned_audio_training_view.py`
- `tests/test_build_candidate_conditioned_audio_training_view.py`
- docs update in `docs/GENERIC_FINETUNING_CORPUS_README.md`
- candidate-conditioned training view under
  `/app/lab_artifacts/generic_finetuning_playbook_20260616/gt90_cycle_001/`
- v9 matrix config and run artifacts
- train-ready decision artifact

## Contract

- No threshold relaxation.
- No release-ready.
- No runtime gate change.
- No model registry edit.
- No fallback from missing candidate source to raw mix.
- Dirty stems can be `source_audio_path` only.
- `target_audio_path` must stay verified/human/reference training target.
- Holdout must use the same candidate-conditioned view, not pseudo targets.
- Docker-only execution.
- Progress logs required for every long run.
