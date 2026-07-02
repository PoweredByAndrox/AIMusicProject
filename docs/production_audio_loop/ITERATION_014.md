# Production Audio Loop - Iteration 014

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it054/prompt.md`
- Materialized prompt sha256: `02821ff68d08eed33a6e56d2bd4266a1f3cdc6e7919ccf636873c5d870666fb1`

## Task

```text
Run strict source-of-truth analysis/funnel for training, finetuning, and problem-discovery ingestion surfaces that can still treat listen-preview, source-ready, blocked, diagnostic, or rejected audio manifests as model-training ground truth without post-aggregate validation plus accepted machine stem gate provenance. Prioritize generic finetuning corpus builders, training-ready manifest selectors, replay/repair queues, and any artifact-to-training conversion path that can consume `delivery_allowed=true` source evidence or diagnostic audio bytes without the accepted-only production chain. Implement the smallest production/test/doc update that makes training/problem-discovery inputs mechanically require accepted post-gate provenance or explicitly mark the batch diagnostic-only and non-training.
```

## Source-Of-Truth Pass

Code Index was queried first for `build_generic_finetuning_corpus_from_lab_artifacts`,
`accepted_contract`, `delivery_allowed`, `training_started`, `train_ready`, and
`diagnostic-only`. CodeGraph then inspected the generic corpus builder, corpus
validator, finetune planner, train-ready decision gate, candidate-conditioned
training views, and the `machine_stem_acceptance_gate.v1` source contract.
Sidecar audit then identified the adjacent recovery-queue/decision-ingest bridge
as the remaining higher-risk artifact-to-training conversion path, so that
bridge was inspected and fixed in the same iteration.

Graphify was not exposed in the available tool surface for this iteration.
Fallback coverage was Code Index, CodeGraph, exact source/test/doc reads,
session-memory reconciliation, and Mac Apple-container validation.

Production files inspected:

- `ml_worker_improved/tools/build_generic_finetuning_corpus_from_lab_artifacts.py`
- `ml_worker_improved/tools/build_generic_audio_training_recovery_queue.py`
- `ml_worker_improved/tools/ingest_generic_audio_training_recovery_decisions.py`
- `ml_worker_improved/tools/validate_verified_audio_training_corpus.py`
- `ml_worker_improved/tools/plan_generic_audio_finetune_jobs.py`
- `ml_worker_improved/tools/build_train_ready_audio_model_decision.py`
- `ml_worker_improved/tools/build_candidate_conditioned_audio_training_view.py`
- `ml_worker_improved/tools/build_multi_candidate_conditioned_audio_training_view.py`
- `ml_worker_improved/tools/build_machine_stem_acceptance_gate.py`

Tests inspected:

- `ml_worker_improved/tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py`
- `ml_worker_improved/tests/test_build_generic_audio_training_recovery_queue.py`
- `ml_worker_improved/tests/test_ingest_generic_audio_training_recovery_decisions.py`
- `ml_worker_improved/tests/test_validate_verified_audio_training_corpus.py`
- `ml_worker_improved/tests/test_plan_generic_audio_finetune_jobs.py`
- `ml_worker_improved/tests/test_build_train_ready_audio_model_decision.py`
- `ml_worker_improved/tests/test_build_candidate_conditioned_audio_training_view.py`
- `ml_worker_improved/tests/test_build_multi_candidate_conditioned_audio_training_view.py`

Docs inspected:

- `ml_worker_improved/docs/GENERIC_FINETUNING_CORPUS_README.md`
- `ml_worker_improved/docs/TRAIN_READY_AUDIO_MODEL_DECISION.md`

## Confirmed Facts

- `build_generic_finetuning_corpus_from_lab_artifacts.py` previously treated
  `delivery_allowed=true` plus `presentation_status=accepted_for_listen_pack`
  as a generic accepted contract.
- Audio training rows were not created from accepted manifests alone, but a
  positive human feedback row could still turn any accepted listen-preview
  manifest with selected audio into `generic_finetuning_audio_training_example`.
- The recovery queue and recovery-decision ingest bridge could also treat
  recovered human-feedback/candidate-map audio pairs as ready or approved audio
  training examples without accepted machine-gate provenance.
- The corpus validator independently accepted old or handwritten human-verified
  rows without `source_gate.accepted_machine_gate=true`, as long as the row had
  existing audio paths and trainable permission.
- `build_machine_stem_acceptance_gate.py` is the inspected post-aggregate
  machine gate that emits `schema=machine_stem_acceptance_gate.v1`,
  `gate_status=accepted`, `final_result_allowed=true`, `delivery_allowed=true`,
  and `presentation_status=accepted_for_listen_pack` only after aggregate
  validation, required stem coverage, machine quality evidence, source
  pair-compatibility, and minus audit gates pass.
- The finetune planner and train-ready decision gate already keep training
  artifacts non-delivery and reject rows or matrix manifests that claim
  `delivery_allowed=true`.

## Implemented

- Added `accepted_machine_gate_contract()` to recognize only accepted
  `machine_stem_acceptance_gate.v1` provenance with all required machine gates,
  empty `failed_checks`, and `blocker_count=0`.
- Added `accepted_machine_gate_contract` to inventory rows and
  `accepted_machine_gate_like` to the corpus summary.
- Changed human-positive audio training ingestion so accepted listen-preview or
  source-ready manifests without accepted machine-gate provenance are written to
  `excluded_examples.jsonl` as
  `accepted_but_missing_accepted_machine_gate_provenance`.
- Preserved reference-verified and verified ground-truth bundle paths as the
  reference-based training route.
- Added `source_gate` provenance to emitted audio training rows.
- Changed the recovery queue so non-reference human-feedback rows receive
  `missing_accepted_machine_gate_provenance` unless the source manifest is an
  accepted `machine_stem_acceptance_gate.v1`.
- Changed recovery decision ingest so old or manual queue rows cannot become
  approved audio training examples unless they are reference-verified or carry
  `accepted_machine_gate=true`.
- Changed the corpus validator so human-only rows require
  `source_gate.accepted_machine_gate=true`; reference-verified rows remain
  allowed through the reference path.
- Updated recovery queue, recovery ingest, and validator tests to cover blocked
  plain listen-preview/candidate-map rows and accepted machine-gate positive
  rows.
- Updated corpus tests to cover the blocked listen-preview case and the accepted
  machine-gate positive path.
- Updated `GENERIC_FINETUNING_CORPUS_README.md` and
  `TRAIN_READY_AUDIO_MODEL_DECISION.md` with the new post-gate/reference
  training eligibility contract.

## Validation

- RED: Mac Apple container, `tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py`
  failed as expected with `audio_training_examples == 1` for a plain accepted
  listen-preview manifest and missing `accepted_machine_gate_like`.
- RED: Mac Apple container, recovery queue/ingest tests failed as expected
  because plain listen-preview rows were still ready, machine-gate provenance was
  missing from queue rows, and ungated recovery rows could still be approved.
- RED: Mac Apple container, validator test failed as expected because a
  human-verified audio row without machine-gate/reference provenance still
  passed corpus validation.
- GREEN: Mac Apple container, `tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py`
  passed: `13 passed`.
- GREEN: Mac Apple container, recovery queue and recovery decision ingest tests
  passed: `24 passed`.
- GREEN: Mac Apple container, validator tests passed: `6 passed`.
- Related non-torch suite passed in Mac Apple container:
  `tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py`,
  `tests/test_build_generic_audio_training_recovery_queue.py`,
  `tests/test_ingest_generic_audio_training_recovery_decisions.py`,
  `tests/test_validate_verified_audio_training_corpus.py`,
  `tests/test_plan_generic_audio_finetune_jobs.py`, and
  `tests/test_build_train_ready_audio_model_decision.py`: `55 passed`.
- Torch-dependent candidate-conditioned tests were attempted in the same Apple
  `python:3.11-slim` container and collection was blocked by missing `torch`.
  No host-runtime result was used as validation.

## Review

PASS:

- The change narrows audio training eligibility rather than widening it.
- Controller and hard-eval metadata training still receive accepted/rejected
  route signals without promoting diagnostic audio bytes into separator targets.
- Delivery/listen semantics remain unchanged: corpus/recovery/validation
  artifacts still write `delivery_allowed=false`, `training_started=false`, and
  non-listen `presentation_status` values.

Remaining risk:

- Candidate-conditioned view tests require a torch-capable Apple-container image
  or explicit dependency install route for full validation in future iterations.
