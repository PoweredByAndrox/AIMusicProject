# Production Audio Loop - Iteration 009

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it049/prompt.md`
- Materialized prompt sha256: `7ea68f607e68c2da84b13fe66dfe8281eabff178d165400bd625db36fa3d067d`

## Task

```text
Run strict source-of-truth analysis/funnel for short listen feedback, review handoff, feedback-cycle, and repair-queue routes that consume staged aggregate/subset/replacement packs or final machine gates. Verify that human feedback, rejected baselines, repair queues, or review packets cannot bypass `machine_stem_acceptance_gate`, aggregate validation, semantic review completion, and minus audit evidence. Implement the smallest fail-closed code/test/doc update for the explicit post-gate listen handoff path.
```

## Source-Of-Truth Pass

Code Index was queried first for feedback template, review packet, repair queue,
feedback manifest, and accepted/machine-gate keywords. CodeGraph then narrowed
symbol impact for the feedback template, review handoff, handoff validator,
review packet, feedback manifest, repair queue, and feedback cycle.

Graphify was not exposed in the available tools. Beads CLI was not available in
this environment, so task buckets were tracked with `update_plan`.

Production files inspected:

- `ml_worker_improved/tools/build_short_listen_feedback_template.py`
- `ml_worker_improved/tools/build_short_listen_review_handoff.py`
- `ml_worker_improved/tools/validate_short_listen_review_handoff.py`
- `ml_worker_improved/tools/run_short_listen_review_packet.py`
- `ml_worker_improved/tools/build_short_listen_feedback_manifest.py`
- `ml_worker_improved/tools/run_short_listen_feedback_cycle.py`
- `ml_worker_improved/tools/build_short_listen_repair_queue.py`
- `ml_worker_improved/tools/build_machine_stem_acceptance_gate.py`

Tests inspected:

- `ml_worker_improved/tests/test_build_short_listen_feedback_template.py`
- `ml_worker_improved/tests/test_build_short_listen_review_handoff.py`
- `ml_worker_improved/tests/test_validate_short_listen_review_handoff.py`
- `ml_worker_improved/tests/test_run_short_listen_review_packet.py`
- `ml_worker_improved/tests/test_build_short_listen_feedback_manifest.py`
- `ml_worker_improved/tests/test_run_short_listen_feedback_cycle.py`
- `ml_worker_improved/tests/test_build_short_listen_repair_queue.py`

Docs inspected:

- `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`
- `ml_worker_improved/docs/SHORT_MINUS_QUALITY_AUDIT_README.md`
- `ml_worker_improved/docs/TRAIN_READY_AUDIO_MODEL_DECISION.md`
- `docs/production_audio_loop/ITERATION_008.md`

## Confirmed Facts

- `build_short_listen_feedback_template.py` accepted old
  `delivery_status=accepted` / `presentation_status=accepted_for_listen_pack`
  pack manifests and could create `ready_for_human_review` without a final
  machine gate.
- `build_short_listen_review_handoff.py` and
  `validate_short_listen_review_handoff.py` allowed subset, repair-selection,
  and candidate-wave listen-file pack schemas as review-ready inputs.
- `run_short_listen_review_packet.py` had a subset branch that skipped goal
  audit and could still produce `ready_for_listen_review`.
- `build_short_listen_feedback_manifest.py` could turn human rows into a
  feedback manifest from an old accepted aggregate pack without machine-gate
  evidence.
- The correct post-iteration-008 shape is a staged aggregate pack plus an
  accepted `machine_stem_acceptance_gate.v1` `gate.json`; the aggregate pack
  itself remains `delivery_status=aggregate_validation_pending`.

## Implemented

- Added `tools/short_listen_gate_contract.py` with reusable checks for:
  - staged aggregate pack contract;
  - accepted machine gate contract tied to the same pack;
  - readable aggregate validation evidence tied to the same pack;
  - readable semantic review evidence tied to the same pack;
  - readable minus audit evidence tied to the same pack when the pack contains
    minus rows.
- Changed `build_short_listen_feedback_template.py` to require a staged
  aggregate pack plus accepted `--machine-gate-json`.
- Changed `build_short_listen_review_handoff.py` to reject non-aggregate listen
  pack schemas and recheck the machine gate recorded by the feedback template.
- Changed `validate_short_listen_review_handoff.py` to revalidate staged
  aggregate and accepted machine-gate evidence.
- Changed `run_short_listen_review_packet.py` to accept and pass
  `--machine-gate-json` into both the feedback template and goal audit, and to
  fail closed if the goal audit still reports incomplete machine evidence.
- Changed `build_short_listen_feedback_manifest.py` and
  `run_short_listen_feedback_cycle.py` so human feedback rows cannot become a
  feedback manifest before machine-gate evidence is present.
- Updated docs so post-gate human review is explicitly staged aggregate pack +
  accepted machine gate only; subset/repair/candidate listen-file packs stay
  staging or diagnostic.

## Validation

RED on Mac Apple container, Windows workspace mounted read-only:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_listen_feedback_template.py tests/test_build_short_listen_review_handoff.py tests/test_validate_short_listen_review_handoff.py tests/test_run_short_listen_review_packet.py tests/test_build_short_listen_feedback_manifest.py tests/test_run_short_listen_feedback_cycle.py -q
```

Result before the production fix:

```text
31 failed, 13 passed
```

GREEN after the code/test update and review fixes:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_listen_feedback_template.py tests/test_build_short_listen_review_handoff.py tests/test_validate_short_listen_review_handoff.py tests/test_run_short_listen_review_packet.py tests/test_build_short_listen_feedback_manifest.py tests/test_run_short_listen_feedback_cycle.py -q
```

Result:

```text
49 passed
```

Wider Mac Apple-container regression:

```text
python -m pytest -p no:cacheprovider tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_listen_coverage_scorecard.py tests/test_build_machine_stem_acceptance_gate.py tests/test_build_short_listen_subset_pack.py tests/test_build_short_listen_replacement_pack.py tests/test_audit_short_listen_goal_status.py tests/test_build_short_listen_feedback_template.py tests/test_build_short_listen_review_handoff.py tests/test_validate_short_listen_review_handoff.py tests/test_run_short_listen_review_packet.py tests/test_build_short_listen_feedback_manifest.py tests/test_run_short_listen_feedback_cycle.py tests/test_build_short_listen_repair_queue.py -q
```

Result:

```text
116 passed
```

## Next Loop Task

```text
Run strict source-of-truth analysis/funnel for remaining artifact/listen-pack builders that still emit `delivery_status=accepted`, `delivery_allowed=true`, or `presentation_status=accepted_for_listen_pack` outside the staged aggregate plus accepted machine-gate post-handoff contract. Prioritize no-reference minus/listen-preview routes, dense delivery gates, music-intelligence packs, and stale artifacts referenced by docs. Implement the smallest fail-closed code/test/doc update that preserves diagnostic artifacts while preventing accepted-only delivery bypass.
```
