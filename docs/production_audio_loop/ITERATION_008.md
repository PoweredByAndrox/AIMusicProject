# Production Audio Loop - Iteration 008

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template placeholder count: 1
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it048/prompt.md`
- Materialized prompt sha256: `bfd23243079de1866d2f727229127cd07bba566c6eafe24c0870144f8c0c632c`

## Task

```text
Run strict source-of-truth analysis/funnel for aggregate short-listen pack builders, validators, and post-aggregate promotion routes that can emit `delivery_status=accepted`, `delivery_allowed=true`, or `presentation_status=accepted_for_listen_pack` before semantic review completion, minus audit evidence, aggregate validation, and final machine_stem_acceptance_gate evidence. Implement the smallest fail-closed code/test/doc update that separates diagnostic/review/listen-preview materialization from explicit post-gate promotion.
```

## Source-Of-Truth Pass

Code-index and CodeGraph were used before edits. Sequential Thinking was used
for decomposition. Graphify was not exposed by the available tools. Beads CLI
(`bd`, `beads`, `bead`) was unavailable in this environment, so the task buckets
were tracked with `update_plan`.

Production code inspected:

- `ml_worker_improved/tools/build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/build_listen_coverage_scorecard.py`
- `ml_worker_improved/tools/build_machine_stem_acceptance_gate.py`
- `ml_worker_improved/tools/build_short_listen_subset_pack.py`
- `ml_worker_improved/tools/build_short_listen_replacement_pack.py`
- `ml_worker_improved/tools/audit_short_listen_goal_status.py`

Tests inspected:

- `ml_worker_improved/tests/test_build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_build_listen_coverage_scorecard.py`
- `ml_worker_improved/tests/test_build_machine_stem_acceptance_gate.py`
- `ml_worker_improved/tests/test_build_short_listen_subset_pack.py`
- `ml_worker_improved/tests/test_build_short_listen_replacement_pack.py`
- `ml_worker_improved/tests/test_audit_short_listen_goal_status.py`

Docs inspected:

- `ml_worker_improved/docs/SHORT_MINUS_QUALITY_AUDIT_README.md`
- `ml_worker_improved/docs/TRAIN_READY_AUDIO_MODEL_DECISION.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`
- `docs/production_audio_loop/ITERATION_007.md`

## Confirmed Facts

- `build_aggregate_short_listen_pack.py` copied source audio and marked a
  successful aggregate pack as `delivery_status=accepted`,
  `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack`.
- `validate_aggregate_short_listen_pack.py` required that same accepted/listen
  status before it would validate an aggregate pack.
- `build_listen_coverage_scorecard.py` treated an accepted/delivery-allowed
  aggregate pack as the normal scorecard input through `schema_failures`.
- `build_machine_stem_acceptance_gate.py` required the aggregate pack itself to
  already be accepted/delivery-allowed before it could emit the final machine
  acceptance gate.
- Sidecar review found two additional post-aggregate promotion surfaces:
  `build_short_listen_subset_pack.py` and
  `build_short_listen_replacement_pack.py` both produced accepted/listen packs
  before the final machine gate.
- `audit_short_listen_goal_status.py` also carried the old pre-gate contract:
  its delivery check expected an accepted/delivery-allowed aggregate pack.
- Docs still described aggregate packs as accepted listen packs, so code/tests
  and docs all preserved a pre-gate accepted/listen status boundary.

## Implemented

- Changed successful aggregate pack materialization to a staging contract:
  - `delivery_status=aggregate_validation_pending`;
  - `delivery_allowed=false`;
  - `presentation_status=aggregate_validation_pending_not_for_listen_pack`;
  - `release_status=aggregate_validation_pending`;
  - `listenable=false`;
  - `aggregate_validation_ready=true`.
- Kept blocked aggregate packs fail-closed with no copied audio retained.
- Changed the aggregate builder CLI success condition from `delivery_allowed` to
  `aggregate_validation_ready`.
- Changed the aggregate validator to accept only staged aggregate packs and to
  reject old pre-accepted aggregate packs.
- Changed the coverage scorecard input contract to require staged aggregate
  packs instead of accepted aggregate packs.
- Changed `build_machine_stem_acceptance_gate.py` so the final machine gate is
  the explicit promotion point from staged aggregate evidence to
  `delivery_status=accepted` / `delivery_allowed=true`.
- Changed `build_short_listen_subset_pack.py` so selected subset packs remain
  aggregate-validation staging artifacts instead of accepted listen packs.
- Changed `build_short_listen_replacement_pack.py` so replacement aggregate
  outputs remain staged and the base aggregate input must already use the staged
  aggregate-validation contract.
- Changed `audit_short_listen_goal_status.py` so goal status auditing accepts
  staged aggregate packs and still requires the final machine gate to prove
  machine evidence.
- Added explicit `listenable=false` guards to the coverage scorecard and final
  machine gate so a manually edited staged manifest cannot stay listenable.
- Review found stale `release_status` gaps in the scorecard and replacement
  base-pack checks; both now require
  `release_status=aggregate_validation_pending`.
- Updated docs to state that aggregate packs are validation staging artifacts
  until validation, semantic review, minus audit, and the final machine gate are
  present.

## Validation

RED on Mac Apple container, Windows workspace mounted read-only:

```text
python -m pytest -p no:cacheprovider tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_listen_coverage_scorecard.py tests/test_build_machine_stem_acceptance_gate.py -q
```

Result before the production fix:

```text
8 failed, 30 passed
```

Additional RED for the sidecar-found subset/replacement promotion surfaces:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_listen_subset_pack.py tests/test_build_short_listen_replacement_pack.py -q
```

Result before the subset/replacement production fix:

```text
5 failed, 2 passed
```

GREEN after the code/test update:

```text
python -m pytest -p no:cacheprovider tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_listen_coverage_scorecard.py tests/test_build_machine_stem_acceptance_gate.py -q
```

Result:

```text
38 passed
```

Subset/replacement GREEN:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_listen_subset_pack.py tests/test_build_short_listen_replacement_pack.py -q
```

Result:

```text
7 passed
```

Goal-audit RED/GREEN for staged aggregate inputs:

```text
python -m pytest -p no:cacheprovider tests/test_audit_short_listen_goal_status.py -q
```

Result after production fix:

```text
8 passed
```

Consolidated affected-layer validation:

```text
python -m pytest -p no:cacheprovider tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_listen_coverage_scorecard.py tests/test_build_machine_stem_acceptance_gate.py tests/test_build_short_listen_subset_pack.py tests/test_build_short_listen_replacement_pack.py tests/test_audit_short_listen_goal_status.py tests/test_run_short_listen_review_packet.py -q
```

Result:

```text
62 passed
```

Post-review release-status focused validation:

```text
python -m pytest -p no:cacheprovider tests/test_build_listen_coverage_scorecard.py tests/test_build_short_listen_replacement_pack.py -q
```

Result:

```text
14 passed
```

## Next Loop Task

```text
Run strict source-of-truth analysis/funnel for short listen feedback, review handoff, and repair-queue routes that consume staged aggregate/subset/replacement packs or final machine gates. Verify that human feedback, rejected baselines, or repair queues cannot bypass `machine_stem_acceptance_gate`, aggregate validation, semantic review completion, and minus audit evidence. Implement the smallest fail-closed code/test/doc update for the explicit post-gate listen handoff path.
```
