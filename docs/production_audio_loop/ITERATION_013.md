# Production Audio Loop - Iteration 013

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it053/prompt.md`
- Materialized prompt sha256: `855a592e63cc263aa74afb96897c955dc93f28c099d5cc4f9e4fb277fe402e23`

## Task

```text
Run strict source-of-truth analysis/funnel for the remaining full-stem and listen-pack assembly surfaces after minus/vocal wrappers are fail-closed. Prioritize any pack builder, manifest aggregator, or user-facing artifact exporter that can still treat source listen-preview evidence as final delivery without a validated aggregate pack plus accepted machine stem gate. Implement the smallest production/test/doc update that makes accepted-only delivery mechanically enforceable end to end.
```

## Source-Of-Truth Pass

Code Index was queried first for `listen_pack`, `aggregate_machine_gate`,
`accepted_for_listen_pack`, `full_stem`, and
`no_reference_full_stem_pack_manifest`. CodeGraph then inspected the
full-stem builder, aggregate short-listen pack, aggregate validator, machine
stem acceptance gate, and corpus planner/runner blast radius.

Graphify was not exposed in the available tool surface for this iteration.
Fallback coverage was Code Index, CodeGraph, exact source/test/doc reads, and
Mac Apple-container validation.

Production files inspected:

- `ml_worker_improved/tools/build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tools/build_testing_stuff_music_corpus_vocal_minus_master_plan.py`
- `ml_worker_improved/tools/run_testing_stuff_music_corpus_vocal_minus_master_plan.py`
- `ml_worker_improved/tools/build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/build_machine_stem_acceptance_gate.py`
- `ml_worker_improved/tools/build_no_reference_delivery_gate.py`

Tests inspected:

- `ml_worker_improved/tests/test_build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tests/test_build_testing_stuff_music_corpus_vocal_minus_master_plan.py`
- `ml_worker_improved/tests/test_run_testing_stuff_music_corpus_vocal_minus_master_plan.py`
- `ml_worker_improved/tests/test_build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_build_machine_stem_acceptance_gate.py`

Docs inspected:

- `ml_worker_improved/docs/NO_REFERENCE_FULL_STEM_PACK_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_HARD_CASE_REPAIR_WAVE_README.md`
- `ml_worker_improved/docs/TESTING_STUFF_MUSIC_CORPUS_VOCAL_MINUS_MASTER_PLAN_README.md`
- `ml_worker_improved/docs/TESTING_STUFF_MUSIC_CORPUS_VOCAL_MINUS_MASTER_EXECUTOR_README.md`

## Confirmed Facts

- `build_aggregate_short_listen_pack.py` already stays fail-closed with
  `delivery_status=aggregate_validation_pending`, `delivery_allowed=false`,
  and `presentation_status=aggregate_validation_pending_not_for_listen_pack`.
- `validate_aggregate_short_listen_pack.py` explicitly expects the aggregate
  pack to remain pending and non-listenable before final gate promotion.
- `build_machine_stem_acceptance_gate.py` is the first inspected layer allowed
  to emit `delivery_status=accepted`, `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack`.
- `build_no_reference_full_stem_pack_manifest.py` still emitted final accepted
  delivery fields for source-only full-stem pack evidence.
- The corpus planner accepted only old final-style full-stem pack manifests;
  after the builder became fail-closed, the planner would otherwise stop seeing
  source-ready full-stem evidence.
- The corpus runner treated the full-stem pack step as blocked whenever the
  builder did not return `delivery_allowed=true`, even though fail-closed
  source-ready evidence is now the expected successful output of that step.

## Implemented

- Changed the full-stem pack manifest builder so successful source selection is
  still final-blocked:
  - `delivery_status=blocked`;
  - `delivery_allowed=false`;
  - `presentation_status=aggregate_machine_gate_required_not_for_listen_pack`;
  - `release_status=not_release_ready`;
  - `source_pack_ready=true`;
  - `requires_staged_aggregate_machine_gate=true`;
  - `final_delivery_blockers=["requires_staged_aggregate_validation_and_machine_gate"]`.
- Updated the full-stem summary to show `source_pack_ready` and
  `requires_staged_aggregate_machine_gate`.
- Added a strict planner contract for full-stem source-ready manifests so
  `--full-stem-pack-root` no longer requires or accepts final delivery fields.
- Renamed the completed planner next gate to
  `full_stem_pack_source_ready_requires_aggregate_machine_gate`.
- Updated the corpus runner full-stem step status to
  `completed_source_ready_requires_aggregate_machine_gate` when the builder
  writes expected fail-closed source-ready evidence.
- Updated fail-fast handling so source-ready completed steps do not stop later
  runnable gates.
- Updated tests and docs to remove the old assumption that full-stem pack
  manifests are accepted listen-pack artifacts.

## Validation

Targeted Mac Apple-container regression:

```text
python -m pytest -p no:cacheprovider tests/test_build_no_reference_full_stem_pack_manifest.py tests/test_build_testing_stuff_music_corpus_vocal_minus_master_plan.py tests/test_run_testing_stuff_music_corpus_vocal_minus_master_plan.py tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_machine_stem_acceptance_gate.py -q --tb=short --disable-warnings
```

Initial container collection failed because the slim image did not include
`numpy`. After installing the required test dependencies inside the Apple
container route:

```text
54 passed
```

Additional checks:

```text
git diff --check
code-index-mcp index .
code-index-mcp grep-body --pattern "full_stem_source_ready_contract" --limit 20
code-index-mcp grep-body --pattern "completed_source_ready_requires_aggregate_machine_gate" --limit 20
```

Result: `git diff --check` passed with only existing line-ending conversion
warnings; Code Index refreshed 12 changed files and located the new planner
contract plus runner step status in production/test symbols.

## Review

PASS:

- The full-stem builder no longer emits final listen delivery fields.
- The planner can still advance corpus state using source-ready full-stem
  evidence without treating it as a user-facing listen result.
- The runner records a successful source-ready step while keeping top-level
  `delivery_allowed=false` and `release_ready=false`; fail-fast continues after
  that successful source-ready step.
- Aggregate validation and machine stem gate remain the inspected final
  promotion path.

Remaining risk:

- This iteration is contract enforcement only. It does not generate new audio or
  prove improved separation quality metrics.

## Next Loop Task

Run the next strict funnel over remaining generic finetuning corpus and
training-data ingestion surfaces that still accept listen-preview or diagnostic
manifests with `delivery_allowed=true` as model-training ground truth without
requiring the post-aggregate machine-gate provenance chain.
