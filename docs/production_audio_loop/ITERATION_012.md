# Production Audio Loop - Iteration 012

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it052/prompt.md`
- Materialized prompt sha256: `7822a108de11ab8706424c758deda0b68854b4444c246a94e0e20238a748663f`

## Task

```text
Run strict source-of-truth analysis/funnel for the remaining no-reference minus, vocal, and full-stem delivery pipeline surfaces that still write `delivery_status=accepted`, `delivery_allowed=true`, or `presentation_status=accepted_for_listen_pack`. Separate intermediate source-manifest contracts from final user-facing delivery and implement the smallest fail-closed update where any pipeline wrapper can bypass staged aggregate validation plus accepted machine-gate evidence.
```

## Source-Of-Truth Pass

Code Index was queried first for the minus/vocal delivery pipeline files,
accepted pipeline symbols, and related docs. CodeGraph then identified the
`accepted_pipeline_manifest()` blast radius in both wrappers.

Graphify was not exposed in the available tool surface for this iteration.
Fallback coverage was Code Index, CodeGraph, exact source/test/doc file reads,
and Mac Apple-container validation.

Production files inspected:

- `ml_worker_improved/tools/run_no_reference_minus_delivery_pipeline.py`
- `ml_worker_improved/tools/run_no_reference_vocal_delivery_pipeline.py`

Tests inspected:

- `ml_worker_improved/tests/test_run_no_reference_minus_delivery_pipeline.py`
- `ml_worker_improved/tests/test_run_no_reference_vocal_delivery_pipeline.py`
- `ml_worker_improved/tests/test_build_no_reference_minus_listen_preview_manifest.py`
- `ml_worker_improved/tests/test_build_no_reference_vocal_listen_preview_manifest.py`
- `ml_worker_improved/tests/test_run_no_reference_delivery_pipeline.py`
- `ml_worker_improved/tests/test_run_no_reference_delivery_batch.py`
- `ml_worker_improved/tests/test_build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_build_machine_stem_acceptance_gate.py`

Docs inspected:

- `ml_worker_improved/docs/NO_REFERENCE_MINUS_DELIVERY_PIPELINE_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_VOCAL_DELIVERY_PIPELINE_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_MINUS_CANDIDATE_SCORE_MATRIX_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_VOCAL_CANDIDATE_SCORE_MATRIX_README.md`

## Confirmed Facts

- `run_no_reference_minus_delivery_pipeline.py` copied final delivery fields
  from `no_reference_minus_listen_preview_manifest.json` into the wrapper
  manifest when the source preview was promotable.
- `run_no_reference_vocal_delivery_pipeline.py` had the same final-field copy
  from `no_reference_vocal_listen_preview_manifest.json`.
- Existing tests encoded the bypass by expecting wrapper CLI success (`0`) and
  `delivery_allowed=true` whenever the source listen-preview manifest accepted.
- The listen-preview builders remain intentional source-evidence producers;
  they are not the final staged aggregate pack or accepted machine stem gate.

## Implemented

- Kept minus/vocal listen-preview manifests as promotable source evidence.
- Changed both wrapper manifests to fail closed even when source preview is
  promotable:
  - `delivery_status=blocked`;
  - `delivery_allowed=false`;
  - `presentation_status=aggregate_machine_gate_required_not_for_listen_pack`;
  - `release_status=not_release_ready`;
  - `dominant_failure_reason=requires_staged_aggregate_validation_and_machine_gate`;
  - `listen_source_promotable=true`;
  - `requires_staged_aggregate_machine_gate=true`;
  - `final_delivery_blockers=["requires_staged_aggregate_validation_and_machine_gate"]`.
- Preserved source preview provenance on the wrapper manifest:
  - `source_listen_preview_delivery_status`;
  - `source_listen_preview_delivery_allowed`;
  - `source_listen_preview_presentation_status`;
  - `source_listen_preview_release_status`.
- Updated progress logs to report `listen_source_promotable` instead of final
  `delivery_allowed`.
- Updated minus/vocal delivery tests so final wrappers exit fail-closed while
  the source listen-preview manifest can still pass its own contract.
- Updated minus/vocal README files to separate source evidence from final
  user-facing delivery.

## Validation

Targeted Mac Apple-container regression:

```text
python -m pytest -p no:cacheprovider tests/test_run_no_reference_minus_delivery_pipeline.py tests/test_run_no_reference_vocal_delivery_pipeline.py -q --tb=short --disable-warnings
```

Initial result exposed one stale log assertion:

```text
1 failed, 21 passed
```

After updating the assertion:

```text
22 passed
```

Wider affected-layer Mac Apple-container regression:

```text
python -m pytest -p no:cacheprovider tests/test_run_no_reference_minus_delivery_pipeline.py tests/test_run_no_reference_vocal_delivery_pipeline.py tests/test_run_no_reference_delivery_pipeline.py tests/test_run_no_reference_delivery_batch.py tests/test_build_no_reference_minus_listen_preview_manifest.py tests/test_build_no_reference_vocal_listen_preview_manifest.py tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_machine_stem_acceptance_gate.py -q --tb=short --disable-warnings
```

Result:

```text
93 passed
```

Code Index was refreshed after edits:

```text
code-index-mcp index .
```

Subagent audit reported PASS for the wrapper-surface hypothesis: minus/vocal
delivery wrappers no longer have accepted final bypasses; accepted producers
remain only at the intentional source/listen-preview builder layer.

## Next Loop Task

```text
Run strict source-of-truth analysis/funnel for the remaining full-stem and listen-pack assembly surfaces after minus/vocal wrappers are fail-closed. Prioritize any pack builder, manifest aggregator, or user-facing artifact exporter that can still treat source listen-preview evidence as final delivery without a validated aggregate pack plus accepted machine stem gate. Implement the smallest production/test/doc update that makes accepted-only delivery mechanically enforceable end to end.
```
