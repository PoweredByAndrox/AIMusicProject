# Production Audio Loop Iteration 003

## Template Use

- Source template: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edit status: unchanged
- Template SHA-256: `a2e2d507879db7ee90bc07a8a0f6a5262236350a3a32366ce7f247c9d3bd4b13`
- Placeholder count observed: 1
- Replaced task text for this iteration:

```text
Audit the accepted-only delivery gates for vocal/minus listen packs and pair manifests. Verify that raw stem manifests, diagnostic artifacts, and accepted-looking candidates with unresolved semantic leak labels cannot be presented as listen/result artifacts. Implement the smallest production validator or gate update with tests that blocks any artifact missing accepted delivery status, objective gate evidence, or semantic leak/failure clearance.
```

## Source-Of-Truth Pass

Inspected evidence:

- `ml_worker_improved/tools/artifact_status_contract.py`
- `ml_worker_improved/tests/test_artifact_status_contract.py`
- `ml_worker_improved/tools/build_no_reference_listen_preview_manifest.py`
- `ml_worker_improved/tools/build_no_reference_vocal_listen_preview_manifest.py`
- `ml_worker_improved/tools/build_no_reference_minus_listen_preview_manifest.py`
- `ml_worker_improved/tools/build_no_reference_stem_listen_preview_manifest.py`
- `ml_worker_improved/tools/build_no_reference_pair_candidate_from_listen_manifests.py`
- `ml_worker_improved/tools/run_no_reference_stem_semantic_judge.py`
- `ml_worker_improved/docs/NO_REFERENCE_LISTEN_PREVIEW_MANIFEST_README.md`

Tool evidence:

- `code-index-mcp stats`: index available for `D:\AIMusicProject`.
- `code-index-mcp query ArtifactStatusView`: located the central artifact status contract.
- `code-index-mcp query build_stem_manifest_audio_pack`: located the stem audio pack builder.
- Compound code-index phrase queries were insufficient for listen-pack routing and were replaced by CodeGraph source exploration.
- CodeGraph inspected `build_manifest`, `failure_reason`, `accepted_manifest`, `source_contract_gate`, `normalize_manifest_status`, `collect_failure_labels`, and `can_promote_to_listen_result`.

## Confirmed Current Problem

- `ArtifactStatusView.assert_promotable()` blocked only missing provenance, blocked status fields, or missing listenable/release-ready status.
- `normalize_manifest_status()` could infer a dominant failure reason from nested labels, but `ArtifactStatusView.blocked`, `listenable`, and `deliverable` ignored those labels when the manifest also claimed `delivery_allowed=true`.
- Therefore an accepted-looking manifest with nested `issue_labels=["foreign_source_recovered:minus:vocals,mix_conservation_failed"]` could still pass `can_promote_to_listen_result()`.
- `build_no_reference_listen_preview_manifest.py` correctly calls `can_promote_to_listen_result()` for dense delivery gates, so the central contract is the highest-leverage fix.
- Sidecar audit confirmed a second P1 hole: pair/master and dense gates treated generic judge `decision/reasons` as diagnostic-only, so `blocked_hard_case` or leak labels in those fields could still coexist with otherwise accepted pair/dense evidence.

## Raw Ideas

| # | Idea | Status | Reason |
|---|---|---|---|
| 1 | Make `artifact_status_contract` treat blocking failure labels as non-promotable | KEEP | Central fix covers vocal, minus, pair, delivery-gate and listen-preview callers |
| 2 | Normalize comma-packed and structured blocker labels in the contract | KEEP | Current manifests store blockers as strings, lists, and dicts |
| 3 | Add direct contract tests for accepted-looking manifests with semantic blockers | KEEP | Regression proof for the exact failure class |
| 4 | Add a shared opt-in helper for semantic-clearance `decision/reasons` fields | KEEP | Plain `reasons` should not be global failure labels, but judge gates must inspect them explicitly |
| 5 | Block pair/master and dense gates when generic `decision/reasons` carry leak labels | KEEP | Closes sidecar-confirmed P1 holes |
| 6 | Rewrite all listen-preview builders to require semantic judge pass now | DEFER | `run_no_reference_stem_semantic_judge` currently expects the presemantic manifest to be listenable input |
| 7 | Block all `semantic_quality_verified=false` centrally | DEFER | Would break the current semantic judge input route before a replacement status is introduced |
| 8 | Patch each pair/listen builder with duplicated label parsing | REJECT | Duplicates gate logic and misses future builders |
| 9 | Treat every `issue_labels` value as fatal | REJECT | Benign labels such as candidate creation must not block promotion |

## TOP Set

1. Central promotion contract blocks accepted-looking manifests with known failure labels.
   - Implemented in this iteration.
2. Failure-label extraction handles comma-packed labels and structured blockers.
   - Implemented in this iteration.
3. Listen-preview docs state that `accepted_for_listen_pack` requires no blocking failure labels.
   - Implemented in this iteration.
4. Pair/master and dense semantic-clearance gates inspect generic judge `decision/reasons`.
   - Implemented in this iteration.

## Implemented In This Set

- Added blocking failure markers to `artifact_status_contract.py`.
- Added normalized extraction for `issue_labels`, `failure_labels`, `failed_checks`, `blocked_checks`, `blocking_issue_labels`, `blockers`, `blocking_reasons`, `check_names`, and `forbidden_actions`.
- Added structured blocker extraction from dict fields such as `code`, `label`, `name`, `id`, `check`, and `reason`.
- Added `ArtifactStatusView.blocking_failure_labels`.
- `ArtifactStatusView.listenable`, `deliverable`, `blocked`, and `assert_promotable()` now fail closed when blocking failure labels exist.
- Added `blocking_labels_from_values()` for explicit gate checks over judge evidence such as generic `decision/reasons`.
- `build_no_reference_vocal_minus_master_manifest.py` and `build_no_reference_segment_preview_pair_listen_manifest.py` now block pair rows when generic both-mode `decision/reasons` carry semantic blockers.
- `build_no_reference_delivery_gate.py` and `build_no_reference_listen_preview_manifest.py` now block dense gate evidence when embedded mode `decision/reasons` carry semantic blockers.
- Added regression tests for accepted-looking listen-pack manifests containing semantic leak/mix-conservation labels and structured blockers.
- Added a pair-candidate regression proving accepted-looking source stem manifests with semantic failure labels are rejected through the shared promotion contract.
- Added master, segment-pair, dense-gate, and final-listen regression tests for semantic blocker labels in generic `decision/reasons`.
- Updated listen-preview, master, segment-pair, and dense-gate README docs.

## Validation

Apple container route:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_artifact_status_contract.py -q
```

Result: `13 passed`.

Expanded blast-radius route:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_artifact_status_contract.py tests/test_build_no_reference_listen_preview_manifest.py tests/test_build_no_reference_vocal_listen_preview_manifest.py tests/test_build_no_reference_minus_listen_preview_manifest.py tests/test_build_no_reference_vocal_minus_master_manifest.py tests/test_build_no_reference_pair_candidate_from_listen_manifests.py tests/test_build_no_reference_segment_preview_pair_listen_manifest.py tests/test_build_no_reference_delivery_gate.py -q
```

Result: `96 passed`.

Note: the first expanded run with only `pytest` installed failed during collection because `test_build_no_reference_segment_preview_pair_listen_manifest.py` imports `numpy` and `soundfile`; the successful Apple container run installed those test-only packages inside the container.

## Next Loop Task

Use the same template again with this task text:

```text
Audit the presemantic nonvocal stem listen-preview route. It currently emits `presentation_status=accepted_for_listen_pack` while `semantic_quality_verified=false`, and `run_no_reference_stem_semantic_judge` expects that manifest to be listenable input. Design and implement the smallest production-safe status/route split so presemantic stem artifacts can feed the semantic judge without being presentable as user listen/result artifacts before semantic clearance.
```
