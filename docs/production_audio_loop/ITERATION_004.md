# Production Audio Loop Iteration 004

## Template Use

- Source template: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edit status: unchanged
- Template SHA-256: `a2e2d507879db7ee90bc07a8a0f6a5262236350a3a32366ce7f247c9d3bd4b13`
- Placeholder count observed: 1
- Replaced task text for this iteration:

```text
Audit the presemantic nonvocal stem listen-preview route. It currently emits `presentation_status=accepted_for_listen_pack` while `semantic_quality_verified=false`, and `run_no_reference_stem_semantic_judge` expects that manifest to be listenable input. Design and implement the smallest production-safe status/route split so presemantic stem artifacts can feed the semantic judge without being presentable as user listen/result artifacts before semantic clearance.
```

## Source-Of-Truth Pass

Inspected evidence:

- `ml_worker_improved/tools/artifact_status_contract.py`
- `ml_worker_improved/tools/build_no_reference_stem_listen_preview_manifest.py`
- `ml_worker_improved/tools/run_no_reference_stem_semantic_judge.py`
- `ml_worker_improved/tools/build_no_reference_stem_candidate_score_matrix.py`
- `ml_worker_improved/tools/build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tools/run_no_reference_stem_semantic_batch.py`
- `ml_worker_improved/tests/test_build_no_reference_stem_listen_preview_manifest.py`
- `ml_worker_improved/tests/test_run_no_reference_stem_semantic_judge.py`
- `ml_worker_improved/tests/test_build_no_reference_stem_candidate_score_matrix.py`
- `ml_worker_improved/tests/test_build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tests/test_run_no_reference_stem_semantic_batch.py`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_SEMANTIC_JUDGE_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_CANDIDATE_SCORE_MATRIX_README.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_SEMANTIC_BATCH_README.md`

Tool evidence:

- `code-index-mcp stats`: index available for `D:\AIMusicProject`.
- `code-index-mcp get-file-summary`: inspected stem preview builder, stem semantic judge, score matrix, and semantic batch source maps.
- CodeGraph inspected `accepted_manifest`, `blocked_manifest`, `build_manifest`, `judge_listen_manifest`, `ArtifactStatusView.listenable`, `matrix_row`, `next_action`, `select_required_stem`, and `row_release_safe`.
- Graphify was not available in the current callable tools; replacement was CodeGraph + code-index + source files.
- Beads/task tracker was not available as a callable CLI/tool; replacement was the Codex plan checklist.

## Confirmed Current Problem

- `build_no_reference_stem_listen_preview_manifest.py` emitted passed presemantic stems as `delivery_status=accepted`, `delivery_allowed=true`, and `presentation_status=accepted_for_listen_pack` while `semantic_quality_verified=false`.
- `ArtifactStatusView.listenable` treats `delivery_allowed=true` plus `accepted_for_listen_pack` as listenable, so the presemantic manifest was user-presentable before semantic clearance.
- `run_no_reference_stem_semantic_judge.py` required `view.listenable`, so simply blocking the manifest would have made all presemantic inputs fail as `listen_manifest_not_accepted`.
- `build_no_reference_stem_candidate_score_matrix.py` also used `view.listenable`, so the split had to preserve downstream semantic-cleared selection without making raw presemantic manifests listenable.

## Raw Ideas

| # | Idea | Status | Reason |
|---|---|---|---|
| 1 | Add explicit semantic-judge-input status for presemantic stem manifests | KEEP | Separates judge transport from user presentation |
| 2 | Let semantic judge accept explicit input-only manifests after gate checks | KEEP | Keeps judge route alive without using listenable status |
| 3 | Make score matrix promote only semantic-accepted input rows to listen-pack rows | KEEP | Restores downstream full-pack path after semantic clearance |
| 4 | Keep old accepted/listenable manifests as legacy judge input | KEEP | Avoids breaking older accepted artifacts in tests/routes |
| 5 | Mark raw presemantic manifests as `blocked` | REJECT | Too ambiguous: blocked diagnostic artifacts should not feed semantic judge |
| 6 | Leave `delivery_status=accepted` but set `delivery_allowed=false` | REJECT | Mixed accepted/non-deliverable status is weaker than explicit pending semantic status |
| 7 | Block every `semantic_quality_verified=false` centrally | DEFER | Still too broad for non-stem routes and diagnostic evidence |
| 8 | Patch full-stem pack to accept input-only rows directly | REJECT | Full pack should consume semantic-cleared matrix rows, not raw presemantic manifests |

## TOP Set

1. Presemantic stem preview manifests are no longer user-presentable.
   - Implemented in this iteration.
2. Semantic judge accepts explicit input-only presemantic manifests.
   - Implemented in this iteration.
3. Score matrix converts only semantic-cleared presemantic inputs into accepted listen-pack selection rows.
   - Implemented in this iteration.
4. Docs state the split between presemantic input, semantic judge output, matrix selection, and release readiness.
   - Implemented in this iteration.

## Implemented In This Set

- Added shared semantic-judge input status constants to `artifact_status_contract.py`.
- `build_no_reference_stem_listen_preview_manifest.py` now emits passed stems as:
  - `delivery_status=pending_semantic_judge`
  - `delivery_allowed=false`
  - `presentation_status=semantic_judge_input_not_for_listen_pack`
  - `release_status=semantic_judge_input_only`
  - `semantic_judge_input=true`
- Added `semantic_judge_input_rows` and `semantic_judge_input_count`; raw `accepted_listen_preview_rows` is empty for presemantic outputs.
- `run_no_reference_stem_semantic_judge.py` accepts explicit presemantic judge inputs only when status, safety, audio, GPU/provenance, and source-contract gates are valid.
- `build_no_reference_stem_candidate_score_matrix.py` keeps raw presemantic rows blocked without semantic judge evidence, but emits matrix rows as accepted/listen-pack selection rows when semantic judge returns `accept_candidate`, `semantic_quality_verified=true`, and no blockers.
- Updated stem preview, semantic judge, score matrix, and semantic batch README docs.
- Added/updated regression tests for presemantic non-presentability, judge-consumable status, blocked non-input manifests, and semantic-cleared matrix acceptance.

## Validation

Apple container targeted RED:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_build_no_reference_stem_listen_preview_manifest.py tests/test_run_no_reference_stem_semantic_judge.py -q
```

Result after test-only changes: expected failures. The old builder still emitted `accepted/accepted_for_listen_pack`, and the old judge rejected presemantic input as `listen_manifest_not_accepted`.

Apple container targeted GREEN:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_build_no_reference_stem_listen_preview_manifest.py tests/test_run_no_reference_stem_semantic_judge.py -q
```

Result: `27 passed`.

Apple container score-matrix RED:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_build_no_reference_stem_candidate_score_matrix.py -q
```

Result after matrix test updates: expected failures proving downstream semantic-cleared input-only rows were still blocked by `view.listenable`.

Apple container score-matrix GREEN:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_build_no_reference_stem_candidate_score_matrix.py -q
```

Result: `10 passed`.

Expanded blast-radius route:

```text
AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_artifact_status_contract.py tests/test_build_no_reference_stem_listen_preview_manifest.py tests/test_run_no_reference_stem_semantic_judge.py tests/test_build_no_reference_stem_candidate_score_matrix.py tests/test_run_no_reference_stem_semantic_batch.py tests/test_build_no_reference_full_stem_pack_manifest.py tests/test_run_no_reference_nonvocal_bounded_wave.py tests/test_run_no_reference_stem_hard_case_repair_wave.py -q
```

Result: `73 passed`.

## Next Loop Task

Use the same template again with this task text:

```text
Audit the semantic-cleared full-stem and aggregate listen-pack boundary. Verify that matrix rows promoted from presemantic stem inputs cannot bypass required stem coverage, source provenance, short-name artifact policy, semantic review evidence, or final validation before becoming user-facing listen packs. Implement the smallest production-safe gate/test/doc update that prevents any semantically unreviewed or coverage-incomplete stem from entering full-stem/aggregate listen outputs.
```
