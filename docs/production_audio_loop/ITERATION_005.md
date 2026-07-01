# Production Audio Loop - Iteration 005

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template placeholder count: 1
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it045/prompt.md`
- Materialized prompt sha256: `006c568ca70dae4c58ce44ad9e01f8c4645865bd86811eb7f141313d357abed5`

## Task

```text
Strict source-of-truth audit/funnel for the accepted aggregate full-stem listen-pack boundary after 35017a0: current code allows tools/build_aggregate_short_listen_pack.py and tools/validate_aggregate_short_listen_pack.py to treat required instrument stems as accepted listen-pack content even when those rows still have semantic_quality_verified=false and no machine semantic-quality evidence. Starting from actual code, tests, docs, and manifests, implement the smallest production-safe fail-closed builder/validator/test/doc update so required instrument stems are blocked with explicit reasons until semantic evidence exists. Preserve partial vocal/minus preview flow, short-name artifact policy, no per-track logic, and no Mac Docker usage.
```

## Source-Of-Truth Pass

Code-index was ready for `D:\AIMusicProject`; CodeGraph was used before source edits. Graphify and Beads were not callable in this session, so the fallback was CodeGraph plus Context Mode plus explicit source reads and the active plan bucket.

Production code inspected:

- `ml_worker_improved/tools/build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tools/build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tools/full_pack_stem_contract.py`
- `ml_worker_improved/tools/build_machine_stem_acceptance_gate.py`

Tests inspected:

- `ml_worker_improved/tests/test_build_no_reference_full_stem_pack_manifest.py`
- `ml_worker_improved/tests/test_build_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_validate_aggregate_short_listen_pack.py`
- `ml_worker_improved/tests/test_build_machine_stem_acceptance_gate.py`

Docs/artifacts inspected:

- `docs/production_audio_loop/ITERATION_004.md`
- `artifacts/loop/it044/prompt.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`

## Confirmed Facts

- `artifacts/loop/it044/prompt.md` was stale for the current code: the Mac mount preflight gap had already been closed by `35017a0`.
- `build_no_reference_full_stem_pack_manifest.py` already blocks missing required non-vocal stems, unverified semantic quality, semantic blockers, source safety violations, and source identity mismatch.
- `build_aggregate_short_listen_pack.py` accepted rows using source delivery/presentation/safety/path checks and required-stem coverage, but did not reject a present required instrument stem whose row had `semantic_quality_verified=false`.
- `validate_aggregate_short_listen_pack.py` rechecked aggregate packaging integrity, short paths, copied audio hashes, required-stem coverage, and promised metadata, but did not reject the same required instrument semantic-evidence gap.
- `build_machine_stem_acceptance_gate.py` already blocks final machine acceptance with `required_stem_quality_not_verified:<stem>` for required non-minus instrument stems, so the upstream aggregate builder/validator were weaker than the final gate.

## Implemented

- Added shared semantic-evidence helpers in `ml_worker_improved/tools/full_pack_stem_contract.py`.
- Updated `ml_worker_improved/tools/build_aggregate_short_listen_pack.py` to block accepted aggregate output when present required instrument stems lack machine semantic-quality evidence.
- Updated `ml_worker_improved/tools/validate_aggregate_short_listen_pack.py` to fail validation for legacy/manual aggregate manifests with the same gap.
- Added regression coverage in `ml_worker_improved/tests/test_build_aggregate_short_listen_pack.py`.
- Extended `ml_worker_improved/tests/test_validate_aggregate_short_listen_pack.py`.
- Updated `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`.
- Resolved code-review P1: mixed verified/unverified rows for the same required instrument stem now fail closed per row.

## Validation

Mac Apple container, Windows workspace mounted read-only:

```text
python -m pytest tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py -q
```

Result:

```text
11 passed, 1 warning
```

The warning is pytest cache write failure from the read-only mounted workspace; it does not affect test assertions.

Extended affected-layer validation:

```text
python -m pytest -p no:cacheprovider tests/test_build_aggregate_short_listen_pack.py tests/test_validate_aggregate_short_listen_pack.py tests/test_build_no_reference_full_stem_pack_manifest.py tests/test_build_machine_stem_acceptance_gate.py -q
```

Result:

```text
36 passed
```

## Code Review

Follow-up review result: PASS.

- P1 mixed verified/unverified required-stem row gap: closed.
- P3 validation trace gap: closed by recording both targeted and extended Mac Apple container commands/results.
- Remaining risk: reviewer did not rerun container tests; runtime validation evidence is from the Mac Apple container runs above.

## Next Loop Task

```text
Run strict source-of-truth analysis/funnel for the remaining downstream handoff between aggregate listen packs, semantic review batch, minus quality audit, and machine_stem_acceptance_gate. Verify that no user-facing listen/result path can skip aggregate validation, semantic review evidence for required stems, minus audit evidence, and final machine gate evidence after the new aggregate builder/validator semantic-evidence check. Implement the smallest fail-closed test/code/doc update for any bypass found, preserving diagnostic/listen-preview artifacts as explicitly non-release.
```
