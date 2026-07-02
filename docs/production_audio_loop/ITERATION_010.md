# Production Audio Loop - Iteration 010

## Template Use

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Template sha256: `6f92da3743bd7098961447453f191458479a9ea5b29162f70a27533dc3d6c098`
- Materialized prompt: `artifacts/loop/it050/prompt.md`
- Materialized prompt sha256: `3d05de80700a623f3b4572a53357ed09c5e6e210ac0b7739d83808feb1bb6489`

## Task

```text
Run strict source-of-truth analysis/funnel for remaining artifact/listen-pack builders that still emit `delivery_status=accepted`, `delivery_allowed=true`, or `presentation_status=accepted_for_listen_pack` outside the staged aggregate plus accepted machine-gate post-handoff contract. Prioritize no-reference minus/listen-preview routes, dense delivery gates, music-intelligence packs, and stale artifacts referenced by docs. Implement the smallest fail-closed code/test/doc update that preserves diagnostic artifacts while preventing accepted-only delivery bypass.
```

## Source-Of-Truth Pass

Code Index was queried first for accepted/listen-pack status strings,
candidate-wave pack builders, minus comparison, dense delivery gates, and
music-intelligence pack routes. CodeGraph then narrowed the implementation
scope to `build_short_candidate_wave_listen_pack`, `copy_minus_audio`,
`build_minus_cmp`, and `build_minus_cmp_switch_inputs`, and impact checks were
run after reindexing.

Graphify CLI was present, but no project graph existed at
`graphify-out/graph.json` and the global graph was empty. Graphify therefore
provided no evidence for this iteration. Beads CLI was not available as
`beads` or `bd`; task buckets were tracked with `update_plan`.

Production files inspected:

- `ml_worker_improved/tools/build_short_candidate_wave_listen_pack.py`
- `ml_worker_improved/tools/build_minus_cmp.py`
- `ml_worker_improved/tools/build_minus_cmp_switch_inputs.py`
- `scripts/macos-ssh.ps1`
- `scripts/macos-bootstrap-workspace-mount.ps1`

Tests inspected:

- `ml_worker_improved/tests/test_build_short_candidate_wave_listen_pack.py`
- `ml_worker_improved/tests/test_build_minus_cmp.py`
- `ml_worker_improved/tests/test_build_minus_cmp_switch_inputs.py`

Docs inspected:

- `AGENTS.md`
- `ml_worker_improved/docs/NO_REFERENCE_STEM_LISTEN_PREVIEW_README.md`
- `docs/production_audio_loop/ITERATION_009.md`

## Confirmed Facts

- `build_short_candidate_wave_listen_pack.py` copied selected candidate-wave
  minus audio and wrote `delivery_status=accepted`,
  `delivery_allowed=true`, `presentation_status=accepted_for_listen_pack`,
  and `listenable=true` for a packaging-only route whose rows still had
  `semantic_quality_verified=false`.
- Its README already described candidate-wave output as staging/diagnostic
  material rejected by post-gate human-review routes, so code and docs were
  inconsistent.
- `build_minus_cmp.py` accepted `short_candidate_wave_listen_pack_manifest.v1`
  only when the source pack claimed accepted delivery. That made the diagnostic
  same-window comparison path depend on the same stale accepted status being
  removed.
- `build_minus_cmp_switch_inputs.py` had the same accepted-only candidate-wave
  assumption, so the documented comparison-to-segment-switch path would have
  regressed after candidate-wave packs became diagnostic-only.
- The safe separation is: candidate-wave selection may succeed and keep copied
  short-name audio for forensic A/B, but delivery/listen presentation must stay
  blocked until staged aggregate validation and the final machine gate accept a
  real aggregate pack.

## Implemented

- Changed `build_short_candidate_wave_listen_pack.py` so successful selection
  writes:
  - `selection_status=selected_for_diagnostic_ab`;
  - `delivery_status=blocked`;
  - `delivery_allowed=false`;
  - `presentation_status=diagnostic_only_not_for_listen_pack`;
  - `release_status=not_release_ready`;
  - `listenable=false`.
- Preserved selected audio copying, short Windows-safe paths, sha256,
  provenance, blocked candidate rows, and failure cleanup behavior.
- Changed `build_minus_cmp.py` to keep accepted-delivery requirements for real
  repair-selection schemas while requiring diagnostic/blocked semantics for
  candidate-wave packs. Stale candidate-wave packs that still claim accepted
  delivery are rejected before comparison.
- Changed `build_minus_cmp_switch_inputs.py` with the same schema-specific
  delivery contract, so completed comparison reports can still produce
  diagnostic switch inputs from candidate-wave sources without accepting stale
  candidate-wave listen delivery.
- Updated tests for both routes, including regressions that block stale
  accepted candidate-wave packs in both comparison and switch-input builders.
- Updated the no-reference stem/listen README to state the diagnostic
  candidate-wave contract and the minus-comparison source requirements.

## Validation

Mac runtime was validated through explicit SSH:

```text
ssh -i %USERPROFILE%\.ssh\id_ed25519_phone -o IdentitiesOnly=yes andrejviluha@192.168.0.138
```

The initial Mac container run failed because
`/Users/andrejviluha/AIMusicProject_win` was an empty/non-mounted path. The
workspace mount was restored with the project bootstrap script:

```text
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-workspace-mount.ps1 -MacHost 192.168.0.138 -MacUser andrejviluha -SshKey C:\Users\PBAX\.ssh\id_ed25519_phone -WindowsProjectRoot D:\AIMusicProject -WindowsHostIp 192.168.0.101 -MacMountPath /Users/andrejviluha/AIMusicProject_win -ForceCredentialReset
```

Bootstrap evidence:

```text
mount_ok=/Users/andrejviluha/AIMusicProject_win
probe=write_read_delete_passed
windows_share=\\192.168.0.101\AIMusicProject
container CLI version 0.12.0
```

Targeted Mac Apple-container regression:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_candidate_wave_listen_pack.py tests/test_build_minus_cmp.py tests/test_build_minus_cmp_switch_inputs.py -q
```

Result:

```text
16 passed
```

Wider related regression:

```text
python -m pytest -p no:cacheprovider tests/test_build_short_candidate_wave_listen_pack.py tests/test_build_minus_cmp.py tests/test_build_minus_cmp_switch_inputs.py tests/test_build_short_listen_feedback_template.py tests/test_build_short_listen_review_handoff.py tests/test_validate_short_listen_review_handoff.py tests/test_run_short_listen_review_packet.py tests/test_build_short_listen_feedback_manifest.py tests/test_run_short_listen_feedback_cycle.py -q
```

Result:

```text
65 passed
```

Code Index was refreshed after edits:

```text
code-index-mcp index .
```

## Next Loop Task

```text
Run strict source-of-truth analysis/funnel for the next remaining accepted/listen delivery surface after candidate-wave diagnostics are blocked. Prioritize no-reference minus listen-preview, dense delivery gate, and stale docs/artifacts that can still imply `accepted_for_listen_pack` without staged aggregate validation plus accepted machine-gate evidence. Implement the smallest fail-closed production/test/doc update.
```
