# Production Audio Loop Iteration 001

## Template Use

- Source template: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edit status: unchanged
- Placeholder count observed: 1
- Replaced task text for this iteration:

```text
Audit the current AIMusicProject production audio separation/training/problem-discovery architecture from source-of-truth code, tests, docs, and artifacts. Start from the current vocal/minus leakage and artifact provenance problems, especially Mac-generated artifacts mirrored into the Windows workspace. Identify architecture changes that make training, candidate routing, failure taxonomy, and accepted-only delivery faster and more reliable without per-track hardcodes. Rank ideas strictly by verifiability and production impact, implement the top safe improvement, then re-run validation and record the next loop tasks.
```

## Source-Of-Truth Pass

Inspected evidence:

- `prompt-for-loop.txt`: one `{НАПИШИ_ЗАДАЧУ_ЗДЕСЬ}` placeholder, template left unchanged.
- `artifacts/mac_api/altm002/bci/stem_manifest.json`: raw separator manifest has no `delivery_status`, `delivery_allowed`, or `presentation_status`; audio paths point to `/Users/andrejviluha/aimusic_local_artifacts/...` even though mirrored files exist under the Windows project.
- `ml_worker_improved/tools/build_stem_manifest_audio_pack.py`: copies stems into short listen-pack paths, but previously required exact source paths or per-stem override paths.
- `ml_worker_improved/tools/build_no_reference_stem_listen_preview_manifest.py`: stem preview gating used the manifest path directly for audio integrity checks.
- `ml_worker_improved/tools/build_generic_finetuning_corpus_from_lab_artifacts.py`: `blocked_like`, `accepted_contract`, and `deep_failure_labels` define training corpus routing; blocked artifacts are hard-eval rows, accepted artifacts require accepted contract plus human/reference verification.

Confirmed current problems:

- Raw Mac manifests can contain Mac-local paths after files are mirrored into `D:\AIMusicProject`, creating `audio_path_missing` false negatives or forcing manual per-stem overrides.
- Raw `stem_manifest.json` is not itself a delivery artifact; downstream builders must produce accepted/blocked listen manifests with explicit gates.
- Failure taxonomy exists in manifests at scale, with common labels such as `mix_conservation_failed`, `foreign_source_recovered:*`, `candidate_delivery_blocked`, and `separator_server_unavailable`.
- The previous Mac/Windows mount fix made future artifacts land in the Windows workspace, but older or externally generated Mac manifests still need provenance-safe path resolution.

Unconfirmed:

- No new semantic audio acceptance was claimed for `altm002/bci`.
- No final production vocal/minus result was accepted in this iteration.
- Exact vocal leakage regions were not re-scored in this iteration; this iteration addressed architecture/provenance friction first.

## Raw Ideas

| # | Idea | Status | Reason |
|---|---|---|---|
| 1 | Add reusable source path prefix mapping for Mac-local manifests mirrored into Windows artifacts | KEEP | Directly fixes current artifact provenance drift and reduces manual overrides |
| 2 | Add deep failure-label hard-negative coverage tests for generic finetuning corpus | KEEP | Training quality depends on rejected/leaky artifacts becoming hard eval, not positives |
| 3 | Add vocal/minus region-level semantic gate before any accepted listen pack | KEEP | User-reported leakage is region-level, not one timestamp |
| 4 | Add accepted-only listen pack validator for raw stem manifests | KEEP | Prevents raw diagnostic manifests from being treated as results |
| 5 | Add failure taxonomy summary command over current artifacts | KEEP | Speeds strict analysis after every loop |
| 6 | Add Mac artifact path provenance audit over `artifacts/mac_api` | KEEP | Detects stale Mac-local paths in manifests |
| 7 | Add no-reference pair gate requiring both vocal and minus manifests accepted | KEEP | Reduces mismatched pair delivery |
| 8 | Add per-candidate model family scoreboard for separator routes | KEEP | Helps choose another model for minus without manual guessing |
| 9 | Add cached DSP feature extraction for repeated leak scans | KEEP | Reduces iteration time |
| 10 | Add structured loop iteration runner that writes prompt/result manifests | KEEP | Makes the loop repeatable |
| 11 | Hardcode `altm002/bci` as accepted | REJECT | Violates accepted-only and no per-track hardcode |
| 12 | Rename old artifact files in place | REJECT | Breaks provenance and violates no migration without explicit command |
| 13 | Lower leak thresholds to pass current audio | REJECT | Masks the failure instead of improving detection |
| 14 | Use host Python for audio scoring | REJECT | Project runtime must use approved container route |
| 15 | Move artifacts back to Mac-local scratch | REJECT | Violates Windows workspace source-of-truth artifact contract |
| 16 | Train on all accepted-like manifests regardless of reference or human verification | REJECT | Risks training on contaminated candidates |
| 17 | Add a new model route without manifests/tests | REJECT | Unverifiable and increases routing entropy |
| 18 | Build a UI-only fix for file name confusion | DEFER | Useful later, but root issue is provenance and manifest mapping |
| 19 | Add RAG cleanup for bulky old artifacts | DEFER | Important but not the current quality bottleneck |
| 20 | Add full benchmark over every artifact now | DEFER | Too broad before path/provenance blockers are fixed |

## Serious Ideas And Scores

| Idea | Impact | Confidence | Effort | Verification | Score |
|---|---:|---:|---:|---:|---:|
| Source path prefix mapping for mirrored Mac artifacts | 5 | 5 | 2 | 5 | 13 |
| Deep failure-label hard-negative tests | 5 | 4 | 3 | 5 | 11 |
| Region-level vocal/minus semantic gate | 5 | 3 | 4 | 5 | 9 |
| Failure taxonomy artifact summary command | 4 | 5 | 2 | 5 | 12 |
| Accepted-only raw manifest validator | 4 | 4 | 2 | 5 | 11 |
| Pair gate requiring accepted vocal and minus manifests | 4 | 4 | 3 | 5 | 10 |
| Cached DSP feature extraction | 4 | 3 | 4 | 4 | 7 |
| Loop iteration runner with prompt/result manifests | 3 | 5 | 3 | 4 | 9 |

Score heuristic: `impact + confidence + verification - effort`.

## TOP-3

1. Source path prefix mapping for mirrored Mac artifacts.
   - Implemented in this iteration.
   - Production value: old and remote Mac manifests can be consumed through a verified Windows/mounted mirror without per-stem overrides.
   - Verification: unit tests cover blocked-before-map and accepted-after-map behavior.

2. Deep failure-label hard-negative corpus coverage.
   - Next implementation candidate.
   - Production value: leakage and conservation failure labels should reliably feed hard eval/negative training surfaces.

3. Region-level vocal/minus semantic delivery gate.
   - Next quality gate candidate after corpus/failure taxonomy coverage.
   - Production value: prevents score-only improvements that leave audible region leaks.

## Implemented In This Set

Implemented TOP-1:

- Added shared `tools/path_prefix_mapping.py`.
- Added `--path-prefix-map FROM=TO` to `build_stem_manifest_audio_pack.py`.
- Added `--path-prefix-map FROM=TO` to `build_no_reference_stem_listen_preview_manifest.py`.
- Manifest provenance now records:
  - original source path;
  - resolved local/mounted mirror path;
  - mapping source and target;
  - whether an explicit override was applied.
- Updated stem preview docs with the new Mac-local-to-mirror path mapping flow.

Validation:

- Apple container route:
  - `tests/test_build_no_reference_stem_listen_preview_manifest.py`
  - `tests/test_build_stem_manifest_audio_pack.py`
- Result: `16 passed`.

## Next Loop Task

Use the same template again with this task text:

```text
Audit whether failure taxonomy labels from blocked vocal/minus artifacts, including foreign_source_recovered, mix_conservation_failed, vocal leakage, and candidate_delivery_blocked, reliably become hard-eval or negative training examples instead of training positives. Implement the smallest production fix with tests so leaky or blocked candidates improve future training/problem-discovery rather than silently disappearing from the learning loop.
```
