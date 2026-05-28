# Deep Research Report Coverage Audit

Date: 2026-05-25
Source report: `C:/Users/PBAX/Downloads/deep-research-report.md`
Scope: audit whether the current project implements the report requirements for modern vocal/instrumental separation.

## Verdict

Not everything from the report is implemented.

The project has implemented the production audit skeleton around FL Studio ground-truth imports, model registry metadata,
registered candidate execution, reference-driven model-zoo benchmarking, exact-reference waveform audit, and RALPH-AI
delivery locking. The current deep-report vocal benchmark artifact produced no accepted production/listen result:
`accepted_count=0`, `selected_candidate=null`, and all four candidates are `blocked_from_delivery`.

The report requirements that remain partial or unimplemented are the perceptual/human metric layer, complete report-aligned
model coverage, domain fine-tuning on real minus/orig, perceptual or critic refinement, mastering-aware adaptation, and a
general accepted no-reference delivery path for the audited vocal target.

## Status Rules

- `implemented`: production code, tests or docs, and at least one artifact or clear runtime contract confirm the behavior.
- `partial`: code or docs cover part of the requirement, but the report-level behavior is incomplete or explicitly not run.
- `not_implemented`: no project production route was confirmed for the report requirement in the inspected source of truth.
- `blocked`: infrastructure exists, but current artifacts/contracts explicitly reject delivery or prohibit a production claim.

Rejected, skipped, or failed candidates are diagnostic-only. They do not count as a listen/result artifact unless they carry
`delivery_status=accepted`, `delivery_allowed=true`, and `presentation_status=accepted_for_listen_pack`.

## Source Of Truth Inspected

Report:

- `C:/Users/PBAX/Downloads/deep-research-report.md`

Production code:

- `ml_worker_improved/tools/build_flstudio_ground_truth_bundle.py`
- `ml_worker_improved/tools/run_model_zoo_benchmark.py`
- `ml_worker_improved/worker/model_registry.py`
- `ml_worker_improved/tools/run_registered_model_candidate.py`
- `ml_worker_improved/tools/train_teacher_mask_separator.py`

Tests:

- `ml_worker_improved/tests/test_build_flstudio_ground_truth_bundle.py`
- `ml_worker_improved/tests/test_run_model_zoo_benchmark.py`
- `ml_worker_improved/tests/test_model_registry.py`

Docs:

- `docs/STEM_QUALITY_BENCHMARK.md`
- `docs/OPERATIONS.md`
- `docs/ADAPTIVE_VOCAL_ISOLATION.md`
- `docs/FLEXIBLE_MINUS_REPAIR.md`

Artifacts:

- `lab_artifacts/golden_audio/deep_report_flstudio_square_oxygen_gt_20260525/ground_truth_manifest.json`
- `lab_artifacts/golden_audio/deep_report_square_oxygen_vocal_model_zoo_benchmark_4_candidates_20260525/model_zoo_benchmark_manifest.json`

Tooling notes:

- Docker is available: Docker 29.4.2, Compose v5.1.3.
- `bd` / `beads` was not available in the current shell; this document and the Codex task checklist were used as the task-tracker fallback.
- `git` is not available through the current PowerShell `PATH`. A full-path Git binary exists under `C:/Program Files/Git`,
  but it reports that `D:/AIMusicProject` is not a Git repository, so repository-backed `git status` / `git diff` are
  unavailable for this workspace.

## Reconciliation

Confirmed by production code:

- `build_flstudio_ground_truth_bundle.py` writes `ground_truth_contract`, `full_mix_reconstruction_contract`, and
  `report_alignment_contract` sourced from `deep-research-report.md`.
- `run_model_zoo_benchmark.py` writes `metric_stack_contract`, exact-reference audit fields, RALPH-AI delivery contract,
  `delivery_allowed`, and `presentation_status`.
- `model_registry.py` exposes model metadata for task scope, benchmark comparability, code/license/weights status, and
  report-alignment notes.
- `run_registered_model_candidate.py` records Docker intended runtime, availability checks, expected manifest enforcement,
  and no fallback/per-track-hardcode contract.
- `train_teacher_mask_separator.py` contains teacher-derived mask training/application infrastructure, but its own contract
  still marks target-reference use and can block listen-pack presentation.

Confirmed by tests:

- FL Studio bundle tests assert `ground_truth_contract`, report alignment fields, required QC, and accepted/rejected
  full-mix reconstruction behavior.
- Model-zoo benchmark tests assert accepted vs rejected delivery, `blocked_from_delivery`, exact-reference delivery status,
  `no_accepted_candidate`, and metric-stack metadata for ViSQOL/PESQ/human listening.
- Registry tests assert `task_scope`, benchmark score context, license metadata, comparability notes, and report notes for
  relevant model entries.

Confirmed by docs:

- `docs/STEM_QUALITY_BENCHMARK.md` documents FL Studio ground-truth bundles, model-zoo delivery lock, metric stack, model
  comparability/license metadata, and Docker-only model routes.
- `docs/OPERATIONS.md` documents Docker-only execution for quality gates, FL Studio bundles, and model/repair routes.
- `docs/ADAPTIVE_VOCAL_ISOLATION.md` and `docs/FLEXIBLE_MINUS_REPAIR.md` explicitly state that adaptive/proxy routes are
  not fine-tuning and do not claim objective SDR/SI-SDR without verified isolated stems.

Confirmed by artifacts:

- The deep-report ground-truth bundle has `ground_truth_contract.verified=true`, `mode=verified_isolated_stems`,
  `report_alignment_contract.source_document=deep-research-report.md`, and `rights_status=user_owned`.
- The same bundle has `full_mix_reconstruction_contract.status=rejected` and `can_claim_source_mix_1_to_1=false`.
- The deep-report 4-candidate vocal benchmark has `overall.status=no_accepted_candidate`, `accepted_count=0`,
  `rejected_count=4`, `selected_candidate=null`, ViSQOL `not_computed`, PESQ `not_computed`, and human gate `not_run`.

Drift / contradictions:

- The report CLI examples are sketches and do not match the real CLI exactly. The real bundle builder uses positional
  `source_mix_path stem_dir output_dir`; the real model-zoo benchmark uses `--ground-truth-manifest` and repeated
  `--candidate-manifest LABEL=PATH`.
- The inspected ground-truth artifact includes a first source file path for `Chris Bulman & Sarju Jagai - No Escape.flac`
  while the artifact name is a Square Oxygen deep-report bundle. Together with `full_mix_reconstruction_contract.status=rejected`,
  this blocks any claim that the supplied full mix is a clean 1:1 sum of the imported stems.
- The inspected FL bundle was built from a broad source directory: the artifact has `source_file_count=33`, and
  `reference_sum_check` reports `source_vs_sum_correlation=0.716142395`, `sum_minus_source_rms_dbfs=-13.140578`, and
  `sum_minus_source_peak_dbfs=5.048492`. This is valid forensic evidence for a rejected reconstruction, not a clean
  exact-pair dataset. Follow-up implementation on 2026-05-25 added an `input_scope_contract` guard so new builds fail
  fast by default when foreign track-prefixed audio or unknown unprefixed audio is present.

Not confirmed:

- A complete Open-Unmix registered route was not found in the inspected registry.
- ViSQOL, PESQ, and human AB/MUSHRA are not implemented as executed benchmark gates in `run_model_zoo_benchmark.py`.
- Domain fine-tune on FL-derived real minus/orig, perceptual/critic refinement, and mastering-aware adaptation are not
  confirmed as report-aligned production workflows.
- The teacher-mask tooling is not confirmed as a successful report-level teacher-student distillation path for the audited
  vocal target; current docs emphasize diagnostic/teacher evidence and rejected/blocked candidates.

## Coverage Matrix

| ID | Report requirement | Status | Evidence | Gap / next task |
| --- | --- | --- | --- | --- |
| R1 | Model-family landscape: HTDemucs, SCNet, Mel/RoFormer, BS-RoFormer-style routes. | partial | `model_registry.py` has HTDemucs, SCNet, MelBand, Audio Separator BS/MelBand entries; `docs/STEM_QUALITY_BENCHMARK.md` lists registered vocal routes. | Add missing report-family routes only when production assets/licenses/runtime are available; do not treat registry presence as quality acceptance. |
| R2 | Do not compare vocal-only SDR with 4-stem average SDR as one leaderboard. | implemented | `model_registry.py` records `task_scope`, `benchmark_score_context`, `comparability_notes`; `test_model_registry.py` asserts vocal-only vs four-stem metadata. | Keep all future benchmark summaries grouped by task scope and training/data regime. |
| R3 | Training objectives should combine waveform/MR-STFT/SI-SDR/perceptual layers, not a single SDR objective. | not_implemented | No confirmed project route maps the report loss stack into a production fine-tune/refinement workflow; adaptive routes explicitly state they do not train weights. | Design a separate Docker training plan before implementing any fine-tune or perceptual loss route. |
| R4 | Metric stack: SDR, SI-SDR, SIR, SAR, ViSQOL, PESQ secondary, human AB/MUSHRA. | partial | `run_model_zoo_benchmark.py` computes/records separation fidelity and exact-reference audit, but marks ViSQOL/PESQ not computed and human listening not run; tests assert this. | Add separate ViSQOL batch gate and AB/MUSHRA artifact workflow; keep PESQ secondary and vocal-only. |
| R5 | Dataset/license/provenance: code license, weights license, audio rights must be separated. | partial | Registry metadata records `code_license`, `weights_license`, `license_status`, `commercial_use_status`; FL bundle records `rights_status`. | Add an auditable license-clearance checklist before any redistributed model/stem/package claim. |
| R6 | Real minus/orig or FL-derived paired material should drive domain adaptation. | partial | FL bundle builder imports verified isolated stems; docs say reference stems may be used for training/routing diagnostics. | Build an actual domain-adaptation dataset manifest and training route; current import alone is not fine-tune. |
| R7 | Handle phase, sync, mastering, reverberation, sample rate, tails, and full-mix drift. | partial | Bundle `required_qc` includes sample rate, sample-accurate alignment, loudness/peak, tail integrity, full-mix reconstruction; artifact full-mix reconstruction is rejected with correlation `0.716142395`. | Investigate source-file drift before any full-mix reconstruction claim; keep per-stem benchmark separate from 1:1 mix claim. |
| R8 | Baseline benchmark before training across HTDemucs, SCNet, Mel/RoFormer, Open-Unmix-like candidates. | partial | Model-zoo benchmark harness exists; current deep-report artifact ran BS-RoFormer, MelBand ViperX, HTDemucs 6s, and aggressive HTDemucs 6s only. | Register/run SCNet and any approved Open-Unmix route in the same benchmark before training decisions. |
| R9 | Domain fine-tune on real minus/orig. | not_implemented | No confirmed report-aligned fine-tune workflow in inspected project code/docs; docs say adaptive controllers are not fine-tune. | Create Docker-only fine-tune plan with dataset manifest, validation split, losses, GPU requirements, and delivery gates. |
| R10 | Perceptual or critic refinement after best separator. | not_implemented | Benchmark metric contract explicitly does not compute perceptual acceptance; no confirmed refinement training route was found. | Add a separate enhancer/refinement candidate route only after a strong baseline passes objective diagnostics. |
| R11 | Mastering-aware adaptation with limiter/clip/codec/width/reverb augmentation. | not_implemented | `bundle_kind` and `master_chain_status` can label mastered/augmentation bundles, but no mastering-aware training workflow is confirmed. | Add augmentation-stage manifests and train/evaluate them as a distinct experiment, not as hidden preprocessing. |
| R12 | Teacher-student distillation for local/FL workflow. | partial | `train_teacher_mask_separator.py` supports teacher-derived training and checkpoint application; docs record many teacher artifacts as diagnostic or rejected. | Convert teacher evidence into a no-reference route that passes exact-reference gates; blocked/teacher-assisted artifacts cannot be production results. |
| R13 | Human AB/MUSHRA on 8-15 second hard segments. | not_implemented | `metric_stack_contract.human_listening_gate.status=not_run`; bundle notes say AB/MUSHRA is separate and not implied. | Build a human-listening artifact workflow with hard-window selection, exported clips, ratings schema, and non-substitution by objective metrics. |
| R14 | Local FL Studio pipeline: export stems, build bundle, benchmark models, select/ensemble/enhance, AB test, re-import. | partial | Bundle builder, benchmark, registry runner, adaptive routes, and docs exist; AB test, full route coverage, and accepted vocal result are not complete. | Keep this as staged pipeline: bundle -> benchmark -> accepted gate -> optional enhancement -> listening workflow -> re-import evidence. |
| R15 | FL Studio export bundle should capture clean/mastered/hard-case metadata and QC. | implemented | `report_alignment_contract` stores `bundle_kind`, `master_chain_status`, `rights_status`, tags, and required QC; tests assert these fields. | For the current artifact, investigate source-file drift before making full-mix reconstruction claims. |
| R16 | Model-zoo benchmark should rank candidate manifests against verified references. | implemented | `run_model_zoo_benchmark.py` requires ground-truth/candidate manifests, writes ranked candidates, selected candidate, and `overall`; tests cover accepted and no-accepted cases. | Continue adding candidate routes; do not relax gates for current rejected candidates. |
| R17 | Select best model/ensemble only if delivery gate accepts; otherwise iterate. | blocked | Current artifact has no accepted candidate; all candidates are `delivery_status=rejected`, `delivery_allowed=false`, `presentation_status=blocked_from_delivery`. | Run additional registered candidates or build new production-grade route; do not publish best-of-bad. |
| R18 | Legal/ethical limits: code, weights, audio rights, and redistributed stems are separate concerns. | partial | Registry and bundle carry rights metadata, but metadata does not grant rights; docs state rights limitations. | Add explicit release checklist for model weights, datasets, source audio, generated stems, and commercial-use status. |

## Gap-Closure Backlog

1. Create a Docker-only ViSQOL batch gate that consumes verified bundle + candidate manifests and writes a non-delivery
   perceptual metric report.
2. Create a human AB/MUSHRA hard-segment workflow with selected 8-15 second windows, clip export, rating schema, and manifest
   linkage back to candidate/reference hashes.
3. Register and verify any approved Open-Unmix route, or explicitly mark it unsupported with license/runtime rationale.
4. Re-run the deep-report vocal benchmark with SCNet and any newly approved model families before any training claim.
5. Investigate the `deep_report_flstudio_square_oxygen_gt_20260525` source-file drift and decide whether to rebuild the
   bundle from a clean single-project FL export.
6. Done for the default builder path on 2026-05-25: new bundles record `input_scope_contract` and reject foreign
   track-prefixed audio plus unknown unprefixed audio unless `--allow-mixed-stem-dir` is explicitly used for diagnostic
   imports. A source-stem manifest remains a stronger future option for non-standard FL exports.
7. Design a separate domain-adaptation experiment for real minus/orig: data manifest, train/validation split, loss stack,
   Docker GPU route, progress logs, and exact-reference acceptance criteria.
8. Design perceptual/critic refinement only as a new candidate route with manifest provenance and objective + human gates,
   not as a hidden postprocess.
9. Add mastering-aware augmentation only as an explicit `bundle_kind` / `master_chain_status` experiment with separate
   benchmark reporting.
10. Convert teacher-derived evidence into a reference-free production route only after it passes the normal RALPH-AI delivery
   contract.
11. Add a license-release checklist for code, weights, datasets, source audio, generated stems, and commercial use.

## Review Result

Findings:

- severity: high
  file: `ml_worker_improved/tools/build_flstudio_ground_truth_bundle.py`
  issue: FL Studio source import scope can contaminate a bundle when `stem_dir` contains unrelated audio.
  evidence: The inspected deep-report bundle has `source_file_count=33`; its first source file is `Chris Bulman & Sarju Jagai - No Escape.flac`, while the artifact is named for Square Oxygen.
  recommendation: Rebuild exact-pair bundles with the new default `input_scope_contract` guard; reserve `--allow-mixed-stem-dir` for diagnostic mixed-directory imports only.

- severity: high
  file: `lab_artifacts/golden_audio/deep_report_square_oxygen_vocal_model_zoo_benchmark_4_candidates_20260525/model_zoo_benchmark_manifest.json`
  issue: No production/listen result exists for the audited deep-report vocal benchmark.
  evidence: `overall.status=no_accepted_candidate`, `accepted_count=0`, `selected_candidate=null`, and all ranked candidates have `delivery_allowed=false`.
  recommendation: Do not present these candidates as results; run more registered candidates or implement a stronger route.

- severity: high
  file: `ml_worker_improved/tools/run_model_zoo_benchmark.py`
  issue: The report's perceptual and human listening layer is not implemented in the benchmark gate.
  evidence: `metric_stack_contract` marks ViSQOL and PESQ as `not_computed`, and human listening as `not_run`.
  recommendation: Add separate ViSQOL and human-listening workflows instead of treating SDR/SI-SDR acceptance as perceptual acceptance.

- severity: medium
  file: `lab_artifacts/golden_audio/deep_report_flstudio_square_oxygen_gt_20260525/ground_truth_manifest.json`
  issue: The bundle cannot support a 1:1 source-mix reconstruction claim.
  evidence: `full_mix_reconstruction_contract.status=rejected`, `can_claim_source_mix_1_to_1=false`, and the first source file path does not match the artifact name's track family.
  recommendation: Rebuild or quarantine this bundle before any full-mix reconstruction or clean single-project claim.

- severity: medium
  file: `ml_worker_improved/worker/model_registry.py`
  issue: Report-aligned baseline coverage is incomplete.
  evidence: HTDemucs, SCNet, Mel/RoFormer, and Audio Separator BS/MelBand entries are present, but no Open-Unmix registry route was confirmed.
  recommendation: Register an approved Open-Unmix route or document it as unsupported with runtime/license reasons.

PASS:

- Reason: The audit artifact follows the RALPH-AI lock and does not promote rejected candidates.
- Evidence: The coverage matrix classifies current rejected candidates as `blocked`, and the gap backlog requires more model routing or new production-grade routes before delivery.
