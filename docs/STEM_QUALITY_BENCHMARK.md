# Stem Quality Benchmark Harness

`ml_worker_improved/tools/run_stem_quality_benchmark.py` creates deterministic synthetic fixtures for
project-specific stem quality regressions. It is a regression benchmark, not a replacement for human
listening on real tracks.

## Contract

- fixtures are synthetic and deterministic;
- `verified_synthetic_truth=true` only applies to generated synthetic stems;
- `fallback_used=false`;
- no SDR/SI-SDR claim is made for real tracks without isolated ground truth;
- clean fixtures protect against P1 false positives;
- known-bad fixtures must produce expected defect labels;
- unexpected P1 labels fail the benchmark unless the fixture explicitly allows a secondary label;
- the published suite is currently `smoke`; larger suites must add fixtures before exposing a new CLI value.

## Covered Defects

- `drum_tonal_leakage`
- `harmonic_leakage_in_drums`
- `vocal_dropout`
- `vocal_tail_loss`
- `vocal_harshness`
- `vocal_residue_in_minus`
- `possible_other_proxy_leakage`
- `proxy_copy_risk`
- `insufficient_presence`
- `near_silent_output`
- `source_conservation_error`

## Docker Usage

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/run_stem_quality_benchmark.py `
  /app/lab_artifacts/quality_benchmark/smoke `
  --suite smoke `
  --fail-on-regression
```

The tool writes:

- `benchmark_manifest.json`;
- `fixtures/<fixture>/source.flac`;
- `fixtures/<fixture>/stems/*.flac`.

## FL Studio Ground Truth Bundles

`ml_worker_improved/tools/build_flstudio_ground_truth_bundle.py` imports a user-supplied full mix plus FL Studio
split mixer tracks into a verified isolated-stem manifest that can be evaluated by
`ml_worker_improved/scripts/run_ground_truth_quality_gate.py`.

Contract:

- input stems must be lossless WAV/FLAC/AIFF exports from the same project timeline;
- the importer enforces an input-scope guard before audio is copied or summed. Files with an FL export track prefix that
  does not overlap the source mix identity are rejected by default and recorded through `input_scope_contract` on
  successful runs. Unprefixed files are allowed only when their filename is semantically recognized as a stem label such
  as `vocal.wav` or `kick.wav`; unknown unprefixed audio is rejected by default;
- files are grouped into canonical stems by semantic mixer-track labels, not by track-specific filenames;
- the importer writes `source_mix.flac`, `reference_sum_mix.flac`, `stems/*.flac`, and `ground_truth_manifest.json`;
- `source_mix.flac` is the authoritative full mix input for separation experiments;
- `stems/*.flac` are individual reference targets used to compare what the system extracts from `source_mix.flac`;
- `reference_sum_mix.flac` is only a diagnostic sum of imported stems for drift/provenance checks, not an
  authoritative mix and not a target for balancing the song;
- `reference_sum_check` must be inspected before claiming the supplied full mix is a verified sum of the stems, but
  ordinary per-stem benchmarks do not require that claim;
- `full_mix_reconstruction_contract.status` is `accepted|rejected`; only `accepted` permits a 1:1 claim that
  `source_mix.flac` equals the imported stem sum. `ground_truth_contract.verified=true` by itself verifies isolated
  stems, not full-mix reconstruction;
- `instrumental` is derived from non-vocal stems and must not be evaluated together with its child stems in the same SIR/SAR report;
- `drums_plus_percussive_fx_aux` is an optional derived composite target. It keeps strict `drums` unchanged and adds
  FX-assigned source tracks whose mixer labels are both percussion-like and FX-like, such as `clap riser.wav`;
- composite targets are recorded in `target_definitions` with `child_stems`, `aux_source_files`, and `selection_rule`.
  `ml_worker_improved/scripts/run_ground_truth_quality_gate.py` treats manifest-defined derived targets as composite
  references and rejects objective reports that request a composite target together with its child stems.
- `report_alignment_contract` records the research-report validation bucket for the bundle:
  `bundle_kind=clean|mastered|hard_case|diagnostic`, `master_chain_status`, `rights_status`, optional license notes,
  validation-pack tags, hard-case tags, and the required QC checklist (`sample_rate_consistency`,
  `sample_accurate_alignment`, `loudness_and_peak_summary`, `tail_integrity_summary`,
  `full_mix_reconstruction_contract`). These fields are provenance/validation metadata only and do not grant audio or
  checkpoint redistribution rights.

Example:

```powershell
docker compose run --rm --no-deps -T `
  ml-worker python /app/tools/build_flstudio_ground_truth_bundle.py `
  "/app/testing_flstudio_music/Square a Saw - Oxygen (Tevlo Vocals Resynthesized).flac" `
  "/app/testing_flstudio_music" `
  "/app/lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_20260511" `
  --track-name "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)" `
  --bundle-kind clean `
  --master-chain-status disabled_clean_export `
  --rights-status user_owned
```

Add `--require-full-mix-reconstruction` when the workflow must fail fast unless the supplied full mix reconstructs
from the imported stems. This is a diagnostic integrity mode for rare 1:1 source-mix claims. It must not be used to
rebalance or replace the authoritative full mix, and it is not required for ordinary per-stem objective benchmarks.

Use `--allow-mixed-stem-dir` only for legacy or forensic imports from a mixed directory. In that mode the builder still
records suspicious foreign-prefix or unknown unprefixed files in
`input_scope_contract.status=accepted_with_warnings`; such bundles must not be used for clean exact-pair claims without
additional audit. The default path is strict and fails before writing `ground_truth_manifest.json` when unrelated scoped
audio is present.

`docker-compose.yml` mounts `./testing_flstudio_music` into `ml-worker` as `/app/testing_flstudio_music:ro`. Do not run
FL Studio reference import or exact-reference audits from the host Python environment, and do not use ad-hoc `-v` mounts
as the only way to make user-supplied reference exports visible to project runtime commands.

## Exact-reference listen audit

`ml_worker_improved/tools/write_exact_reference_listen_audit.py` is the delivery guard for a specific listen candidate
when an explicit FL Studio reference export exists. It does not create or repair audio. It writes:

- `exact_reference_listen_audit.json`;
- `forensic_evidence/<stem>_waveform_forensic_report.json`;
- `forensic_evidence/<stem>_waveform_trace.npz`;
- aligned residual audio and compact SVG evidence.

Contract:

- compare the actual candidate FLAC against the exact reference PCM, not against an inferred summary score;
- keep `source_mix.flac` as the only authoritative separation input;
- reject delivery when SDR, waveform cosine, residual peak, or waveform-forensic status fail the configured thresholds;
- do not present rejected listen candidates as production stems; they are diagnostic artifacts until objective gate and
  exact-reference audit agree;
- reference stems may be used for audit, training, routing diagnostics, and teacher-profile validation, but must not be
  copied into a no-reference production result.

`ml_worker_improved/tools/run_model_zoo_benchmark.py` now enforces this guard for every evaluated candidate stem. The
benchmark manifest records `exact_reference_listen_audit_enforced=true`; each ranked candidate gets
`exact_reference_audit_status`, `exact_reference_delivery_status`, `exact_reference_audit_dir`, and per-stem audit
reports. A candidate whose exact-reference audit fails is kept as `blocked_from_delivery` even if another score would
have ranked it highly.

## RALPH-AI audio delivery contract

`run_model_zoo_benchmark.py` enforces a no-best-of-bad delivery rule for audio candidates:

- Reference: verified reference stems define the objective target when they are available.
- Audit: objective metrics and exact-reference waveform audit must both pass.
- Lock: failed candidates are marked with a blocking presentation status and must not be presented as a listen/result
  artifact. Model-zoo benchmark manifests use `presentation_status=blocked_from_delivery`; adaptive listen-pack manifests
  use `presentation_status=blocked_from_listen_pack`.
- Produce: only candidates with `delivery_status=accepted`, `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack` may be promoted.
- Halt/iterate: when no candidate passes, `next_action` must continue repair/model generation or reject the candidate
  space with an explicit reason. It must not pick the highest-scoring rejected candidate as the result.

The benchmark manifest records:

- `ralph_ai_delivery_contract_enforced=true`;
- `best_of_bad_candidate_delivery_forbidden=true`;
- `rejected_candidates_are_diagnostic_only=true`.
- `metric_stack_contract`, sourced from `deep-research-report.md`. The objective gate computes separation-fidelity
  metrics (`sdr_db`, `si_sdr_db`, `sir_db`, `sar_db`) plus exact-reference waveform audit evidence. It explicitly marks
  ViSQOL, PESQ, and human AB/MUSHRA as not computed by this gate, so objective acceptance cannot be misreported as
  perceptual/listening acceptance.

Each ranked candidate records `ralph_ai_delivery_contract`. For rejected candidates,
`ralph_ai_delivery_contract.status=blocked` and `result_delivery_allowed=false`.

## Model-zoo comparability and license metadata

`ml_worker_improved/worker/model_registry.py` exposes report-aligned metadata in `model_registry.entries[]` and candidate
suggestions:

- `task_scope` separates `four_stem` routes from `vocal_instrumental` routes;
- `benchmark_score_context.metric_family` records whether a reported number is four-stem average SDR, vocal-only SDR,
  or checkpoint-local provenance metadata;
- `benchmark_score_context.directly_comparable_with` prevents treating vocal-only Mel/RoFormer scores as directly
  comparable with HTDemucs/SCNet four-stem averages;
- `code_license`, `weights_license`, `license_status`, and `commercial_use_status` record the code/checkpoint rights
  review state;
- `comparability_notes` and `report_alignment_notes` preserve the report warning that paper scores, extra-data results,
  community checkpoint labels, and local domain benchmarks are not interchangeable.

These fields are audit metadata. They do not change ranking thresholds and do not make an unavailable or unlicensed
checkpoint deliverable.

`run_ground_truth_quality_gate.py` also enforces the candidate manifest delivery contract. A candidate can pass the
waveform/objective thresholds and still be rejected for delivery when its own manifest says it is teacher-assisted,
holdout-missing, timestamp/mask-prior based, or otherwise not a no-reference production route. The report records this as:

- `candidate_manifest_delivery_contract.status=rejected`;
- `delivery.reason=candidate_manifest_delivery_blocked`;
- `overall.contract_failed_stems=["candidate_manifest"]`.

This prevents reference-locked or teacher-distilled diagnostic artifacts from being promoted only because their objective
metrics are high against the same training/reference bundle.

## Complex oracle mask teacher

`ml_worker_improved/tools/build_complex_oracle_mask_teacher.py` is a reference-assisted teacher/upper-bound route for
cases where a no-reference proxy route keeps choosing contaminated material. It builds a complex STFT mask from the
verified target stem and applies that mask to the authoritative `source_mix.flac`.

Contract:

- source input: authoritative `source_mix.flac`;
- target reference stems are read at application time to build the complex mask;
- output audio is rendered from `source_mix.flac * complex_mask`, not copied as a target FLAC file;
- `contract.target_reference_assisted=true`;
- `contract.target_reference_used_at_application=true`;
- `contract.application_reference_free=false`;
- `contract.delivery_allowed_without_reference=false`;
- `contract.no_reference_production_claim=false`;
- `presentation_status=diagnostic_requires_exact_reference_gate`;
- the artifact must be rechecked with `run_model_zoo_benchmark.py` before it can be used as a teacher/listen target.

This route is not a production no-reference separator. Its purpose is to prove the attainable 1:1 target from the
current full mix/reference bundle and to create teacher evidence for the next production model/profile route.

Square Oxygen complex-mask teacher checkpoint on 2026-05-18:

- candidate artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_vocal_melody_20260518`;
- RALPH-AI gate artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_vocal_melody_gate_20260518`;
- accepted listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_vocal_melody_accepted_listen_20260518`;
- status: `accepted_candidate_found`;
- `presentation_status=accepted_for_listen_pack`;
- mean reference likeness `99.999998`;
- vocals exact SDR `153.929156`, cosine `100.0`, residual RMS `-119.999994 dBFS`;
- melody exact SDR `150.463798`, cosine `100.0`, residual RMS `-119.999962 dBFS`.

Square Oxygen `strings_orchestra` complex-mask teacher checkpoint on 2026-05-18:

- candidate artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_strings_orchestra_20260518`;
- RALPH-AI gate artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_strings_orchestra_gate_20260518`;
- accepted listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_strings_orchestra_accepted_listen_20260518`;
- status: `accepted_candidate_found`;
- `presentation_status=accepted_for_listen_pack`;
- reference likeness `100.0`;
- exact SDR `145.654504`, cosine `100.0`, residual RMS `-120.0 dBFS`.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/write_exact_reference_listen_audit.py `
  --reference "/app/testing_flstudio_music/Square a Saw - Oxygen (Tevlo Vocals Resynthesized) - Tevlo Vocals Bounce.wav" `
  --candidate "/app/lab_artifacts/golden_audio/<listen-pack>/01_joint_vocals.flac" `
  --output-dir "/app/lab_artifacts/golden_audio/<listen-pack>_exact_vocals_audit" `
  --stem-name vocals
```

When a verified reference stem exists, `ml_worker_improved/tools/route_drum_model_zoo_segments.py` can route
DRUM candidates with `--target-ref`. This is a supervised reference-aware mode for FL Studio benchmark tracks:
it must be reported as target-reference assisted and must not be claimed as a no-reference production result.
Use it to choose between model candidates, identify leakage, and build measurable improvement cases.

## Oracle stem upper-bound harness

`ml_worker_improved/tools/oracle_stem_upper_bound.py` builds a reference-assisted diagnostic upper bound for one
stem. It applies an ideal ratio mask derived from the verified target stem and optional non-target reference stems to
the authoritative `source_mix.flac`, writes `01_ideal_ratio_mask_oracle.flac`, and records an
`oracle_upper_bound_report.json`.

Contract:

- this is diagnostic/teacher evidence only;
- `contract.target_reference_assisted=true`;
- `contract.application_reference_free=false`;
- `contract.delivery_allowed_without_reference=false`;
- `contract.no_reference_production_claim=false`;
- `contract.per_track_hardcode_allowed=false`;
- the output must not be shipped as a production no-reference stem;
- a passing oracle proves that a stronger model/router/mask can theoretically reach the reference target;
- a failing oracle indicates reference/mix/phase/model-stack limits that must be resolved before more repair tweaks.

The harness also reports `bar_spanning_drop_melody_leakage` when a candidate DRUM stem contains sustained non-drum
harmonic content over long beat/bar/phrase windows. This class is intentionally not the same as kick-synchronous tail
leakage: it is for drop melody that remains audible between kicks across a musical phrase.
When the residual is strongly harmonic but cannot be attributed to a named non-target reference stem, the source is
reported as `unattributed_harmonic_residual`; this is still a blocking diagnostic signal, not permission to suppress a
track-specific timestamp by hand.
For reference-assisted DRUM debugging, `ml_worker_improved/tools/drum_audit_guided_residual_debleed.py` can consume this
label as `phrase_reference_projection`. The phrase route is source-of-truth assisted, not a no-reference production
claim: it requires a verified target DRUM stem plus named non-target reference stems, can be restricted to generic repair
reasons/sources, and can use multi-source least-squares projection when the residual is jointly explained by multiple
drop/harmonic sources such as melody and bass. It must still be re-gated after rendering.
When debleed over an already contaminated DRUM candidate produces an inaudibly small delta, use
`ml_worker_improved/tools/reference_guided_drum_reconstructor.py` as the teacher/oracle route instead. It solves reference
projection gains from the full mix and verified source-of-truth stems, then emits a reference-locked DRUM teacher artifact
with `target_reference_assisted=true`, `application_reference_free=false`, and `no_reference_production_claim=false`.
This is the correct upper-bound/training artifact for proving what the clean DRUM should sound like; it must not be
presented as a no-reference extraction result.

When the teacher proves that the requested DRUM is available but a single no-reference candidate still contains
drop/melody residue, `ml_worker_improved/tools/reference_guided_drum_candidate_router.py` can build a supervised
segment-router over already generated DRUM candidates. It does not copy the target reference and does not synthesize a
new stem; it selects fixed-size segments from real model outputs using verified DRUM reference error. Its contract must
remain explicit: `target_reference_assisted=true`, `application_reference_free=false`,
`delivery_allowed_without_reference=false`, and `no_reference_production_claim=false`.

Square Oxygen reference-guided candidate router on 2026-05-16:

- artifact root: `lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_candidate_router_v3_nofade_20260516`;
- listen root: `lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_candidate_router_v3_nofade_listen_20260516`;
- selected candidate counts: `source_residual_plus_proxy=192`, `scnet_tonal_default=238`, `scnet_xl=125`,
  `tonal_stronger=317`, `htdemucs_ft=167`;
- full-track SDR improved over the previous no-reference best from `14.871205` to `15.677729`;
- user problem windows improved: `56.5-58.0` from `14.721752` to `15.273595`, `69.5-71.0` from `13.322340` to
  `13.991718`, `76.5-78.0` from `14.479154` to `14.793431`, and `82.0-105.0` from `14.346562` to `15.268690`;
- crossfade was explicitly disabled for the accepted diagnostic artifact because model-output blending degraded the
  objective reference check.

`ml_worker_improved/tools/teacher_dictionary_drum_separator.py` is a separate teacher-derived profile-bank experiment:
training is reference-assisted and application is reference-free, but the first Square Oxygen pass was objectively worse
than the previous best candidate. Keep it as rejected infrastructure until a stronger dictionary/NMF design passes the
same reference gate.

Square Oxygen follow-up on 2026-05-16:

- user listening rejected `flstudio_square_oxygen_reference_guided_candidate_router_v3_nofade_listen_20260516` as
  audibly unchanged;
- simple Wiener masks and nonnegative dictionary inference were tested as the next NMF-like direction and remained below
  the previous best candidate, so they must not be promoted as listen results;
- a supervised teacher-residual diagnostic was generated instead:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_supervised_teacher_residual_drums_20260516`;
- listen root:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_supervised_teacher_residual_drums_listen_20260516`;
- primary file:
  `01_supervised_teacher_residual_alpha090_drums.flac`;
- contract: `target_reference_assisted=true`, `application_reference_free=false`,
  `delivery_allowed_without_reference=false`, `no_reference_production_claim=false`;
- objective delta against the previous best candidate: full-track SDR `14.871205 -> 34.871205`, `56.5-58.0`
  `14.721752 -> 34.721752`, `69.5-71.0` `13.322340 -> 33.322340`, `76.5-78.0`
  `14.479154 -> 34.479154`, and `82.0-105.0` `14.346562 -> 34.346562`.
  This proves the audible target when the verified DRUM reference is allowed, but it is not a no-reference separator.

Square Oxygen micro-clean follow-up on 2026-05-16:

- user listening accepted the `alpha090` direction as materially cleaner, with only very low remaining melody residue
  around `87.0-105.0` and `190.0-220.0`;
- two global residual-blend variants were generated without timestamp-specific rules:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_supervised_teacher_residual_drums_microclean_20260516`;
- listen root:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_supervised_teacher_residual_drums_microclean_listen_20260516`;
- listen files:
  `01_supervised_teacher_residual_alpha095_drums.flac` and
  `01_supervised_teacher_residual_alpha097_drums.flac`;
- contract: `target_reference_assisted=true`, `target_reference_used_at_application=true`,
  `application_reference_free=false`, `delivery_allowed_without_reference=false`,
  `no_reference_production_claim=false`, `timestamp_specific_rules=false`;
- `alpha095` leaves 5% of the previous residual and improves full-track SDR `14.871205 -> 40.857774`;
- `alpha097` leaves 3% of the previous residual and improves full-track SDR `14.871205 -> 45.083678`;
- on the user-reported windows, `alpha097` improves `87.0-105.0` SDR `14.393960 -> 44.604720`
  and `190.0-220.0` SDR `14.373861 -> 44.569650`.

Square Oxygen inter-kick melody residue follow-up on 2026-05-16:

- user listening still detected very quiet melody between kicks in `alpha097`;
- forensic check showed the remaining `alpha097` residual is still measurable in the reported windows, while the verified
  `drums.flac` reference has near-zero projection to `melody.flac`;
- a strict reference-locked teacher was generated:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_interkick_clean_reference_locked_teacher_20260516`;
- listen root:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_interkick_clean_reference_locked_teacher_listen_20260516`;
- primary listen file:
  `02_reference_locked_teacher_drums.flac`;
- contract remains teacher/oracle only: `target_reference_assisted=true`,
  `target_reference_used_at_application=true`, `application_reference_free=false`,
  `delivery_allowed_without_reference=false`, `no_reference_production_claim=false`;
- `interkick_melody_residual_audit.json` reports exact-reference residual `0.0` max abs error against the verified
  `drums.flac` target, with reference-vs-melody explained energy below `0.001%` in the user-reported windows.

Square Oxygen DRUM completeness audit on 2026-05-16:

- after the strict teacher removed melody leakage, user listening reported a quiet clap layer;
- source-of-truth manifest confirms `clap.wav`, `pre clap.wav`, `snare.wav`, hats, kick, cymbal, shaker, toms,
  and reverb kick are already assigned to `drums`;
- the only explicit clap-like source outside `drums` is `clap riser.wav`, assigned to `fx`;
- diagnostic artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_completeness_audit_20260516`;
- listen root:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_completeness_audit_listen_20260516`;
- listen files include strict reference drums, isolated `clap_riser_only_fx_aux`, and
  `strict_drums_plus_clap_riser_aux` variants;
- contract is now represented as a separate reference target definition, not as a separator repair:
  `drums_plus_percussive_fx_aux = drums + percussive_fx_aux`;
- generated bundle:
  `lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_drums_plus_percussive_fx_aux_20260516`;
- listen root:
  `lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_drums_plus_percussive_fx_aux_listen_20260516`;
- user-preferred listening target:
  `02_drums_plus_percussive_fx_aux.flac`.

## Reference teacher stem pack

`ml_worker_improved/tools/build_reference_teacher_stem_pack.py` copies verified FL Studio reference stems into a
benchmark-compatible full-stem teacher pack. It is used when the goal is to compare model output against the exact
user-supplied isolated stems or to create a compact listen pack with every verified target.

Contract:

- the command is Docker-only;
- input is a verified `ground_truth_manifest.json`;
- by default only atomic stems are copied; derived stems such as `instrumental` require `--include-derived`;
- `instrumental` must be benchmarked separately from its child stems to avoid double-counting in SIR/SAR reports;
- the objective quality gate rejects one report that mixes a derived `instrumental` target with child targets such as
  `drums`, `bass`, `piano`, `strings_orchestra`, `melody`, `fx`, or `other`;
- output `stem_manifest.json` is benchmark-compatible;
- `contract.target_reference_assisted=true`;
- `contract.delivery_allowed_without_reference=false`;
- `contract.reference_teacher_pack=true`;
- `contract.registered_model_candidate=false`;
- `contract.fallback_used=false`;
- `contract.per_track_hardcode_allowed=false`;
- `reference_assistance.not_a_no_reference_production_route=true`.

The teacher pack can legitimately score near 100% against the same verified bundle because it is a copied reference
target, not an inference result. Use it as a target, regression oracle, listen reference, or supervised-learning input.
Do not present it as proof that the no-reference separator produced the same stems.

## Reference-guided stem residual repair

`ml_worker_improved/tools/build_reference_guided_stem_residual_repair.py` builds a supervised residual-repair candidate
for one requested stem. It consumes a verified ground-truth manifest and a generated candidate manifest, then applies a
bounded residual correction toward the verified reference stem. This is a learning/benchmark artifact for measuring how
much of the remaining error is repairable from the current candidate; it is not a no-reference separator.

Contract:

- the command is Docker-only;
- inputs are a verified `ground_truth_manifest.json`, one generated candidate `stem_manifest.json`, and `--stem`;
- output `stem_manifest.json` is benchmark-compatible and must be rechecked by `run_model_zoo_benchmark.py`;
- `contract.target_reference_assisted=true`;
- `contract.delivery_allowed_without_reference=false`;
- `contract.registered_model_candidate=false`;
- `contract.fallback_used=false`;
- `contract.per_track_hardcode_allowed=false`;
- `reference_assistance.not_a_no_reference_production_route=true`.

Use this route to create an accepted supervised target candidate and a measurable delta from the generated baseline.
Only promote the learned behavior into production after converting the residual policy into reference-free rules and
validating it on held-out tracks.

## Stem delta profile repair

`ml_worker_improved/tools/build_stem_delta_profile_repair.py` learns a diagnostic band profile from a generated
candidate, a reference-guided teacher repair, and the verified ground-truth stem. The learned profile can then be
applied to the generated candidate without reading the target reference during the application step.

Contract:

- profile learning is reference-assisted and must be treated as an offline diagnostic/learning artifact;
- application is marked with `contract.application_reference_free=true` and
  `contract.target_reference_used_at_application=false`;
- the artifact remains `contract.delivery_allowed_without_reference=false` because the profile source is supervised;
- `contract.fallback_used=false`;
- generated profiles must not contain timestamps or track-name branches;
- a profile-repaired candidate must be rechecked by `run_model_zoo_benchmark.py` before any listen-pack promotion.

## Teacher profile bank

`ml_worker_improved/tools/build_teacher_profile_bank.py` turns one or more verified teacher/reference training examples
into a reusable stem profile bank, then applies that profile to a candidate without reading the target reference during
application. This is the first step toward teacher-derived no-reference routing, but it is not a delivery shortcut.

Contract:

- profile-bank training is reference-assisted and records `profile_source_reference_assisted=true`;
- application is reference-free and records `target_reference_assisted=false`,
  `target_reference_used_at_application=false`, and `application_reference_free=true`;
- a profile bank with missing holdout validation records `requires_holdout_validation=true`,
  `holdout_validation_status=missing`, and `delivery_allowed_without_reference=false`;
- profile banks must not store timestamp rules or per-track branches;
- every applied candidate must still pass `run_model_zoo_benchmark.py`; rejected candidates must remain diagnostic.
- `select-from-audit` is the required guard when evaluating a sweep of `orig_*` and `cal_*` teacher-profile variants.
  It writes a selector manifest, does not modify audio, and rejects calibrated variants whose adaptive audit score
  regresses against their paired original candidate. A selected calibrated regression must trigger
  `reject_selected_calibrated_candidate_and_register_new_vocal_route`, not a listen-pack promotion.
  A selected `cal_*` without an `orig_*` pair is also blocked as
  `blocked_selected_calibrated_missing_original_pair`; an unknown selected profile is blocked as
  `blocked_invalid_selected_profile`.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/build_teacher_profile_bank.py build-bank `
  --training-example drums=/app/lab_artifacts/golden_audio/<bundle>/ground_truth_manifest.json,/app/lab_artifacts/golden_audio/<candidate>/stem_manifest.json,/app/lab_artifacts/golden_audio/<teacher>/stem_manifest.json `
  --output-profile-bank /app/lab_artifacts/golden_audio/<run>/teacher_profile_bank.json

docker compose run --rm --no-deps -T ml-worker python /app/tools/build_teacher_profile_bank.py apply-bank `
  --profile-bank /app/lab_artifacts/golden_audio/<run>/teacher_profile_bank.json `
  --candidate-manifest /app/lab_artifacts/golden_audio/<candidate>/stem_manifest.json `
  --stem drums `
  --output-dir /app/lab_artifacts/golden_audio/<run>/drums
```

Square Oxygen first pass on 2026-05-14:

- artifact root: `lab_artifacts/golden_audio/flstudio_square_oxygen_teacher_profile_bank_route_20260514`;
- sweep root: `lab_artifacts/golden_audio/flstudio_square_oxygen_teacher_profile_bank_sweep_20260514`;
- DRUM baseline `tonal_strong`: rejected, `reference_likeness=90.734160`, `SDR=14.833282`,
  `SI-SDR=14.688647`;
- best tested DRUM profile-bank candidate was still rejected and below baseline (`mix0.05_gain3`:
  `reference_likeness=90.177423`, `SDR=14.520868`, `SI-SDR=14.624394`);
- VOCALS baseline `melband`: rejected, `reference_likeness=82.603726`, `SDR=9.166258`,
  `SI-SDR=8.605978`;
- best tested VOCALS profile-bank candidate was still rejected and below baseline (`mix0.05_gain3`:
  `reference_likeness=82.600357`, `SDR=9.157503`, `SI-SDR=8.598157`).

Decision: keep teacher profile bank as infrastructure and rejected diagnostic evidence. It is useful because it prevents
same-track teacher leakage from being promoted, but it is not yet a stronger no-reference route than the current best
DRUM/VOCALS candidates.

## Teacher mask separator

`ml_worker_improved/tools/train_teacher_mask_separator.py` trains a teacher-derived mask separator and then applies it to
`source_mix.flac` without reading the target reference during application. It is useful for separating three cases that
previous profile routes conflated:

- reference-free application from a trained mask model;
- blocked same-track teacher distillation with learned temporal position features;
- blocked complex-mask-prior reconstruction for proving an upper bound.

Contract:

- training may read `--teacher-target`;
- application reads only `--source-mix` and the trained artifact;
- `contract.target_reference_used_at_application=false`;
- `contract.application_reference_free=true`;
- `--context-frame-radius` may add local neighboring STFT frames as source-derived features; this is not a timestamp
  rule and records `stores_learned_local_context_features=true`;
- sparse targets must not treat zero lower-quantile target activity as "all frames active"; the trainer raises the
  activity threshold from a high-quantile target floor and records `training.diagnostics.active_frame_threshold`;
- blocked teacher-mask candidates are not copied into `--listen-dir`; their manifests record
  `presentation.status=blocked_from_listen_pack` and `presentation.result_delivery_allowed=false`;
- missing holdout keeps `contract.delivery_allowed_without_reference=false`;
- if `--time-feature-count > 0`, the artifact records `stores_timestamp_rules=true`,
  `stores_learned_temporal_position_features=true`, and `production_no_reference_generalization_claim=false`;
- if a complex prior is stored, the artifact records `stores_learned_complex_mask_prior=true` and
  `target_reference_reconstructive_prior=true`; this is diagnostic/teacher evidence only.
- `--source-feature-memory-neighbors` stores source-derived frame features and teacher-derived masks in the checkpoint.
  Application still reads only `--source-mix` plus the checkpoint, but the manifest records
  `stores_learned_source_feature_memory=true` and `source_feature_memory_generalization_claim=false`;
- `--source-feature-memory-store-complex-masks` stores a source-feature keyed complex-mask memory. This can reconstruct
  the teacher target from the same source mix without reading the reference during application, but it records
  `stores_learned_source_feature_complex_memory=true`, `target_reference_reconstructive_prior=true`, and remains blocked
  from production/listen delivery until holdout validation passes.
- `--checkpoint` applies a saved teacher-mask checkpoint to `--source-mix` without accepting `--teacher-target`. This is
  the only valid application-only path for this tool; if the checkpoint stores teacher-derived memory, the manifest still
  blocks delivery until a distinct holdout track passes.

Square Oxygen `strings_orchestra` teacher-mask checkpoint on 2026-05-18:

- no-reference/profile attempts remained rejected:
  `flstudio_square_oxygen_strings_orchestra_teacher_delta_profile_v1_gate_20260518`,
  `flstudio_square_oxygen_strings_orchestra_teacher_dictionary_application_v1_gate_20260518`, and
  `flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_v1_gate_20260518`;
- the blocked temporal complex-prior artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_temporal_complex_v3_20260518`;
- policy-fixed quality gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_temporal_complex_v3_gate_policyfix_20260518`;
- objective stem metrics pass as upper-bound evidence: SDR `80.802151`, SI-SDR `80.802372`, SIR `139.25812`,
  SAR `80.802372`, cosine `1.0`;
- delivery remains rejected by manifest policy:
  `delivery.reason=candidate_manifest_delivery_blocked`;
- blocking flags:
  `holdout_validation_missing`, `not_a_no_reference_production_route_until_holdout_passes`, `stores_timestamp_rules`,
  `target_reference_reconstructive_prior`.

Square Oxygen `strings_orchestra` no-temporal context attempts on 2026-05-18:

- `flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_no_temporal_v3_gate_20260518` remained rejected:
  SDR `-1.648912`, SI-SDR `-46.46566`, reference likeness `6.393617`;
- `flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_context_v3_gate_20260518` used local frame context
  and a corrected sparse-target activity threshold, but still remained rejected: SDR `0.0`, SI-SDR `-32.715873`,
  reference likeness `16.29056`;
- `flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_context_v4_targetheavy_gate_20260518` used aggressive
  target-preservation training pressure, but still collapsed to silence and remained rejected: SDR `0.0`, SI-SDR `0.0`,
  reference likeness `16.291134`;
- decision: local source-derived context and loss reweighting are useful infrastructure, but they are not sufficient to
  create an accepted no-reference `strings_orchestra` route from the current single-track teacher. These artifacts must
  stay diagnostic/blocked until a distinct holdout track passes.

Square Oxygen `strings_orchestra` source-feature memory checkpoint on 2026-05-18:

- magnitude-only source-feature memory artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_memory_v1_20260518`;
- magnitude-only gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_memory_v1_gate_20260518`;
- result: rejected before delivery; application was reference-free but magnitude masks were not enough for waveform
  fidelity (`SDR=-1.603768`, `SI-SDR=-11.328267`, `cosine=0.261934`);
- complex source-feature memory artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_complex_memory_v1_20260518`;
- complex source-feature memory gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_complex_memory_v1_gate_20260518`;
- checkpoint-only application artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_complex_memory_v1_checkpoint_apply_20260518`;
- checkpoint-only application gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_strings_orchestra_teacher_mask_separator_source_complex_memory_v1_checkpoint_apply_gate_20260518`;
- objective stem metrics pass as teacher evidence: SDR `80.796899`, SI-SDR `80.797120`, SIR `143.685228`,
  SAR `80.797120`, cosine `1.0`;
- checkpoint-only application records `inputs=source_mix,checkpoint` with no `teacher_target`, and reproduces the same
  objective metrics as the train+apply run;
- delivery remains rejected by manifest policy:
  `delivery.reason=candidate_manifest_delivery_blocked`;
- blocking flags:
  `holdout_validation_missing`, `not_a_no_reference_production_route_until_holdout_passes`,
  `target_reference_reconstructive_prior`.

Square Oxygen complex-mask teacher profile-bank attempt on 2026-05-18:

- profile bank:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_teacher_profile_bank_vocal_melody_20260518/teacher_profile_bank.json`;
- base no-reference candidate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v1_adaptive_rescue_20260518/stem_manifest.json`;
- teacher target:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_oracle_mask_teacher_vocal_melody_20260518/stem_manifest.json`;
- applied vocals artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_teacher_profile_bank_vocal_melody_apply_vocals_20260518`;
- applied melody artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_complex_teacher_profile_bank_vocal_melody_apply_melody_20260518`;
- both applied candidates remained rejected by the 95% reference gate, so they are diagnostic/training artifacts only;
- decision: the accepted complex-mask teacher is now the locked 1:1 teacher target, but a simple band-profile delta over
  `joint_v1_adaptive` is not enough for a no-reference production vocal/melody route.

Square Oxygen DRUM result on 2026-05-13:

- `old_best`: rejected, `reference_likeness=90.402893`, `SDR=14.699967`, `SI-SDR=14.550260`;
- global band profile: rejected, `reference_likeness=83.903544`;
- percussive-gated profile: rejected, `reference_likeness=87.444149`;
- profile-mix sweep `0.15/0.25/0.40`: all rejected and all below `old_best`;
- source-percussive-reservoir sweep `0.08/0.15/0.25`: all rejected, with lower `SIR` than `old_best`;
- guarded source-percussive-reservoir `0.08` with candidate-confirmed anti-leak gating: rejected,
  `reference_likeness=90.036152`, `SDR=14.526825`, `SI-SDR=14.521900`, still below `old_best`.

Decision: do not promote the learned profile or source-reservoir variants as production DRUM repair. The evidence shows
that the supervised residual teacher improves the stem by adding target-specific missing content that is not recoverable
from a global frequency profile or simple source percussive reservoir without pulling leakage. The next production route
must add a stronger model candidate or a learned detector/router validated on held-out reference tracks, not an EQ-style
profile shortcut.

Safety guard: source-percussive-reservoir is allowed only through source/candidate percussive-confidence gates and has
an anti-leak regression test for source-only transients. This makes the experiment safer, but does not make it a
deliverable production candidate.

## No-reference DRUM bus projection candidate

`ml_worker_improved/tools/drum_bus_projection_body_gain.py` builds a no-reference DRUM candidate from generated model
stems. It starts from the current best DRUM stem, projects correlated non-drum bus energy out of it, then applies a
guarded dynamic body-gain stage using another generated DRUM model as body reference. It does not consume the verified
target DRUM reference.

Contract:

- output `stem_manifest.json` is benchmark-compatible;
- `contract.target_reference_assisted=false`;
- `contract.fallback_used=false`;
- `contract.per_track_hardcode_allowed=false`;
- validation windows, when supplied, are report-only and do not drive routing.

Square Oxygen DRUM result on 2026-05-13:

- `old_best`: rejected, `reference_likeness=90.402893`, `SDR=14.699967`, `SI-SDR=14.550260`,
  `SIR=32.076652`, `SAR=14.679910`;
- `bus_projection_ft`: rejected, but best no-reference candidate so far, `reference_likeness=90.450914`,
  `SDR=14.724630`, `SI-SDR=14.575785`, `SIR=32.229366`, `SAR=14.700656`;
- `bus_projection_6s`: rejected, `reference_likeness=90.435532`, `SDR=14.716862`, `SI-SDR=14.567744`;
- `max_body_gain` sweep `1.00/1.06/1.18` did not beat `bus_projection_ft`.

Decision: keep `bus_projection_ft` as a rejected diagnostic candidate only. It improves all objective scores over
`old_best`, yet remains far below the accepted reference-guided teacher candidate and must not be presented as a
successful listen result. The next step is a stronger no-reference model candidate rather than more body-gain tuning.

## No-reference DRUM tonal leakage cleanup

`ml_worker_improved/tools/drum_tonal_leakage_repair.py` builds a no-reference cleanup candidate for DRUM stems that
already preserve punch/body but still contain stable pitched or harmonic residue. The tool is designed as a generic
post-route candidate, not a track-specific fix: validation windows are report-only, the repair does not consume the
verified target DRUM reference, and the emitted `stem_manifest.json` keeps `contract.target_reference_assisted=false`,
`contract.fallback_used=false`, and `contract.per_track_hardcode_allowed=false`.

Square Oxygen DRUM follow-up on 2026-05-13:

- user listening preferred `02_projection_debleed.flac` over later muffled variants, but reported audible drop residue
  around `00:57`, `01:10`, and `01:17`;
- diagnostic artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_drop_residue_diagnostics_20260513.json`;
- compare listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_debleed_tonal_cleanup_compare_listen_20260513`;
- `tonal_soft`: rejected, `reference_likeness=90.613254`, `SDR=14.795265`, `SI-SDR=14.649035`;
- `tonal_strong`: rejected diagnostic candidate, `reference_likeness=90.734160`,
  `SDR=14.833282`, `SI-SDR=14.688647`, `SIR=32.785030`, `SAR=14.802736`;
- validation-window deltas are intentionally small to avoid the muffled-drum regression:
  `user_0057 rms_delta=-0.016197 dB / percussive_minus_harmonic_delta=+0.025121 dB`,
  `user_0110 rms_delta=-0.004733 dB / percussive_minus_harmonic_delta=+0.017925 dB`,
  `user_0117 rms_delta=-0.021342 dB / percussive_minus_harmonic_delta=+0.022872 dB`.

Decision: reject `tonal_strong` for delivery and keep it only as diagnostic evidence. It improves the objective
benchmark over `bus_projection_ft` without using the target reference, but it is still far below the accepted
reference-guided teacher candidate and must not be presented as a successful listen result. If listening confirms that
residue remains, the next production route should add a stronger model candidate or model-router signal, not push this
cleanup into destructive suppression.

## Registered no-reference DRUM route gate

`ml_worker_improved/tools/export_hq_stems_full.py` emits a no-reference production contract for full-track stem exports:

- `contract.target_reference_assisted=false`;
- `contract.target_reference_used_at_application=false`;
- `contract.application_reference_free=true`;
- `contract.delivery_allowed_without_reference=true`;
- `contract.fallback_used=false`;
- `contract.per_track_hardcode_allowed=false`;
- `contract.not_timestamp_hardcoded=true`.

This contract is provenance only. It allows `run_model_zoo_benchmark.py --require-no-reference-candidates` to evaluate
the candidate as a no-reference route, but the candidate still needs to pass objective reference quality gates before it
can be promoted to a listen result.

Registered DRUM-capable routes include the standard full-track exports plus generic aggressive melody-first variants:

- `htdemucs_ft`;
- `htdemucs_6s`;
- `hybrid_htdemucs_ft_bs_roformer_6s`;
- `htdemucs_6s_aggressive_melody_first`;
- `hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first`;
- `scnet_xl_ihf_musdb18`.

## Registered no-reference MELODY route gate

`ml_worker_improved/tools/build_tonal_harmonic_melody_candidate.py` adds a registered source-mix-only melody candidate:

- registry id: `tonal_harmonic_melody_v1`;
- target/output stem: `melody`;
- source input: authoritative `source_mix.flac` only;
- `reference_sum_mix.flac` is not read;
- verified target/reference stems are not read at application time;
- manifest contract sets `target_reference_assisted=false`, `target_reference_used=false`,
  `target_reference_used_at_application=false`, `application_reference_free=true`,
  `delivery_allowed_without_reference=true`, `fallback_used=false`, `hidden_inference=false`,
  `per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`.

Square Oxygen MELODY checkpoint on 2026-05-17:

- candidate artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_harmonic_melody_v1_20260517`;
- listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_harmonic_melody_v1_listen_20260517`;
- sweep artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_harmonic_melody_sweep_20260517`;
- sweep listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_harmonic_melody_sweep_listen_20260517`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_harmonic_melody_sweep_benchmark_20260517`;
- best no-reference sweep candidate: `tonal_04_mid_focus`, rejected,
  `reference_likeness=26.936408`, `SDR=-1.571140`, `SI-SDR=-8.718745`;
- registered stack benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_melody_registered_stack_benchmark_20260517`;
- registered stack status: `candidate_space_exhausted`, no accepted melody candidate.

Decision: do not promote any 2026-05-17 MELODY candidate as production. The new route is a reusable registered
candidate and improves over the old aggressive melody-first routes on this checkpoint, but it remains objectively
rejected and must be treated as diagnostic until a stronger model/router passes the quality gate.

## Registered no-reference MELODY model-stack debleed candidate

`ml_worker_improved/tools/build_model_stack_melody_candidate.py` adds `model_stack_melody_v1` as the next
candidate after `tonal_harmonic_melody_v1` fails with audible vocal residue:

- registry id: `model_stack_melody_v1`;
- target/output stem: `melody`;
- source input: authoritative full mix plus no-reference model-stack stems only;
- base manifests must pass the no-reference contract before they can be used;
- verified target/reference stems are not read at application time;
- `reference_sum_mix.flac` is not read;
- model-derived `vocals`, `drums`, and `bass` stems are suppressor priors, not target references;
- listen artifacts remain diagnostic until `run_model_zoo_benchmark.py --require-no-reference-candidates`
  accepts the candidate.

This route addresses the semantic failure mode of source-only HPSS: vocals are harmonic, so a tonal mask can keep
them as false melody. The model-stack route builds a source-tonal prior from the authoritative full mix, validates
the no-reference model-stack base manifest, and removes positive projection against model-derived `vocals` by
default. Drum/bass projection suppression remains available as explicit parameters, but is not default because
the Square Oxygen checkpoint showed that aggressive drum/bass subtraction damages melody body when the model
drums/bass stems are correlated with musical content.

Square Oxygen MELODY checkpoint on 2026-05-18:

- artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_melody_v1_tonalprior_vocaldebleed_20260518`;
- listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_melody_v1_tonalprior_vocaldebleed_listen_20260518`;
- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_melody_v1_tonalprior_vocaldebleed_benchmark_20260518`;
- status: rejected by objective gate, not production delivery;
- improvement over `tonal_mid_focus`: `reference_likeness=29.568723` vs `26.936408`,
  `SDR=-0.987055` vs `-1.571140`, `SI-SDR=-7.943438` vs `-8.718745`;
- reference projection diagnostic: vocal projection reduced from `-29.109 dBFS`
  (`tonal_mid_focus`) to `-51.230 dBFS` (`tonalprior_vocaldebleed`) while melody projection stayed comparable
  (`-30.209 dBFS` -> `-30.277 dBFS`).

## Registered no-reference JOINT VOCAL/MELODY route

`ml_worker_improved/tools/build_joint_vocal_melody_candidate.py` adds `joint_vocal_melody_v1` for the failure mode where
melody extraction and vocal extraction must be evaluated as separate stems instead of being mixed into one harmonic
candidate:

- registry id: `joint_vocal_melody_v1`;
- target/output stems: `vocals`, `melody`;
- source input: authoritative full mix plus no-reference model-stack stems only;
- base manifests must pass the no-reference contract before they can be used;
- verified target/reference stems are not read at application time;
- `reference_sum_mix.flac` is not read;
- the melody branch uses `model_stack_melody_v1` with vocal projection suppression enabled, then applies an adaptive
  raw-melody rescue only when the de-vocal melody candidate loses too much projected melody body;
- the vocal branch uses spectral vocal dominance plus cross-projection debleed against the selected melody candidate;
- listen artifacts remain diagnostic until `run_model_zoo_benchmark.py --require-no-reference-candidates` accepts both
  stems.

Square Oxygen JOINT VOCAL/MELODY checkpoint on 2026-05-18:

- artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v1_adaptive_rescue_20260518`;
- listen artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v1_adaptive_rescue_listen_20260518`;
- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v1_adaptive_rescue_benchmark_20260518`;
- status: rejected by objective gate, not production delivery;
- improvement over the earlier joint route: mean reference likeness `54.169` vs `46.747`, melody `29.569` vs `15.194`,
  vocals `78.8` vs `78.3`;
- improvement over the aggressive hybrid base: mean reference likeness `54.169` vs `45.940`, melody `29.569` vs
  `13.516`;
- remaining blocker: the `melody` stem is still far below the 95 percent reference-likeness gate, so the next production
  step must add a stronger melody/other model candidate rather than promoting this diagnostic route.

Square Oxygen registered-route sweep on 2026-05-15:

- artifact root: `lab_artifacts/golden_audio/flstudio_square_oxygen_registered_no_ref_drum_routes_gate_20260515`;
- ground-truth bundle: `lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_20260511`;
- that sweep evaluated six candidates before SCNet registration; all six were no-reference-contract accepted but
  objective rejected;
- `tonal_stronger_contract_v3`: rejected, still best no-reference candidate, `reference_likeness=90.649920`,
  `SDR=14.805072`, `SI-SDR=14.659249`;
- `htdemucs_ft_contract`: rejected, `reference_likeness=80.310146`, `SDR=10.880711`, `SI-SDR=10.742185`;
- `htdemucs_6s_contract`: rejected, `reference_likeness=79.031149`, `SDR=10.333932`, `SI-SDR=10.098891`;
- `hybrid_bs_contract`: rejected, `reference_likeness=78.022616`, `SDR=10.066169`, `SI-SDR=9.764810`;
- `htdemucs_6s_aggressive_melody_first`: rejected, `reference_likeness=51.723186`, `SDR=4.905379`,
  `SI-SDR=3.838854`;
- `hybrid_bs_aggressive_melody_first`: rejected, `reference_likeness=51.053976`, `SDR=4.909562`,
  `SI-SDR=3.780833`.

SCNet XL IHF follow-up on 2026-05-15:

- candidate artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_xl_ihf_musdb18_20260515`;
- listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_xl_ihf_musdb18_listen_20260515`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_xl_ihf_musdb18_gate_20260515_containercp_vendor`;
- `scnet_xl_ihf_musdb18`: rejected, `presentation_status=diagnostic_rejected`,
  `reference_likeness=91.745626`, `SDR=15.849678`, `SI-SDR=15.735349`;
- `tonal_stronger_contract_v3` in the same comparison: rejected, `reference_likeness=90.649920`,
  `SDR=14.805072`, `SI-SDR=14.659249`.

SCNet bus-projection repair follow-up on 2026-05-15:

- candidate artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_bus_projection_repair_20260515`;
- listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_bus_projection_repair_listen_20260515`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_bus_projection_repair_gate_20260515`;
- `scnet_bus_projection_repair`: rejected, `presentation_status=diagnostic_rejected`,
  `reference_likeness=91.809051`, `SDR=15.920294`, `SI-SDR=15.807748`;
- the repair improved the measured SCNet baseline slightly, but it is still below the hard reference gate and must not be
  labelled as accepted/final.

SCNet tonal-leakage repair follow-up on 2026-05-15:

- compact listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_tonal_variants_listen_20260515`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_tonal_variants_gate_20260515`;
- `scnet_projection_tonal_default`: rejected, `presentation_status=diagnostic_rejected`,
  `reference_likeness=92.139455`, `SDR=16.174866`, `SI-SDR=16.069559`;
- `scnet_projection_tonal_guarded`: rejected, `reference_likeness=92.091957`, `SDR=16.145080`,
  `SI-SDR=16.038788`;
- `scnet_projection_tonal_mild`: rejected, `reference_likeness=92.070613`, `SDR=16.129136`,
  `SI-SDR=16.022366`;
- these tonal repair variants improve the measured SCNet repair baseline, but they remain below the hard reference gate
  and must not be labelled as accepted/final.

SCNet tonal-transient repair follow-up on 2026-05-15:

- candidate artifacts:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_tonal_transient_default_20260515`,
  `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_tonal_transient_guarded_20260515`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_transient_variants_gate_20260515`;
- `scnet_tonal_transient_default`: rejected, `reference_likeness=83.541365`, `SDR=11.044293`,
  `SI-SDR=12.435583`;
- `scnet_tonal_transient_guarded`: rejected, `reference_likeness=82.600524`, `SDR=11.200600`,
  `SI-SDR=11.785947`;
- transient restoration over the tonal-clean candidate reintroduced too much mismatch under the reference gate and should
  not be used as the next DRUM production route without a different transient-specific design.

SCNet micro-leakage debleed follow-up on 2026-05-15:

- compact listen artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_micro_debleed_listen_20260515`;
- benchmark artifact: `lab_artifacts/golden_audio/flstudio_square_oxygen_micro_debleed_gate_20260515`;
- `scnet_micro_debleed_guarded`: rejected, `reference_likeness=92.077279`, `SDR=16.189285`,
  `SI-SDR=16.152683`;
- `scnet_micro_debleed_aggressive`: rejected, `reference_likeness=92.005251`, `SDR=16.097245`,
  `SI-SDR=16.129913`;
- the guarded micro-debleed variant improves SDR/SI-SDR against `scnet_projection_tonal_default`, but loses
  reference-likeness; keep it as a diagnostic listening tradeoff, not the new measured-best DRUM route.

Source-residual and low-activity DRUM cleanup follow-up on 2026-05-15:

- source-residual benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_source_residual_gate_20260515`;
- drop-cleanup compare listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_drop_cleanup_listen_pack_20260515`;
- `source_residual_plus_proxy_debleed`: rejected but best measured no-reference diagnostic route,
  `reference_likeness=92.242870`, `SDR=16.245675`, `SI-SDR=16.230931`, `SIR=36.625756`,
  `SAR=16.291089`;
- `source_residual_debleed`: rejected, `reference_likeness=92.159269`, `SDR=16.207561`,
  `SI-SDR=16.196631`;
- window projection recheck showed the remaining audible "drop/melody" complaint is dominated by named `fx` leakage
  in low-activity DRUM regions: e.g. `43.8-44.6` and `149.5-150.5` were almost entirely explained by the verified
  `fx` reference while the verified `drums` stem was inactive;
- `low_activity_only_gate`: rejected, `reference_likeness=91.406885`, `SDR=15.785402`,
  `SI-SDR=15.683810`; it lowers the low-activity FX residue windows by roughly 3-5 dB while keeping strong DRUM
  windows close to the baseline;
- `low_activity_strict_gate`: rejected, `reference_likeness=90.683031`, `SDR=15.222931`,
  `SI-SDR=15.115183`; it lowers the same FX residue windows more aggressively but costs more reference likeness and
  target-body accuracy.

Decision: keep `source_residual_plus_proxy_debleed` as the measured-best no-reference diagnostic candidate. Keep
`low_activity_only_gate` and `low_activity_strict_gate` as user-facing diagnostic comparisons for drop-residue listening
only; they must not replace the measured-best route because the verified reference gate shows lower global 1:1 likeness.
The next production fix should add a stronger DRUM/FX discriminator or model candidate, not simply raise low-activity
suppression until the stem becomes quieter.

Kick-synchronous proxy-tail cleanup follow-up on 2026-05-15:

- user listening reported that `01:22-01:45` still has audible melodic/drop residue on kick moments; forensic projection
  confirmed this is not ordinary low-activity leakage and not strongly explained by the named `melody` reference stem.
  The strongest short-window non-target projection is mostly low/bass-like or FX-linked while the DRUM reference remains
  highly correlated, so the route must treat it as kick-synchronous proxy leakage / tonal tail, not as silence cleanup;
- TDD added a generic `post_transient_tail_gate` layer to `ml_worker_improved/tools/drum_micro_leakage_debleed.py`,
  including a guard that requires non-drum/source-residual proxy evidence before suppressing harmonic tails. A clean
  harmonic drum tail without proxy evidence must not be gated;
- artifact roots:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_kick_tail_gate_proxy_surgical_20260515`,
  `lab_artifacts/golden_audio/flstudio_square_oxygen_kick_tail_gate_proxy_assertive_20260515`;
- compact listen artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_kick_sync_proxy_tail_compare_listen_20260515`;
- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_kick_tail_gate_proxy_benchmark_20260515`;
- the ungated tail cleanup variants were rejected and destructive (`kick_tail_surgical reference_likeness=88.006239`,
  `kick_tail_medium=78.516812`, `kick_tail_strict=66.005018`), proving that post-kick harmonic suppression without
  proxy evidence wrongly treats valid drum resonance as leakage;
- proxy-gated variants are safer but still rejected: `kick_tail_proxy_surgical reference_likeness=91.639277`,
  `SDR=15.839417`, `SI-SDR=15.964825`; `kick_tail_proxy_assertive reference_likeness=90.193482`,
  `SDR=14.932130`, `SI-SDR=15.222970`;
- local `01:22-01:45` projection recheck showed only tiny improvement from `proxy_surgical` on the worst kick windows
  (for example `84.5-85.0` bass projection changed from `-12.564 dB` to `-12.794 dB`) and regressions elsewhere
  (`94.0-94.5` bass projection changed from `-13.391 dB` to `-13.264 dB`).

Decision: keep `post_transient_tail_gate` as an available diagnostic/repair primitive because it is now guarded by
proxy evidence and covered by tests, but do not promote the generated proxy-tail candidates. The remaining audible
kick-synchronous residue requires a stronger DRUM/non-DRUM discriminator or a new model/segment-router candidate; simply
attenuating post-kick harmonic tails lowers objective 1:1 quality and risks muffled drums.

Reference-stem projection diagnostic on 2026-05-15:

- `drum_micro_leakage_debleed.py` now has an explicit `--leakage-evidence-mode` contract. `verified_reference_stems`
  marks candidates as `target_reference_assisted=true`, `application_reference_free=false`, and
  `delivery_allowed_without_reference=false`, so FL Studio stems can be used for teacher diagnostics without being
  mislabeled as a no-reference production route;
- evidence-mode contract:
  - `model_proxy`: requires at least one model/proxy `--leakage-stem-path`, forbids `--source-mix-path`, and remains
    `target_reference_assisted=false`, `application_reference_free=true`, `delivery_allowed_without_reference=true`;
  - `source_residual_proxy`: requires `--source-mix-path`, may include supplemental model/proxy `--leakage-stem-path`
    entries, and remains `target_reference_assisted=false`, `application_reference_free=true`,
    `delivery_allowed_without_reference=true`;
  - `verified_reference_stems`: requires verified non-target `--leakage-stem-path` entries, forbids `--source-mix-path`,
    and is diagnostic/teacher only with `target_reference_assisted=true`, `application_reference_free=false`,
    `delivery_allowed_without_reference=false`;
- safety contract:
  - numeric gate/reduction/headroom parameters are fail-fast validated before audio processing;
  - length mismatches are explicitly reported in `diagnostics.length_audit` under `policy=trim_to_shortest_input`, so a
    candidate cannot silently hide proxy/reference tail truncation in provenance;
- artifact roots:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_refstem_projection_melody_fx_surgical_20260515`,
  `lab_artifacts/golden_audio/flstudio_square_oxygen_refstem_projection_melody_fx_assertive_20260515`,
  `lab_artifacts/golden_audio/flstudio_square_oxygen_refstem_projection_all_nondrum_surgical_20260515`;
- compact listen artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_refstem_projection_compare_listen_20260515`;
- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_refstem_projection_benchmark_20260515`;
- all reference-stem projection variants remain rejected by the 95% reference gate, but they improve objective fit over the
  current no-reference candidate: `current_best reference_likeness=92.242870`, `SDR=16.245675`, `SI-SDR=16.230931`;
  `ref_melody_fx_surgical reference_likeness=92.833254`, `SDR=16.570158`, `SI-SDR=16.591209`;
  `ref_melody_fx_assertive reference_likeness=92.819031`, `SDR=16.308367`, `SI-SDR=16.647664`;
  `ref_all_nondrum_surgical reference_likeness=92.778459`, `SDR=16.586155`, `SI-SDR=16.609794`;
- leakage matrix improved the main non-target projection source (`fx`) from `0.053506709` on `current_best` to
  `0.040087623` on `ref_melody_fx_surgical` and `0.031330338` on `ref_melody_fx_assertive`, confirming that verified
  non-drum stems help identify the kick-synchronous residue. Remaining failures are still
  `reference_likeness_below_threshold`, `sample_residual_above_threshold`, and transient/body mismatches, so the route
  is teacher/diagnostic only until a reference-free discriminator can reproduce this evidence without verified stems.

Residual reference-leakage gate update on 2026-05-15:

- `run_ground_truth_quality_gate.py` now evaluates micro-window non-target leakage twice:
  - candidate-level projection: `candidate_window` against non-target reference stems;
  - residual-level projection: `(candidate_window - verified_target_reference_window)` against non-target reference stems;
- the residual-level path adds `micro_residual_reference_stem_leakage` and
  `worst_windows_by_residual_leakage_source`, so a valid kick/snare transient can no longer mask melody/drop residue that
  remains after subtracting the verified DRUM stem;
- `micro_quality_windows` also records `worst_windows_by_time_bucket` with bounded 10-second buckets. This prevents loud
  unrelated defects elsewhere in the track from hiding quieter local defects in user-reported windows such as
  `01:22-01:45`;
- `micro_quality_windows.issue_label_counts` is part of the delivery contract input, so blocking labels are aggregated
  from all micro windows instead of only the global `worst_windows` preview;
- quiet sustained residual leakage is labelled by residual source strength even when it does not cross spike/transient
  thresholds;
- audit artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_residual_leakage_gate_audit_20260515`;
- the audit report now exposes local DRUM issues in the target area: bucket `80.000-90.000` flags `85.625-86.200s`
  windows with `candSource=melody` and `micro_reference_stem_leakage`; bucket `90.000-100.000` flags `92.250-99.250s`
  windows with `candSource=melody`, `micro_residual_spike`, `micro_transient_mismatch`, and
  `micro_target_transient_dropout`;
- final audit counts include `micro_residual_reference_stem_leakage=5873`, `micro_reference_stem_leakage=5628`,
  `micro_residual_spike=7168`, and `micro_transient_mismatch=7285`, and all of those labels are now present in
  `forensic_delivery_contract.blocking_issue_labels`;
- audit-guided residual debleed follow-up:
  - tool: `ml_worker_improved/tools/drum_audit_guided_residual_debleed.py`;
  - artifacts:
    `lab_artifacts/golden_audio/flstudio_square_oxygen_audit_guided_residual_debleed_20260515`,
    `lab_artifacts/golden_audio/flstudio_square_oxygen_audit_guided_residual_debleed_surgical_20260515`;
  - compact listen pack:
    `lab_artifacts/golden_audio/flstudio_square_oxygen_audit_guided_drum_compare_listen_20260515`;
  - both audit-guided outputs are reference-assisted diagnostics and remain rejected, not no-reference production
    candidates;
  - objective comparison against `current_best`: `current_best reference_likeness=92.242870`, `SDR=16.245675`,
    `SI-SDR=16.230931`; all-source audit-guided candidate `reference_likeness=92.883158`, `SDR=16.498631`,
    `SI-SDR=16.505467`; melody/piano/fx surgical candidate `reference_likeness=92.617036`, `SDR=16.405932`,
    `SI-SDR=16.405410`;
  - target-bucket evidence is still mixed and does not clear the hard gate, so this is a listening/diagnostic pack for
    the reported DRUM melody/drop residue, not a promoted final stem.
- this is a diagnostic/reference-quality gate, not a timestamp hardcode and not a reference-assisted production route.
  It is intended to expose the exact class of error reported in kick/snare windows such as `01:22-01:45`: non-drum
  harmonic residue surviving on top of target drum events.

Decision: do not promote the aggressive melody-first routes as DRUM production candidates. The source-residual plus proxy
debleed route is now the best measured no-reference diagnostic DRUM route on Square Oxygen, but it is still rejected by
the hard reference gate and must not be labelled as accepted/final. The next production improvement should add a stronger
DRUM/FX discriminator or stronger model candidate, then rerank against the remaining forensic labels instead of applying
destructive low-activity suppression without gate evidence.

Delivery-gate rebenchmark artifact:

- manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_delivery_gate_rebenchmark_20260513/model_zoo_benchmark_manifest.json`;
- accepted selected candidate: `residual_s080`, `presentation_status=accepted_for_listen_pack`,
  `reference_likeness=97.898723`, `SDR=26.941043`, `SI-SDR=27.895583`;
- rejected no-reference candidates keep `delivery_allowed=false` and `presentation_status=blocked_from_delivery`;
- listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_delivery_gate_teacher_vs_no_reference_listen_20260513`.

The accepted `residual_s080` artifact is reference-guided (`target_reference_assisted=true`) and can be used as a
teacher/quality target, not as proof that the no-reference production DRUM route is solved.

No-reference registered-model exhaustion check:

- manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_no_reference_registered_hybrid_benchmark_20260513/model_zoo_benchmark_manifest.json`;
- status: `candidate_space_exhausted`;
- `selected_candidate=null`;
- `tonal_strong`: rejected, `presentation_status=diagnostic_rejected`, `reference_likeness=90.734160`;
- `debleed`: rejected, `presentation_status=diagnostic_rejected`, `reference_likeness=90.450914`;
- `htdemucs_ft_existing`: rejected, `reference_likeness=90.228159`;
- `htdemucs_6s_existing`: rejected, `reference_likeness=89.204681`;
- `hybrid_registered`: rejected, `reference_likeness=77.967430`.

Decision: the current no-reference DRUM model stack is exhausted on Square Oxygen. Do not ship the best rejected file.
Use the accepted reference-guided artifact as a teacher/quality target, or add a genuinely stronger DRUM model candidate,
then rerun the same objective gate.

Projection-debleed forensic recheck on 2026-05-14:

- no-reference cleanup sweep:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_cleanup_sweep_20260514`;
- no-reference gate manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_sweep_forensic_gate_20260514/model_zoo_benchmark_manifest.json`;
- baseline `projection_debleed`: rejected, `presentation_status=diagnostic_rejected`,
  `reference_likeness=90.022397`, `SDR=14.156531`, `SI-SDR=13.986470`;
- best no-reference variant `tonal_stronger`: rejected, `presentation_status=diagnostic_rejected`,
  `reference_likeness=90.214802`, `SDR=14.225147`, `SI-SDR=14.057961`;
- teacher/upper-bound gate manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_teacher_s100_forensic_gate_20260514/model_zoo_benchmark_manifest.json`;
- `reference_guided_s100`: rejected by the current forensic contract, but much closer to the verified target:
  `reference_likeness=98.918522`, `SDR=33.808797`, `SI-SDR=37.057965`.

Decision: the no-reference tonal cleanup over `02_projection_debleed.flac` only makes a small objective improvement and
must stay diagnostic. The `reference_guided_s100` artifact is not a production result because it uses the verified target
reference, but it is the current teacher signal for what the production DRUM route must learn to approximate without
target-reference access. The next production attempt should add a stronger no-reference DRUM candidate/router signal, not
increase tonal suppression over `projection_debleed`.

Projection-debleed s100 delta-profile and no-reference contract gate on 2026-05-14:

- delta-profile sweep root:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_delta_profile_s100_sweep_20260514`;
- no-reference contract gate manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_delta_profile_s100_no_ref_gate_20260514/model_zoo_benchmark_manifest.json`;
- all seven candidates were rejected with `selected_candidate=null`;
- `reference_guided_s100` remained the closest diagnostic teacher (`reference_likeness=98.918522`, `SDR=33.808797`,
  `SI-SDR=37.057965`) but was hard-rejected by `candidate_contract_not_no_reference_safe`;
- best delta-profile application was worse than `tonal_stronger`: `delta_mix005_gain2` had
  `reference_likeness=89.731173`, `SDR=14.030748`, `SI-SDR=13.960644`;
- regenerated no-reference tonal candidate with the completed manifest contract:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_stronger_contract_v3_20260514/stem_manifest.json`;
- contract-v3 gate manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_stronger_contract_v3_no_ref_gate_20260514/model_zoo_benchmark_manifest.json`;
- `tonal_stronger_contract_v3` passed the no-reference manifest contract (`no_reference_contract.status=accepted`) but
  failed objective delivery (`reference_likeness=90.214802`, `SDR=14.225147`, `SI-SDR=14.057961`).

Decision: the s100 delta-profile path is not a stronger no-reference DRUM route. It is useful as diagnostic evidence, but
it worsens objective similarity relative to the already rejected tonal cleanup. The new benchmark hard gate is now the
required safety layer for any future no-reference production DRUM candidate.

Teacher-guided DRUM delta-profile follow-up:

- teacher manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_teacher_residual_from_tonal_strong_s080_20260513/stem_manifest.json`;
- teacher benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_teacher_residual_from_tonal_strong_s080_benchmark_20260513/model_zoo_benchmark_manifest.json`;
- accepted reference-guided teacher from current base: `reference_likeness=97.396903`, `SDR=25.804777`,
  `SI-SDR=26.482495`, `SIR=45.637425`, `SAR=26.566563`;
- delta-profile benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_teacher_guided_delta_benchmark_20260513/model_zoo_benchmark_manifest.json`;
- `delta_mix025`: rejected, `reference_likeness=88.723339`, `SDR=13.617908`, `SI-SDR=14.468651`;
- `delta_mix050`: rejected, `reference_likeness=85.786747`, `SDR=11.959009`, `SI-SDR=14.133467`;
- `delta_reservoir008`: rejected, `reference_likeness=88.725098`, `SDR=13.629074`, `SI-SDR=14.474622`;
- listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drum_teacher_guided_delta_compare_listen_20260513`.

Decision: do not promote the learned delta-profile candidates. The accepted reference-guided residual teacher proves
that missing target content can be recovered with reference supervision, but the current generic band/profile transfer
does not convert that teacher into a usable no-reference route. The next production route must learn a stronger
segment/model policy from multiple verified tracks or add a stronger external DRUM model candidate.

## Reference-guided vocal ensemble

`ml_worker_improved/tools/build_reference_guided_vocal_ensemble.py` builds a supervised teacher candidate from a
verified FL Studio vocal stem and multiple full-track vocal candidate manifests. It is a benchmark/learning artifact,
not a production no-reference separator.

Contract:

- the command is Docker-only;
- inputs are a verified `ground_truth_manifest.json` plus repeated `--candidate-manifest LABEL=PATH` values;
- output `stem_manifest.json` is benchmark-compatible and compact;
- per-window blend weights and diagnostics are written to `ensemble_window_diagnostics.json`, referenced by hash from
  the manifest, and must not be embedded directly in `stem_manifest.json`;
- `contract.target_reference_assisted=true`;
- `contract.delivery_allowed_without_reference=false`;
- `contract.registered_model_candidate=false`;
- `contract.fallback_used=false`;
- `contract.per_track_hardcode_allowed=false`.

Use this route to measure how much quality is still available from existing candidates when a verified reference is
present. Do not register it as a normal model-zoo candidate until its learned behavior is converted into reference-free
rules and validated on held-out tracks. A local Square Oxygen run reached `reference_likeness=96.342948`, but delivery
remained rejected by the objective SDR/SI-SDR/SAR gates; treat that artifact as diagnostic, not final.

## Reference quality failure localization

`ml_worker_improved/tools/localize_reference_quality_failures.py` turns a rejected reference-backed candidate into a
ranked diagnostic report. It is meant to explain what must be repaired next instead of handing a known-bad file to
listening review without context.

Contract:

- the command is Docker-only;
- inputs are a verified `ground_truth_manifest.json` and one candidate `stem_manifest.json`;
- the report separates `active_reference` windows from `near_silent_reference` windows;
- near-silent windows with residual below `candidate_residual_floor_db` are labelled
  `near_silent_reference_residual_floor` and are not counted as active vocal defects;
- candidate residual above the floor on verified silent reference should remain actionable as `silence_residual_leakage`;
- active vocal windows are ranked by SDR/SI-SDR/correlation severity and include dominant frequency error bands;
- stdout prints a compact summary only, while the full window ledger is written to
  `reference_quality_failure_localization.json`.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/localize_reference_quality_failures.py `
  --ground-truth-manifest /app/lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_20260511/ground_truth_manifest.json `
  --candidate-manifest /app/lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_vocal_ensemble_w000625_dense_compact_20260513/candidate/stem_manifest.json `
  --output-dir /app/lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_error_localization_compact_20260513 `
  --stem vocals
```

The Square Oxygen diagnostic showed two distinct classes: 13 near-silent residual-floor windows and 30 active vocal
defect windows. The active defects were concentrated around the high-mid/high bands, especially `5600-8000 Hz`,
`2800-4000 Hz`, `4000-5600 Hz`, and `1400-2000 Hz`. This makes the next repair rule measurable: improve active-window
band SDR/SI-SDR without increasing silent residual leakage.

## Reference-guided active band repair

`ml_worker_improved/tools/build_reference_guided_band_repair.py` consumes a localization report and creates a
reference-assisted repair candidate. It repairs only windows marked as active vocal defects and only the dominant
frequency bands reported by `localize_reference_quality_failures.py`.

Contract:

- the command is Docker-only;
- inputs are a verified ground-truth manifest, a candidate manifest, and a localization report;
- the candidate source audio path must match the verified ground-truth mix path;
- it is supervised by the verified reference stem and therefore must not be registered as a no-reference production
  model;
- near-silent reference windows labelled `near_silent_reference_residual_floor` are not repaired;
- malformed localization reports are rejected fail-fast before audio files are written;
- output `stem_manifest.json` is benchmark-compatible but has `target_reference_assisted=true` and
  `delivery_allowed_without_reference=false`;
- repair details are written to `band_repair_diagnostics.json`;
- success must be measured by a fresh `run_model_zoo_benchmark.py --stem vocals` pass and a post-repair localization
  report.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/build_reference_guided_band_repair.py `
  --ground-truth-manifest /app/lab_artifacts/golden_audio/flstudio_ground_truth_square_oxygen_20260511/ground_truth_manifest.json `
  --candidate-manifest /app/lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_vocal_ensemble_w000625_dense_compact_20260513/candidate/stem_manifest.json `
  --localization-report /app/lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_error_localization_final_20260513/reference_quality_failure_localization.json `
  --output-dir /app/lab_artifacts/golden_audio/flstudio_square_oxygen_reference_guided_band_repair_20260513/candidate `
  --stem vocals
```

The Square Oxygen supervised band-repair candidate improved the reference-backed vocal result from rejected to accepted:
`reference_likeness=96.963515`, `SDR=20.114750`, `SI-SDR=20.080975`, and `SAR=20.081669`. Post-repair localization
reduced active defect windows from 30 to 26 while keeping near-silent residual windows at 13. Treat this as a supervised
teacher result; the production task is to convert the learned repair policy into a reference-free route and validate it
on held-out tracks.

## Objective Delivery Policy

`run_ground_truth_quality_gate.py` does not fail the process only because a candidate misses an audio-quality
threshold. It writes the decision into JSON instead:

- `overall.status=passed|failed` records objective threshold status;
- `delivery.status=accepted|rejected` controls whether a candidate may be presented as a successful listen result;
- `delivery.reason=objective_threshold_failed` means the file is diagnostic only and must not be labelled as best,
  selected, accepted, or final output;
- evaluate-mode summaries repeat `delivery_status` and `delivery_reason` for each candidate.
- model-zoo summaries additionally set `delivery_allowed=false` and `presentation_status=blocked_from_delivery` for
  rejected candidates, even if a rejected file is useful as evidence during debugging.

## Production model registry

`ml_worker_improved/worker/model_registry.py` is the source-of-truth registry for model-zoo candidate families.
It does not run inference by itself. It records model capabilities, required checkpoints, command templates,
expected manifest paths, target stems, and issue labels each candidate family is expected to address.

Current registered families are generated from `default_model_registry()` and must stay synchronized with code:

- `htdemucs_ft`;
- `htdemucs_6s`;
- `htdemucs_6s_aggressive_melody_first`;
- `tonal_harmonic_melody_v1`;
- `model_stack_melody_v1`;
- `joint_vocal_melody_v1`;
- `joint_vocal_melody_v2`;
- `joint_vocal_melody_v3`;
- `htdemucs_ft_drums_plus_percussive_fx_aux`;
- `htdemucs_6s_drums_plus_percussive_fx_aux`;
- `hybrid_htdemucs_ft_bs_roformer_6s_drums_plus_percussive_fx_aux`;
- `querysep_banquet_strings_orchestra_v1`;
- `model_stack_strings_orchestra_v1`;
- `model_stack_strings_orchestra_hybrid_bs_v1`;
- `scnet_xl_ihf_musdb18`;
- `hybrid_htdemucs_ft_bs_roformer_6s`;
- `hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first`;
- `mdx23c_vocals`;
- `melband_roformer_vocals`;
- `becruily_karaoke_vocals`;
- `bs_mamba2_vocals`;
- `audio_separator_bs_roformer_vocals`;
- `audio_separator_melband_roformer_viperx_vocals`.

`scnet_xl_ihf_musdb18` is registered through `ml_worker_improved/tools/export_scnet_full.py`. It runs the local
SCNet XL IHF MUSDB18 checkpoint on the full source track and writes benchmark-compatible `stem_manifest.json` files
with top-level `stems.drums`, `stems.bass`, `stems.other`, and `stems.vocals`. It is a no-reference registered model
candidate: the manifest must keep `target_reference_assisted=false`, `application_reference_free=true`,
`fallback_used=false`, `per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`.

`htdemucs_ft_drums_plus_percussive_fx_aux` and `htdemucs_6s_drums_plus_percussive_fx_aux` are registered through
`ml_worker_improved/tools/export_drums_plus_percussive_fx_aux.py`. They run full-track HTDemucs aggressive
melody-first routes, then write benchmark-compatible candidate manifests with top-level
`stems.drums_plus_percussive_fx_aux`. This is a no-reference candidate route for the derived
`drums_plus_percussive_fx_aux` target; it must keep `target_reference_assisted=false`,
`target_reference_used_at_application=false`, `delivery_allowed_without_reference=true`, `fallback_used=false`,
and `per_track_hardcode_allowed=false`. It must not reuse the verified reference stem or any track-specific
timestamp/EQ rule.

`querysep_banquet_strings_orchestra_v1` is registered through
`ml_worker_improved/tools/build_querysep_strings_orchestra_candidate.py`. It is a no-reference long-tail
candidate route for `strings_orchestra`: first it builds source-mix base stems with
`export_track_stems_aggressive_melody_first`, then calls `querysep_long_tail_build` with
`backend=banquet_byoq`, `dry_run=false`, and target `strings_orchestra`. The wrapper must not read the
verified `strings_orchestra` reference stem at application time, must not use track timestamps, and must leave
the result in `presentation_status=requires_objective_gate` until `run_model_zoo_benchmark.py` or the RALPH
gate accepts it. By default the wrapper passes `querysep_output_root=null` so the long-tail builder maps the
current run output directory through `path_mappings`; use `--querysep-output-root` only when the QuerySep
worker needs a different shared-volume root.

Square Oxygen `querysep_banquet_strings_orchestra_v1` runtime checkpoint on 2026-05-18:

- candidate artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_querysep_banquet_strings_orchestra_v1_20260518`;
- status: rejected before delivery;
- failure reason: `long_tail_model_router_rejected_all_candidates`;
- candidates tested by `querysep_long_tail_build`: `banquet_byoq`,
  `banquet_byoq_harmonic_residual_query`, `windowed_proxy_decorrelated_candidate`, and `diagnostic_bus_sum`;
- selected production candidate: none;
- blocker labels: `long_tail_presence_too_weak`, `windowed_proxy_correlation_too_high`,
  `possible_other_proxy_leakage`, and `proxy_copy_correlation_too_high`;
- diagnostic reference check against the verified `strings_orchestra` target showed the rejected candidates
  were not gain-fixable: cosine to reference was about `0.006` for the Banquet candidates and scale-fit SDR
  was about `0 dB`;
- decision: keep the route registered as a tested no-reference candidate family, but do not deliver its
  current Square Oxygen output. The next strings/orchestra step requires a stronger candidate family or
  teacher-to-no-reference training route, not threshold relaxation.

`model_stack_strings_orchestra_v1` is registered through
`ml_worker_improved/tools/build_model_stack_strings_orchestra_candidate.py`. It is a no-reference
strings/orchestra candidate route that builds from `source_mix.flac` plus source-mix base model stems, applies
harmonic residual and non-target debleed logic, and writes a benchmark-compatible `strings_orchestra` candidate
manifest. It must keep `target_reference_assisted=false`, `target_reference_used_at_application=false`,
`application_reference_free=true`, `delivery_allowed_without_reference=true`, `fallback_used=false`,
`per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`. Its output remains diagnostic until the
objective gate accepts it; rejected candidates must not be copied into accepted/production listen packs or
described as production listen results.

Square Oxygen `model_stack_strings_orchestra_v1` runtime checkpoint on 2026-05-18:

- broad source-tonal candidate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_v1_20260518`;
- broad candidate gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_v1_gate_20260518`;
- broad candidate status: rejected before delivery with `delivery_reason=objective_threshold_failed`;
- broad candidate reference diagnostics: `reference_likeness=1.172573`, `sdr_db=-20.851238`,
  `si_sdr_db=-36.832609`, `sir_db=-24.009601`, `sar_db=-12.080961`;
- stricter other-only candidate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_v1_other_only_20260518`;
- stricter other-only gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_v1_other_only_gate_20260518`;
- stricter other-only status: rejected before delivery with `delivery_reason=objective_threshold_failed`;
- stricter other-only reference diagnostics: `reference_likeness=0.706019`, `sdr_db=-6.571893`,
  `si_sdr_db=-42.431295`, `sir_db=-17.178614`, `sar_db=-24.958913`;
- decision: keep the route registered and tested as a no-reference candidate family, but do not deliver either
  current Square Oxygen output. The current HTDemucs-derived base stems do not expose enough usable
  `strings_orchestra` signal for this route; the next production step needs stronger source-mix base model
  assets or a teacher-trained no-reference route, not another presentation/listen artifact.

`model_stack_strings_orchestra_hybrid_bs_v1` uses the same no-reference strings/orchestra builder as
`model_stack_strings_orchestra_v1`, but switches the source-mix base export to
`hybrid_htdemucs_ft_bs_roformer_6s`. The route requires the mounted vendor assets under
`/app/vendor/music_source_separation_training` and `/app/vendor/bs_roformer_official/*`, prepares the BS-RoFormer
runtime cache before direct model-stack export, and must still pass the objective gate before any delivery/listen
claim. The model-stack base manifest resolver also accepts `source_audio_path` in no-reference base manifests, so
SCNet-generated multistem manifests can be used as diagnostic bases without weakening source hash validation.

Square Oxygen stronger-base `strings_orchestra` checkpoints on 2026-05-18:

- hybrid BS-RoFormer base artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_hybrid_bs_v1_20260518`;
- hybrid BS-RoFormer gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_hybrid_bs_v1_gate_20260518`;
- hybrid BS-RoFormer status: rejected before delivery with `delivery_reason=objective_threshold_failed`;
- hybrid BS-RoFormer diagnostics: `reference_likeness=1.173903`, `sdr_db=-20.816123`,
  `si_sdr_db=-36.925829`, `sir_db=-24.367917`, `sar_db=-11.805324`;
- SCNet base artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_xl_ihf_strings_base_20260518`;
- SCNet-based model-stack diagnostic artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_scnet_base_v1_20260518`;
- SCNet-based diagnostic gate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_model_stack_strings_orchestra_scnet_base_v1_gate_20260518`;
- SCNet-based diagnostic status: rejected before delivery with `delivery_reason=objective_threshold_failed`;
- SCNet-based diagnostics: `reference_likeness=1.470943`, `sdr_db=-28.350428`,
  `si_sdr_db=-35.236250`, `sir_db=-24.582079`, `sar_db=-9.728442`;
- decision: do not deliver these strings/orchestra candidates. Both stronger source-mix bases still fail by a wide
  margin against the verified reference stem. The next useful step is not another listen pack; it is either a
  teacher-trained no-reference strings route or a different specialist long-tail model family with a real accepted
  gate result.

`export_drums_plus_percussive_fx_aux.py` also supports a base-manifest composition mode for stronger no-reference
strict-DRUM candidates:

```bash
python /app/tools/export_drums_plus_percussive_fx_aux.py \
  --base-manifest /app/path/to/strict_drum_candidate/stem_manifest.json \
  --output-dir /app/path/to/drums_plus_percussive_fx_aux_candidate
```

This mode copies the candidate's top-level `stems.drums` into a derived
`stems.drums_plus_percussive_fx_aux` manifest without running hidden inference and without reading the verified target
reference at application time. The exporter now validates the declared parent manifest chain before writing a
no-reference delivery contract: if any declared ancestor manifest used a target reference, fallback, timestamp hardcode,
or per-track hardcode, the route fails fast instead of relabeling a teacher-derived output as production no-reference.
It is useful for benchmarking whether the current best verified no-reference strict-DRUM route is a stronger base for
the composite target than raw Demucs. The route remains a candidate only: the output must pass
`run_model_zoo_benchmark.py --require-no-reference-candidates` before any delivery claim.

Square Oxygen diagnostic on 2026-05-17:

- base strict candidate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_stronger_contract_v3_20260514`;
- derived composite candidate:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_from_tonal_stronger_v3_20260517`;
- benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_from_tonal_stronger_v3_benchmark_20260517`;
- result: rejected, but stronger than raw registered Demucs composite candidates
  (`reference_likeness=78.276160`, `SDR=15.145527`, `SI-SDR=15.018001` vs roughly `50%` likeness for the first
  registered Demucs composite sweep).

The `drum_tonal_leakage_repair.py` route is replayable from its saved diagnostic manifest. Use replay mode to rebuild the
same audio from the recorded input/config and fail if the output hash drifts:

```bash
python /app/tools/drum_tonal_leakage_repair.py \
  --replay-manifest /app/lab_artifacts/golden_audio/flstudio_square_oxygen_projection_debleed_tonal_stronger_contract_v3_20260514/drum_tonal_leakage_repair_manifest.json \
  --replay-output-dir /app/lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_stronger_v3_manifest_replay_20260517 \
  --listen-dir /app/lab_artifacts/golden_audio/flstudio_square_oxygen_tonal_stronger_v3_manifest_replay_listen_20260517
```

Replay validation on 2026-05-17 reproduced
`ce57650399cad832fc0726b65c9dea7b6825ddda26f94e50c466f320249f09c5`, matching the stored
`04_projection_debleed_tonal_stronger_contract_v3.flac`.

Source-of-truth reconciliation on 2026-05-17 found that the preferred Square Oxygen DRUM lineage passes through
`flstudio_square_oxygen_drum_router_target_only_20260511`, whose router manifest records
`target_reference_used=true`. Therefore this listen pack is a teacher-derived diagnostic benchmark for distilling a
repeatable route, not a clean production no-reference proof by itself. New no-reference manifest composition must keep
declared ancestry visible and reject tainted ancestors.

## No-reference percussive-FX aux estimator

`ml_worker_improved/tools/build_percussive_fx_aux_candidate.py` builds a diagnostic no-reference
`drums_plus_percussive_fx_aux` candidate from existing safe manifests:

- base input: a no-reference strict `drums` candidate manifest;
- aux inputs: one or more no-reference full-stem manifests containing non-drum stems such as `other` or `fx`;
- application-time reference use is forbidden;
- the tool rejects source manifests that used target references, fallback, per-track hardcodes, or timestamp hardcodes;
- the tool rejects missing or incomplete source `contract` blocks, rejects runs where none of the requested aux stems
  exist, and constrains `--listen-label` so listen files cannot be written outside `--listen-dir`;
- output keeps `target_reference_assisted=false`, `target_reference_used_at_application=false`,
  `application_reference_free=true`, `delivery_allowed_without_reference=true`, `fallback_used=false`,
  `per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`;
- output exposes the estimated aux layer as `stems.percussive_fx_aux` in addition to the composite
  `stems.drums_plus_percussive_fx_aux`, so FX/clap listening can inspect the extracted aux layer directly without
  modifying the base DRUM candidate; listen packs now include `02_percussive_fx_aux.flac` for this direct check;
- the candidate still requires `run_model_zoo_benchmark.py --require-no-reference-candidates` before any delivery claim.

The estimator uses frame-level transient/noise evidence (`spectral_flatness`, spectral flux, activity floor, and a
bounded aux RMS cap) to avoid copying sustained melody from non-drum stems. It is a candidate generator, not an accepted
separator.

`ml_worker_improved/tools/build_percussive_fx_aux_profile_candidate.py` is the stricter clap/riser-class diagnostic route.
It uses a verified `percussive_fx_aux` teacher file only to build a spectral-temporal event profile, then applies that
profile to `full_mix - base_drum_candidate` so the output follows events present in the full mix rather than teacher
timestamps. The manifest must record `profile_source_reference_assisted=true`, `target_reference_used_at_application=false`,
`application_reference_free=true`, and `delivery_allowed_without_reference=false` until a holdout/generalization gate
promotes the learned profile. The tool writes both `stems.percussive_fx_aux` and the composite
`stems.drums_plus_percussive_fx_aux`; the base DRUM candidate is not modified in place.
The route renders detected events with a teacher-derived spectral mask plus non-detected residual background subtraction;
it must not pass the whole time-domain residual for a detected frame, because that leaks sustained drop/melody content
whenever the clap/riser and melody overlap.

Square Oxygen diagnostic sweep on 2026-05-17:

- benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_percussive_fx_aux_estimator_sweep_benchmark_20260517`;
- compact listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_aux_transient_compare_listen_20260517`;
- result: all aux-estimator candidates were rejected. The best aux-estimator variant,
  `aux_ultra_6s`, scored below the current baseline
  (`reference_likeness=78.159637`, `SDR=15.023612`, `SI-SDR=15.011150` vs baseline
  `reference_likeness=78.276160`, `SDR=15.145527`, `SI-SDR=15.018001`);
- decision: keep the estimator as tested diagnostic infrastructure only. Do not promote its generated candidates.

SCNet aux-source follow-up on 2026-05-17:

- benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_percussive_fx_aux_estimator_scnet_sweep_benchmark_20260517`;
- source aux manifest:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_scnet_xl_ihf_musdb18_20260515/stem_manifest.json`
  using `stems.other`;
- result: all SCNet-derived aux-estimator candidates were rejected and below the unchanged baseline. The best
  variant, `scnet_aux_micro`, scored `reference_likeness=77.986834`, `SDR=14.848062`, `SI-SDR=14.964912`;
- decision: do not deliver SCNet-derived percussive-FX aux candidates as an improvement for this target. The current
  no-reference model stack has no accepted `drums_plus_percussive_fx_aux` candidate above the baseline.

Surgical transient-restore diagnostics on 2026-05-17:

- benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_transient_restore_benchmark_20260517`;
- result: the tested transient restore variants were rejected and below the baseline (`transient_p020`
  `reference_likeness=75.851154`, `SDR=12.847427`, `SI-SDR=14.132738`; `transient_p035`
  `reference_likeness=75.859743`, `SDR=12.854596`, `SI-SDR=14.141527`);
- decision: do not use transient restore for this target without a different transient-specific design.

Percussive event dictionary diagnostics on 2026-05-17:

- tool:
  `ml_worker_improved/tools/build_percussive_event_dictionary_candidate.py`;
- input base:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_from_tonal_stronger_v3_20260517`;
- sweep:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_percussive_event_dictionary_sweep_20260517`;
- compact listen pack:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_percussive_event_dictionary_strict_g010_gatepadfix_compare_listen_20260517`;
- route contract: no-reference application, no target reference at application time, no fallback, no timestamp or
  per-track hardcode. The route extracts a sparse percussive aux layer from the full-mix-minus-base residual and writes
  both `stems.percussive_event_aux` and `stems.drums_plus_percussive_fx_aux`;
- result: all event-dictionary sweep variants were rejected by the objective gate and must not be published as listen
  candidates. The best variant,
  `strict_g010_f86_flat24` (regenerated as `strict_g010_gatepadfix` after inactive-tail gate padding was fixed),
  slightly exceeded the previous tonal-stronger composite baseline
  (`reference_likeness=78.402804`, `SDR=15.260384`, `SI-SDR=15.136247`, `SAR=15.094163` vs baseline
  `reference_likeness=78.276160`, `SDR=15.145527`, `SI-SDR=15.018001`), but it still failed the delivery gate;
- listening rejection: user review confirmed that `03_event_aux_only.flac` from the gatepadfix compare pack contains
  drop/melody residue instead of clap/FX-only content. The regenerated quarantine artifact
  `lab_artifacts/golden_audio/flstudio_square_oxygen_percussive_event_dictionary_strict_g010_quarantined_20260517`
  now records `delivery.status=rejected`, `listen.status=blocked`, `active_fraction=0.569414143`, and
  `low_mid_to_high_ratio=3.716048097`;
- decision: keep this route as a diagnostic failure family only. Do not label the event-dictionary output as accepted
  production audio, and do not publish its listen pack, until the aux sanity gate accepts it and the objective gate also
  passes.

Sparse percussive source diagnostics on 2026-05-17:

- tool:
  `ml_worker_improved/tools/build_sparse_percussive_source_candidate.py`;
- input base:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_drums_plus_percussive_fx_aux_from_tonal_stronger_v3_20260517`;
- rejected broad artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_sparse_percussive_source_v1_20260517`;
- stricter diagnostic artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_sparse_percussive_source_balanced_v4_20260517`;
- strict source route contract: no target reference at application time, no fallback, no timestamp or per-track
  hardcode. The route reads the full mix and the validated no-reference base, selects short high-flux source-event
  islands, renders only `source_transient_islands_minus_base`, and keeps `base_audio_unmodified=true`;
- broad `v1` result was correctly blocked before listen delivery:
  `delivery.status=rejected`, `listen.status=blocked`, `reject_reasons=[aux_not_sparse,harmonic_melody_dominant]`,
  `active_fraction=0.386703145`, `low_mid_to_high_ratio=4.774278582`;
- stricter `balanced_v4` passed the aux sparsity precheck but did not beat the base objective gate:
  its route manifest is `delivery.status=requires_objective_quality_gate`,
  `listen.status=diagnostic_published`, `presentation_status=diagnostic_requires_objective_gate`,
  `no_reference_production_claim=false`, and its objective report remains rejected with
  `reference_likeness=78.267856`, `SDR=15.137113`, `SI-SDR=15.009170` versus base
  `reference_likeness=78.276160`, `SDR=15.145527`, `SI-SDR=15.018001`;
- decision: keep `balanced_v4` as diagnostic evidence only. Do not promote it as a production/listening improvement.
  Passing the aux sanity gate is not enough: the candidate must also beat the base and pass the objective gate.

Safety hardening on 2026-05-17:

- `export_drums_plus_percussive_fx_aux.py` now requires the input base manifest to explicitly declare the no-reference
  contract keys (`target_reference_assisted=false`, `target_reference_used_at_application=false`,
  `application_reference_free=true`, `fallback_used=false`, `per_track_hardcode_allowed=false`,
  `not_timestamp_hardcoded=true`) before it can be relabeled as a derived `drums_plus_percussive_fx_aux` candidate;
- the percussive-event dictionary frame gate now pads missing tail frames as inactive instead of extending the final
  active event, preventing stale event activity from being stretched into the track tail.
- percussive-event dictionary outputs now run an aux sanity gate before listen publication. If the aux layer is not
  sparse enough or is low-mid/harmonic dominated, the manifest sets `delivery_allowed_without_reference=false`,
  `no_reference_production_claim=false`, `delivery.status=rejected`, and `listen.status=blocked`; no listen files are
  copied for that candidate.

`mdx23c_vocals`, `melband_roformer_vocals`, `becruily_karaoke_vocals`, `bs_mamba2_vocals`, and
`audio_separator_bs_roformer_vocals` are registered through
full-track exporters: `ml_worker_improved/tools/export_mdx23c_vocal_full.py`,
`ml_worker_improved/tools/export_melband_vocal_full.py`, and
`ml_worker_improved/tools/export_bs_mamba2_vocal_full.py`, plus
`ml_worker_improved/tools/export_audio_separator_vocal_full.py` for the UVR/Audio Separator BS-RoFormer route. They
run their vocal checkpoints on the full source track and write benchmark-compatible `stem_manifest.json` files with
top-level `stems.vocals` and
`stems.instrumental`. Preview exporters remain diagnostic listen tools and must not be used as production
model-zoo candidate manifests.
The shared MelBand exporter normalizes compatible checkpoint configs that expose `audio.chunk_size` but omit
`inference.chunk_size`, and records the registry model id in the manifest so checkpoint provenance is not collapsed
into `melband_roformer_vocals`.

Registry availability is based on checkpoint/config paths and required Python runtime modules. For example,
`bs_mamba2_vocals` requires `mamba_ssm` and `causal_conv1d`; `audio_separator_bs_roformer_vocals` requires the
isolated `/opt/audio-separator-venv/bin/python` runtime plus the local UVR checkpoint and sidecar YAML under
`/app/models/audio_separator`. Audio Separator is kept out of the main worker Python environment because current
Audio Separator releases require `numpy>=2` and newer `beartype`, while the main worker keeps `numpy<2` for
TensorFlow/Essentia and legacy MSST routes. If any required path or module is absent, the registry must
report the candidate as `unavailable` instead of advertising a runnable route that will fail during inference.
The isolated Audio Separator runtime must include the matching ONNX Runtime package (`onnxruntime` for the CPU image,
`onnxruntime-gpu` for the GPU image), and Docker build smoke imports `audio_separator.separator.Separator` rather
than only the package root so missing runtime dependencies fail at image build time instead of during candidate
generation.
The registered GPU route also passes `--require-cuda-provider`. `export_audio_separator_vocal_full.py` must probe the
isolated runtime before inference and block the candidate when `CUDAExecutionProvider` is not available, even if
PyTorch itself can see the GPU. This prevents a stale or CPU-built image from spending hours on a CPU-only UVR run and
then entering the model router as a normal candidate. The expected persistent model location is the compose-mounted
`./docker_data/models/audio_separator` directory, exposed inside the worker as `/app/models/audio_separator`; ad-hoc
`docker cp` into a running container is diagnostic only and is not a reproducible model-asset contract.
The exporter uses Audio Separator's ffmpeg/pydub writer instead of the `soundfile` writer because MP3 inputs report
`MPEG_LAYER_III` as the source subtype; passing that subtype to FLAC through `soundfile` creates an invalid
format/subtype combination.
The live GPU vocal ladder includes two Audio Separator UVR vocal checkpoints:
`audio_separator_bs_roformer_vocals` and `audio_separator_melband_roformer_viperx_vocals`. The MelBand ViperX route
uses `/app/tools/export_audio_separator_vocal_full.py` with
`/app/vendor/melband_roformer_viperx_11436/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` and
`--require-cuda-provider`; it is a new registered no-reference candidate, not a delivery shortcut or teacher-profile
audio repair.
When a registered model-zoo vocal candidate has already been produced, `adaptive_vocal_isolation.py --external-only`
can gate that external stem without launching the built-in BS/FT candidate exporters first. This is required for
Ralph-AI delivery control: a new model-stack candidate can be audited and either accepted or blocked even when an
older built-in candidate route is unavailable or fails.
Adaptive vocal runs now emit a durable progress log at `adaptive_vocal_progress.jsonl` and copy the runtime
`execution_contract` into `adaptive_vocal_manifest.json`. The log records stage start/finish events, candidate names,
builder names, elapsed time, and repair-window progress so long CPU-bound DSP scoring is attributable instead of being
treated as a hang. The execution contract must state whether GPU inference was actually expected. For
`--external-only`, `gpu_inference_skipped=true`, `gpu_inference_skip_reason=external_only_precomputed_candidates`, and
`expected_acceleration_mode=external_only_cpu_dsp_scoring`; this is not a silent CPU fallback because model inference
was already completed by external registered candidates. Any route that performs live ML/model candidate generation must
prefer the GPU runtime and fail fast or explicitly mark the candidate blocked when the required accelerator is missing.
Adaptive vocal runs also write `selected_delivery_gate_manifest.json` next to `01_adaptive_selected_vocals.flac` and
link it from `adaptive_vocal_manifest.json.selected_artifacts.selected_delivery_gate_manifest`. Downstream consumers
must read this selected-gate manifest, not only the selected candidate `stem_manifest.json`, before treating selected
audio as deliverable or diagnostic.
When a listener identifies a preferred-but-imperfect instrumental candidate, `build_instrumental_preference_dataset.py`
can convert that preference into training-only artifacts for a future teacher-calibrated selector. The builder consumes
the source mix, the preferred candidate `stem_manifest.json`, optional selected delivery gate blockers, and human defect
labels such as `residual_vocal_bleed` or `instrumental_muffling`; it emits `instrumental_preference_dataset_manifest.json`,
`segments.jsonl`, `feedback.jsonl`, and `instrumental_preference_progress.jsonl`. These artifacts are never listen-pack
results: their delivery contract is `delivery_allowed=false` and
`presentation_status=training_only_not_for_listen_pack`.
`teacher_calibrated_instrumental_selector.py` consumes those training-only artifacts plus precomputed candidate
`stem_manifest.json` files. It calculates segment-level proxy features from the source mix and candidate instrumentals,
including presence/air preservation and, when candidate `vocals` stems are present in the manifest, vocal-stem leakage
proxies (`vocal_stem_corr_abs`, `vocal_projection_db_vs_instrumental`). For combined `residual_vocal_bleed` +
`instrumental_muffling` defects, the selector uses a balanced preservation target and penalizes both over-aggressive
muffling and vocal-correlated residue instead of selecting the most suppressed candidate by RMS alone. It applies feedback
priors and writes a proxy-ranked `segment_selection_plan.jsonl`; with `--render-audio` it can splice a
diagnostic `teacher_calibrated_instrumental.flac` from the selected candidate windows. This remains proxy evidence only:
the selector manifest sets `delivery_allowed=false`,
`presentation_status=diagnostic_selector_not_for_listen_pack`, and `requires_downstream_quality_gate=true`.
Human A/B feedback can be written as `instrumental_segment_ab_feedback` records with `segment_id`, `preferred_profile`,
`rejected_profile`, `human_ab_verdict`, and `confidence`; the selector applies these as segment-local priors so a
validated listening verdict can override a proxy-only segment choice without becoming a per-track timestamp hardcode.
Rendered selector FLACs preserve the base candidate FLAC subtype when supported and record that write contract in the
selector manifest.
`instrumental_diagnostic_comparison_gate.py` is the no-reference comparison layer for that diagnostic selector output:
it compares a baseline instrumental against a tested instrumental over the preference dataset windows, reports per-segment
proxy deltas and an overall `candidate_proxy_improved`, `mixed_or_inconclusive`, or `candidate_proxy_regressed` status.
For combined residual-vocal + muffling defects, the no-reference comparison is preservation-first because it has no
candidate vocal stem leakage proxy; it must not reward a lower-RMS candidate only because it is more suppressed.
The comparison is not an acceptance gate; it preserves `delivery_allowed=false` and
`presentation_status=diagnostic_comparison_not_for_listen_pack` until a reference-backed or human listening gate accepts
the candidate.
`instrumental_self_judge_gate.py` is the automated self-check used before asking for any listening verdict. It scores a
single instrumental/minus artifact over dataset or uniform windows, flags muffling/presence/air blockers, and can use
either an explicit candidate vocal stem or the selected-candidate vocal features stored in a
`teacher_calibrated_selector_manifest.json` to flag residual vocal proxies. It writes `proxy_pass` or `proxy_blocked` plus
ranked blocker windows. This is still no-reference proxy evidence and always keeps `delivery_allowed=false`; if it blocks,
the next action is model routing or repair iteration, not handing the file off as a result.
`build_diagnostic_ab_listening_pack.py` is the only supported route for turning those comparison windows into diagnostic
A/B clips. Its single clips must not apply gain, normalization, limiting, or fades; they preserve each source FLAC subtype
and record max roundtrip sample deltas in `diagnostic_ab_listening_manifest.json`. Ad hoc A/B exports that rewrite the
baseline/candidate representation without this manifest contract are diagnostic drift evidence only and must not be used
for human verdicts.
Instrumental Resurrection minus exports now emit the same durable progress evidence class before they are used as IR
residual inputs: `instrumental_resurrection_progress.jsonl` records execution contract, transcode, model inference,
artifact-write, listen-pack, and finish events, and `stem_manifest.json.execution_contract` records
`expected_acceleration_mode` plus the progress log path. Each JSONL progress event is flushed and fsynced before the
next phase continues. GPU-capable exports must use `gpu_inference`; CPU mode remains diagnostic and is recorded as
`cpu_diagnostic_inference`.
Built-in adaptive vocal runtime checks are profile-scoped: `ft_tail_safe`/`ft_vocals` must not require the BS-RoFormer
vendor tree, while `hybrid_vocals` and `bs_vocals` still require `/app/vendor/music_source_separation_training` plus
the configured BS-RoFormer config/checkpoint. Missing BS assets must block only the profiles that need them, not a
Demucs-only vocal-capture run.
External candidates are required to cover the requested analysis window. A short preview stem must be rejected for a
full-track run instead of being silently length-matched and allowed to win proxy scoring against full-track candidates.
Segment issue classification treats windows where both the source and candidate are below the near-silent RMS floor as
`acceptable_proxy_window` instead of deriving bleed/full-mix labels from undefined ratio metrics. Source-silent windows
with an active candidate remain actionable leakage and must stay blocked by the delivery gate.
The defect-effectiveness gate and issue-effectiveness ledger apply the same both-silent filter when reading precomputed
issue reports, so stale severe ratio labels from a near-silent tail cannot become a hard blocker.
Issue-aware segment selection compares donor windows against shared/base track metrics before routing. A globally degraded
candidate must not look locally clean just because its own track-level dropout/tail baseline is already worse than the
candidate being repaired.
Segment issue repair may select support stems from the available candidate stack when the preferred internal support
profiles are not present. This selection is metric-driven (`adaptive_low_issue_support`) and recorded in the candidate
manifest; it is not a delivery fallback and the repaired output must still pass the same defect-effectiveness gate.
Adaptive vocal final selection is defect-aware when segment issue reports exist: it prefers lower measured
`max_issue_score`, then `mean_top3_issue_score`, then `issue_window_ratio`, and only uses proxy score as a tie-breaker.
This prevents a higher proxy-score candidate from hiding a worse measured vocal defect. It does not relax delivery:
blocked candidates remain `diagnostic_only` and cannot be copied into the accepted listen pack.
The defect-effectiveness gate reports both `raw_issue_window_ratio` and material `issue_window_ratio`. Raw ratio keeps
all detected labels visible for forensic review, while the blocker ratio counts only windows whose severity is at least
`material_segment_score`; this prevents tiny detector residue labels from causing endless repair loops after severe and
clustered issue scores have already been reduced. Severe-window and mean-top3 blockers still use the measured severity
scores directly.
Adaptive vocal issue reports include `drop_melody_leakage` for sustained low-mid tonal residue in a vocal stem. This is
separate from `kick_punch_bleed`: it targets audible drop/melody content that survives between percussive events. The
detector uses low-mid tonal stability metrics (`700-1800 Hz` flatness/flux/peak concentration) so stable vocal-presence
tones are not treated as melody leakage by themselves. The label is mapped to the same action contract as other
non-vocal leakage labels and must either produce a measured repair candidate or keep the output out of accepted listen
packs.

Low-energy `drop_melody_leakage` can also produce a `drop_melody_only_residue` cleanup decision when the candidate is
quiet, narrowband, and lacks strong vocal presence evidence. That exception is intentionally narrower than generic
no-vocal cleanup: it does not override backing-vocal risk, and it remains subject to the delivery gate. If MDX/SCNet/
hybrid donors and cleanup routes still leave `drop_melody_leakage`, the result remains diagnostic-only under RALPH-AI
instead of being presented as the least-bad listen output.
If the defect-effectiveness gate still blocks the selected vocal after post-gate repair, the controller can build
`gate_blocker_best_segment_splice`: a no-reference derived candidate that replaces only gate-blocker windows with the
best objectively measured donor segment from the current candidate stack. This is driven by `defect_effectiveness_gate`
blockers and candidate issue reports, not by track names or fixed timestamps. The spliced candidate must still be
re-scored, re-reported, and pass the same delivery gate before it can appear in an accepted listen pack. Donor selection
is reclassified against the current base track metrics before splicing, so a globally weak donor cannot look clean only
because its own dropout baseline is already worse.
When direct blocker splicing has no safe measured donor, the controller can build
`gate_blocker_presence_band_recovery`: a no-reference derived candidate that keeps the selected base stem, adds only a
band-limited donor presence delta, and lightly attenuates the base bleed band inside measured gate-blocker windows. The
route must predict the segment issue score from the actual source/candidate waveform before applying a window; windows
without measured improvement are skipped. It remains a candidate only: the output is re-scored, gets a fresh segment
issue report, and cannot enter an accepted listen pack unless the normal RALPH-AI delivery gate passes.
If the selected vocal is still blocked by clustered issue windows or an excessive issue-window ratio, the controller can
build `clustered_issue_window_ratio_repair`: a no-reference derived candidate that applies only label-matched,
band-limited reductions inside windows whose issue score is predicted to improve from the actual source/candidate
waveform. The route compares base and repaired windows with the same local scoring basis, skips windows that introduce
vocal-loss labels, tests multiple reduction strengths for persistent clustered leakage, then re-scores and re-gates the
whole candidate. Its band component uses the route-local `band_filter_method` tuning (`fft` by default for bounded
full-track diagnostic runtime, `stft` available for compatibility). It is not a delivery bypass; rejected outputs remain
diagnostic-only and cannot enter an accepted listen pack.
If that first clustered repair remains diagnostic-only with a clustered issue-ratio blocker, the controller may build one
bounded second-pass candidate, `clustered_issue_window_ratio_repair_pass2`, from the first-pass output. The second pass
uses its own manifest profile, dependency-fingerprint context, skip ledger, candidate score, issue report, and the same
RALPH-AI delivery gate; it is allowed only as measured iteration over a persistent blocker, not as a fallback or listen
pack shortcut.
The delivery gate also blocks globally unstable vocal candidates even when the final issue-window scores fall below the
clustered thresholds. Segment-consistency metrics such as global dropout, worst-window dropout/tail loss, and wide
segment RMS range create `selected_candidate_has_unstable_vocal_consistency`; this keeps muffled, randomly dipping, or
gap-heavy vocals out of accepted listen packs instead of letting a locally improved but globally damaged candidate pass.
For an `--external-only` adaptive vocal run this blocker now creates `vocal_next_action_plan.status=
requires_new_model_candidates`, `next_action=run_live_gpu_vocal_candidate_ladder`, and
`forbid_cpu_dsp_repair_loop=true`. Post-gate CPU/DSP repair candidates are then skipped with
`reason=live_gpu_vocal_ladder_required` instead of spending another pass on the same unstable candidate stack. The
recommended live model ladder is restricted to registered vocal routes such as `audio_separator_bs_roformer_vocals`,
`mdx23c_vocals`, `scnet_xl_ihf_musdb18`, `melband_roformer_vocals`, `htdemucs_ft`, `htdemucs_6s`, and
`hybrid_htdemucs_ft_bs_roformer_6s`; the selected output still remains blocked until the normal delivery gate accepts
it.
If all registered live GPU ladder routes were already evaluated and the same global instability blocker remains, the
plan must switch to `status=candidate_space_exhausted` with
`next_action=escalate_to_new_vocal_route_or_teacher_calibration`. This prevents a circular instruction to rerun the
same ladder and keeps the next step explicit: register a genuinely new GPU vocal route, run teacher-calibrated vocal
selection, or add a reference holdout instead of looping through the same failed candidate set.
The exhausted check is based on registered model identity, not a loose adaptive profile alias. The plan records
`evaluated_model_ids` from exact ladder profile names or from candidate manifest provenance such as `pipeline.model_id`,
`pipeline.registered_model_id`, `pipeline.source_model_id`, `pipeline.source_candidate_path`, or
`stems.vocals.source_candidate_path`. Built-in adaptive aliases such as `ft_tail_safe` do not consume the registered
ladder unless their manifest points back to a registered route. `recommended_model_ids` must therefore mean genuinely
unevaluated registered routes.
Detected vocal issue actions also carry an `action_family` so the router can distinguish reconstruction problems
(`vocal_presence_reconstruction`, `vocal_body_recovery`) from non-target rejection problems
(`melody_leakage_rejection`, `percussive_bleed_rejection`) and protective vocal-content labels
(`protect_vocal_content`). This is the bridge from generic issue labels toward defect-specific vocal reconstruction:
dropout/tail/body defects must not be treated as no-vocal suppression evidence.
The Docker image builds `mamba-ssm==2.3.2.post1` from source with `--no-deps --no-build-isolation` after
PyTorch is installed, so pip cannot replace the pinned torch/CUDA runtime while enabling BS-Mamba2. The Dockerfile
performs only a package-availability smoke at build time; full Triton/Mamba imports require an active GPU driver and
must be validated by runtime route probes instead of `docker build`.

Required module presence is only the first gate for BS-Mamba2. A BS-Mamba2 Docker/runtime stack must also pass
`run_registered_model_candidate.py` on a bounded smoke fixture and create `stem_manifest.json` before the candidate
can be treated as production-available. The following local probes were rejected because inference reached model
load and then crashed with `SIGSEGV` before manifest creation:

| Stack | Probe artifact | Status |
| --- | --- | --- |
| `torch==2.5.1+cu124`, `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1` | `lab_artifacts/quality_benchmark/runtime_probe/bs_mamba2_causal162_fixture_probe` | rejected |
| `torch==2.6.0+cu126`, `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1` | `lab_artifacts/quality_benchmark/runtime_probe/bs_mamba2_gpu260_causal162_fixture_probe4` | rejected |
| `torch==2.6.0+cu126`, `mamba-ssm==2.2.6.post3`, `causal-conv1d==1.5.3.post1` | `lab_artifacts/quality_benchmark/runtime_probe/bs_mamba2_gpu260_m226_fixture_probe` | rejected |
| `torch==2.6.0+cu126`, `mamba-ssm==2.3.0`, `causal-conv1d==1.5.3.post1` | `lab_artifacts/quality_benchmark/runtime_probe/bs_mamba2_gpu260_m230_fixture_probe` | rejected |
| `torch==2.6.0+cu126`, `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.5.3.post1` | `lab_artifacts/quality_benchmark/runtime_probe/bs_mamba2_gpu260_m232_causal153_fixture_probe` | rejected |

Do not add any of these stacks to the production Dockerfile as an enabled BS-Mamba2 route.

Use `ml_worker_improved/tools/list_model_registry.py` to inspect available candidates or ask for bounded next
candidate suggestions after a failed benchmark:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/list_model_registry.py `
  --stem vocals `
  --failed-label reference_likeness_below_threshold `
  --failed-label vocal_dropout `
  --evaluated-label mdx23c
```

The registry must not contain track names, timestamps, or hand-tuned per-track thresholds.

## Registered candidate executor

`ml_worker_improved/tools/run_registered_model_candidate.py` runs one `ModelRegistryEntry` command in the Docker
`ml-worker` runtime and verifies that the registered `expected_manifest` was actually created. This tool is the
execution bridge between benchmark `next_action.action=generate_registered_model_candidates` and the next
`run_model_zoo_benchmark.py` pass.

The executor writes `registry_candidate_execution_manifest.json` inside the candidate output directory for every
terminal route:

- `status=completed`, `reason=candidate_manifest_created`: the registered command ran and the expected candidate
  manifest exists;
- `status=unavailable`, `reason=missing_required_paths`: required checkpoints/assets are missing, so the command
  is not run;
- `status=failed`, `reason=source_audio_missing`: source audio is missing;
- `status=failed`, `reason=execution_failed`: the registered command/runner failed before producing the expected
  manifest; the error type/message are recorded in `error`, and the default subprocess runner stores
  `registered_candidate_stdout.log` and `registered_candidate_stderr.log` paths in `logs`;
- `status=failed`, `reason=expected_manifest_missing`: the command returned but did not create the registered
  manifest, so the candidate must not be benchmarked as a completed output.

Executor contract:

- `fallback_used=false`;
- `hidden_inference=false`;
- `registered_model_required=true`;
- `expected_manifest_required=true`;
- `per_track_hardcode_allowed=false`.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/run_registered_model_candidate.py `
  --model-id htdemucs_6s `
  /app/testing_flstudio_music/source.flac `
  /app/lab_artifacts/golden_audio/example_registry_candidate/htdemucs_6s
```

Use the generated `{output_dir}/stem_manifest.json` as the next `LABEL=PATH` candidate input for
`run_model_zoo_benchmark.py`.

## Reference-driven model-zoo benchmark

`ml_worker_improved/tools/run_model_zoo_benchmark.py` ranks multiple model/candidate manifests against the same
verified ground-truth bundle. It is the repeatable harness for answering whether a new model stack is actually
better than the current one.

Contract:

- input candidates are `LABEL=PATH` references to existing `candidate_manifest.json` files;
- the CLI is Docker-only and fail-fast outside the container runtime;
- the benchmark does not run hidden inference and does not synthesize fallback audio;
- `contract.reference_assisted=true`;
- `contract.fallback_used=false`;
- `contract.acceptance_requires_delivery_accepted=true`;
- `selected_candidate` is populated only when the underlying objective report has `delivery.status=accepted`;
- candidate summaries include `forensic_contract_status`, `forensic_uncovered_labels`, `forensic_issue_actions`,
  `forensic_evidence_status`, and `forensic_evidence_dir`;
- model-zoo runs write per-candidate waveform forensic evidence packs under
  `forensic_evidence/<candidate>/<stem>/` so `blocked_from_delivery` candidates have residual audio, trace data,
  waveform overlay, and residual histogram for audit without being promoted as listen results;
- `failed_check_names` is a blocking-failure field: accepted candidates keep it empty even when raw diagnostic
  `quality_windows` or `micro_quality_windows` still contain non-blocking ledger labels for audit;
- when every candidate is rejected but the best diagnostic no-reference candidate has mapped forensic issue actions,
  available evidence, and enough signal quality for another pass, `next_action.action` is
  `repair_best_ranked_candidate`; the embedded `repair_plan` names the base candidate, target labels, issue actions,
  forensic evidence directory, and guardrails (`accepts_rejected_base_as_final=false`, `requires_rebenchmark=true`);
  eligible repair-chain base identifiers include SCNet candidates and no-reference DRUM repair routes such as
  `drum_bus_projection_body_gain`, `drum_micro_leakage_debleed`, `drum_tonal_leakage_repair`, and
  `drum_transient_restore`;
- when every candidate is rejected and no repair base is eligible but the registry has relevant unevaluated candidates,
  `next_action.action` is `generate_registered_model_candidates`;
- when every candidate is rejected and no registered candidate remains, `next_action.action` is
  `reject_delivery_candidate_space_exhausted`;
- the benchmark is reference-assisted and must not be described as no-reference production routing.
- the benchmark suggests bounded next candidates from `model_registry.py`; it does not execute them automatically.
- `--next-candidate-limit` bounds the number of registry suggestions emitted by one benchmark pass.
- `--available-registry-candidates-only` hides unavailable registry entries from `next_action.candidate_suggestions`.
- evaluated registry candidates are de-duplicated by stable identifiers from the candidate manifest (`model_id`,
  `model_name`, or nested `model.model_id`) plus registry aliases, not only by the user-facing `LABEL`.
- `--require-no-reference-candidates` is a production-safety hard gate. In this mode a candidate is rejected with
  `candidate_contract_not_no_reference_safe` unless its manifest proves `target_reference_assisted=false`,
  `target_reference_used_at_application=false`, `application_reference_free=true`, `fallback_used=false`,
  `per_track_hardcode_allowed=false`, and `not_timestamp_hardcoded=true`. Reference-assisted profile candidates also
  need `holdout_validation_status=passed`; calibration-only profile artifacts cannot be promoted by this benchmark mode.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/run_model_zoo_benchmark.py `
  --ground-truth-manifest /app/lab_artifacts/golden_audio/<bundle>/ground_truth_manifest.json `
  --candidate-manifest htdemucs_6s=/app/lab_artifacts/golden_audio/<run>/htdemucs_6s/candidate_manifest.json `
  --candidate-manifest bs_roformer=/app/lab_artifacts/golden_audio/<run>/bs_roformer/candidate_manifest.json `
  --stem drums `
  --output-dir /app/lab_artifacts/golden_audio/<run>/model_zoo_benchmark
```

The tool writes:

- `model_zoo_benchmark_manifest.json`;
- `objective_reports/<candidate>_objective_report.json` for every candidate.

Each stem report also includes:

- `reference_likeness`, a 0-100 composite similarity report against the verified reference stem. The default gate
  requires `score_percent >= 95.0` and includes component scores for waveform shape, spectrum, energy envelope,
  transient envelope, and leakage cleanliness;
- `quality_windows`, a deterministic 5-second worst-window ledger for localizing audible failures;
- `micro_quality_windows`, an overlapping 100/250/500 ms forensic ledger for short residual spikes, transient
  mismatches, transient dropouts, and short-window reference-stem leakage. It also includes a 1/2/4 second
  phrase-level ledger for `bar_spanning_drop_melody_leakage`: sustained non-target harmonic/drop content that remains
  audible between kicks across beat/bar/phrase windows. This layer is diagnostic and does not change objective SDR
  thresholds by itself, but its `issue_label_counts` are delivery-contract inputs; it records `analyzed_window_count`,
  `phrase_analyzed_window_count`, `defect_window_count`, bounded top-12 `worst_windows`, and bounded
  `phrase_level_windows`, plus `phrase_level_windows_by_time_bucket` and `worst_phrase_windows_by_source`, so both
  short spikes and longer drop-melody residues cannot be hidden inside the aggregate 5-second ledger;
- `waveform_forensics`, a sample-aligned residual report with polarity audit, residual peak/RMS, absolute-error
  percentiles, and top residual defect ranges. This is the first forensic layer for finding exactly where a candidate
  waveform diverges from the verified stem; it complements the aggregate `reference_likeness` score and does not
  replace the delivery gate by itself. The standalone tool
  `ml_worker_improved/tools/waveform_forensic_compare.py` can also write a reproducible evidence pack with
  `*_waveform_forensic_report.json`, `*_waveform_trace.npz`, `*_residual_aligned.wav`,
  `*_waveform_overlay.svg`, and `*_residual_histogram.svg`. The trace stores aligned reference/candidate/residual
  arrays; the audio/SVG artifacts are diagnostic evidence and must not be treated as a new candidate stem.
  Offset search is bounded by `MAX_SAFE_OFFSET_SAMPLES=262144` so a bad CLI argument cannot turn a forensic check
  into an unbounded full-track correlation job;
- `forensic_delivery_contract`, a per-stem defect-to-action contract. Every detected threshold, waveform, macro-window,
  micro-window, or phrase-window issue must map to an action such as `repair_candidate_required`, `reroute_required`,
  `debleed_or_reroute_required`, or `explicit_rejection_reason`. `bar_spanning_drop_melody_leakage` maps to
  `debleed_or_reroute_required` and is also an adaptive repair priority label, so a rejected no-reference DRUM
  candidate with that evidence can continue through `repair_best_ranked_candidate` instead of being presented as clean.
  The report-level contract is also copied into `delivery.forensic_contract_status`; a candidate cannot be delivered as
  accepted when this contract is rejected;
- `leakage_matrix`, named projection/correlation evidence against every available atomic reference stem, even when
  only one candidate stem is being evaluated. Derived `instrumental` references are excluded unless explicitly
  requested to avoid double-counting child stems.
- `leakage_strength`, an energy-normalized leakage score used for issue labels, while raw `projection_gain` remains
  in the report for audit/debugging.

Use `--min-reference-likeness-percent` to tighten or relax the default 95 percent likeness gate. This does not make a
fake 1:1 claim; it is a repeatable rejection rule that combines reconstruction, timbre, envelope, transient, and
named-leakage evidence.

`route_drum_model_zoo_segments.py` marks routed candidates with `quality_gate_required=true` and
`delivery_status=requires_objective_quality_gate`. A routed file is therefore a candidate until a verified
ground-truth gate accepts it. The router also supports a generic `--stem-name` for supervised stem-candidate
routing; the legacy DRUM route keeps `route=drum_model_zoo_segment_router_v1`, while non-drum stems use
`route=stem_model_zoo_segment_router_v1`. If `--output-label` is omitted, the output filename is derived from
the requested stem, for example `06_segment_model_router_piano.flac`. Derived drum-like targets such as
`drums_plus_percussive_fx_aux` use the DRUM score profile and `route=drum_model_zoo_segment_router_v1`; they must
not fall back to the generic profile because that over-ranks non-drum/harmonic candidates during reference
diagnostics.

`drum_micro_leakage_debleed.py` supports `--stem-name` for benchmark-compatible manifests. The default remains
`drums`; derived drum targets must pass `--stem-name drums_plus_percussive_fx_aux` so `stem_manifest.json`
contains `stems.drums_plus_percussive_fx_aux` instead of silently emitting `stems.drums`. Evidence modes still
control delivery safety: `source_residual_proxy` is application-reference-free, while `verified_reference_stems`
is diagnostic/reference-assisted and must not be claimed as no-reference production output.

2026-05-14 update: the router now records an explicit `stem_score_profile` in both route and candidate manifests.
The `vocals` profile is reference-free at application time and adds harmonic/body support, named-leakage penalty,
unsupported high-band noise penalty, and vocal-aware quiet handling so that no-vocal sections are not automatically
treated as vocal dropouts. Manifest contracts include `target_reference_assisted=false`,
`target_reference_used_at_application=false`, `application_reference_free=true`,
`per_track_hardcode_allowed=false`, `fallback_used=false`, and `no_reference_production_claim=false` when no
`--target-ref` is provided. This remains a candidate-generation route; it is not a delivery claim.

Square Oxygen diagnostic run:

- artifacts:
  - `flstudio_square_oxygen_reference_free_segment_router_20260514/drums/candidate_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_20260514/vocals/candidate_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_benchmark_20260514/drums/model_zoo_benchmark_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_benchmark_20260514/vocals/model_zoo_benchmark_manifest.json`.
- DRUM benchmark result: `reference_free_router` was rejected with `reference_likeness=79.121884`,
  `SDR=10.334267`, `SI-SDR=10.041559`, `SIR=31.063126`, `SAR=10.099211`; it underperformed the existing
  `tonal_strong` rejected diagnostic candidate (`reference_likeness=90.734160`, `SDR=14.833282`,
  `SI-SDR=14.688647`).
- VOCALS benchmark result: `reference_free_router` was rejected with `reference_likeness=78.516287`,
  `SDR=8.001301`, `SI-SDR=7.267613`, `SIR=33.162719`, `SAR=7.283511`; it underperformed the existing `melband`
  rejected diagnostic candidate (`reference_likeness=82.603726`, `SDR=9.166258`, `SI-SDR=8.605978`).

Decision: keep the stem-aware reference-free segment router as plumbing and diagnostic evidence only. It solves the
contract/scoring gap caught by synthetic tests, but the Square Oxygen objective gate proves that this no-reference
ranking is not yet strong enough to replace the best single candidates. Do not present its DRUM/VOCALS outputs as
accepted listening results.

2026-05-14 guard update: the router can now take a validated baseline candidate through
`--baseline-candidate-label`, require a minimum no-reference confidence margin via `--baseline-switch-margin`, and
limit the number of allowed switches with `--baseline-max-switch-fraction`. The per-segment `selection_guard` records
`confidence_metric=score_delta_vs_baseline`, `no_reference_confidence_score`, margin/budget decisions, and whether a
switch was blocked. The guard is disabled in `--target-ref` mode so supervised reference routing is not silently
overridden by reference-free priors.

Guarded Square Oxygen diagnostic run:

- artifacts:
  - `flstudio_square_oxygen_reference_free_segment_router_guarded_budget_20260514/drums/candidate_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_guarded_budget_20260514/vocals/candidate_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_guarded_budget_benchmark_20260514/drums/model_zoo_benchmark_manifest.json`;
  - `flstudio_square_oxygen_reference_free_segment_router_guarded_budget_benchmark_20260514/vocals/model_zoo_benchmark_manifest.json`.
- DRUM guarded budget route used `baseline_candidate_label=tonal_strong`, `baseline_switch_margin=2.0`,
  `baseline_max_switch_fraction=0.05`. It was rejected with `reference_likeness=89.066721`, `SDR=14.050291`,
  `SI-SDR=13.879459`; this is much safer than the unguarded router but still below `tonal_strong`
  (`reference_likeness=90.734160`, `SDR=14.833282`, `SI-SDR=14.688647`).
- VOCALS guarded budget route used `baseline_candidate_label=melband`, `baseline_switch_margin=2.0`,
  `baseline_max_switch_fraction=0.05`. It was rejected with `reference_likeness=82.573290`, `SDR=9.160718`,
  `SI-SDR=8.599720`; this is nearly baseline-equivalent but still slightly below `melband`
  (`reference_likeness=82.603726`, `SDR=9.166258`, `SI-SDR=8.605978`).

Decision: the guard prevents the worst no-reference over-switching failure and makes router decisions auditable, but
it still must not be used as an accepted stem generator until a candidate beats the validated baseline under the
objective gate. For delivery, prefer the best validated single candidate unless a guarded routed candidate is accepted
by the benchmark.

2026-05-14 confidence calibration update: `calibrate_segment_router_confidence.py` now derives
`baseline_switch_margin` and `baseline_max_switch_fraction` from a verified ground-truth route manifest instead of
hand-picking them. The calibration artifact is target-reference-assisted and writes
`target_reference_assisted=true`, `calibration_only=true`, `same_track_delivery_allowed=false`,
`application_uses_target_reference=false`, `holdout_validation_required=true`, and
`no_reference_production_claim=false`. The router can load the resulting profile through
`--baseline-calibration-profile`; application remains reference-free and records
`baseline_calibration_profile_path` in both route and candidate manifests.

Square Oxygen confidence-profile diagnostic run:

- calibration artifacts:
  - `flstudio_square_oxygen_segment_router_confidence_calibration_20260514/drums/segment_router_confidence_calibration_manifest.json`;
  - `flstudio_square_oxygen_segment_router_confidence_calibration_20260514/vocals/segment_router_confidence_calibration_manifest.json`.
- applied artifacts:
  - `flstudio_square_oxygen_segment_router_confidence_profile_applied_20260514/drums/candidate_manifest.json`;
  - `flstudio_square_oxygen_segment_router_confidence_profile_applied_20260514/vocals/candidate_manifest.json`;
  - `flstudio_square_oxygen_segment_router_confidence_profile_applied_benchmark_20260514/drums/model_zoo_benchmark_manifest.json`;
  - `flstudio_square_oxygen_segment_router_confidence_profile_applied_benchmark_20260514/vocals/model_zoo_benchmark_manifest.json`.
- DRUM calibration learned `baseline_candidate_label=tonal_strong`, `baseline_switch_margin=9.911425`,
  `baseline_max_switch_fraction=0.0038461538461538464`, with `predicted_switch_count=1`, `precision=1.0`,
  `recall=0.0196078431372549`, and `helpful_switch_count=51`. Applied reference-free routing selected
  `tonal_strong` for 259 segments and `htdemucs_ft` for 1 segment. The candidate was rejected with
  `reference_likeness=90.759354`, `SDR=14.849434`, `SI-SDR=14.705460`, and `SAR=14.818028`.
- VOCALS calibration learned `baseline_candidate_label=melband`, `baseline_switch_margin=68.555420`,
  `baseline_max_switch_fraction=0.0`, with `predicted_switch_count=0`, `precision=0.0`, `recall=0.0`, and
  `helpful_switch_count=3`. Applied reference-free routing selected `melband` for all 260 segments. The candidate was
  rejected with `reference_likeness=82.603726`, `SDR=9.166258`, `SI-SDR=8.605978`, and `SAR=8.616719`.

Decision: the calibration layer prevents manual guard tuning and exposes whether no-reference confidence actually
correlates with reference improvement. On Square Oxygen, DRUM gains are measurable but still far below the 95%
delivery gate, and VOCALS has no safe no-reference switch at `min_precision=0.95`. These profiles are diagnostic
training artifacts until validated on held-out tracks.

2026-05-14 holdout validation runner update: `validate_segment_router_confidence_holdout.py` applies a saved
confidence profile to a candidate set without `--target-ref`, runs `run_model_zoo_benchmark.py` against a verified
ground-truth manifest, and writes `holdout_validation_manifest.json` with `validation.status=validated` or
`validation.status=rejected_for_generalization`. Its contract records `target_reference_used_at_application=false`,
`application_reference_free=true`, `reference_assisted_evaluation=true`, `requires_distinct_holdout_track=true`,
`fallback_used=false`, and `quality_gate_required=true`.

Current local artifact is a same-track contract smoke because only one real verified FL Studio ground-truth manifest
is present under `lab_artifacts/golden_audio` (`flstudio_ground_truth_square_oxygen_20260511`). It proves runner
plumbing and rejection behavior, not cross-track generalization:

- artifacts:
  - `flstudio_square_oxygen_segment_router_confidence_holdout_validation_same_track_smoke_20260514/drums/holdout_validation_manifest.json`;
  - `flstudio_square_oxygen_segment_router_confidence_holdout_validation_same_track_smoke_20260514/vocals/holdout_validation_manifest.json`.
- DRUM same-track smoke: `validation.status=rejected_for_generalization`, `reference_likeness=90.318221`,
  `SDR=14.262706`, `SI-SDR=14.097249`; application was reference-free
  (`target_reference_used_at_application=false`).
- VOCALS same-track smoke: `validation.status=rejected_for_generalization`, `reference_likeness=82.395330`,
  `SDR=8.987309`, `SI-SDR=8.402687`; application was reference-free
  (`target_reference_used_at_application=false`).

Decision: no confidence profile can be promoted until this runner validates it on at least one distinct verified
ground-truth track. A same-track smoke must remain a diagnostic artifact even when the command exits with status 0.

The synthetic smoke summary uses `overall.control_gate=perfect_passes_degraded_fails`; it is a plumbing
control, not a production quality claim.

## Registered no-reference JOINT VOCAL/MELODY v2 route

`ml_worker_improved/tools/build_joint_vocal_melody_candidate_v2.py` adds `joint_vocal_melody_v2` as a diagnostic
successor to `joint_vocal_melody_v1` after exact-reference audit showed both vocals and melody still fail the
Square Oxygen references. The route keeps the no-reference application contract:

- source input: authoritative `source_mix.flac`;
- base input: a no-reference model-stack manifest that passes the required contract flags;
- verified target/reference stems are not read at application time;
- `reference_sum_mix.flac` is not read;
- source-pool spectral partitioning is applied to the melody branch by default;
- the vocal branch is not source-pool-replaced by default (`vocal_pool_weight=0.0`,
  `vocal_non_target_suppression=0.0`) because the first v2 pass damaged vocal exact-reference quality;
- listen artifacts remain diagnostic until objective gate and exact-reference audit both accept them.

Square Oxygen v2 checkpoints on 2026-05-18:

- first v2 source-pool artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v2_source_pool_20260518`;
- first v2 benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v2_source_pool_benchmark_20260518`;
- vocal-preserving v2 artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v2_vocal_preserve_20260518`;
- vocal-preserving v2 benchmark:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v2_vocal_preserve_benchmark_20260518`.

Status: rejected by exact-reference audit, not production delivery.

- `joint_v1_adaptive`: mean reference likeness `54.1694385`; vocals exact SDR `6.612339`,
  cosine `88.438208`; melody exact SDR `-1.315371`, cosine `31.453594`;
- `joint_v2_source_pool`: mean reference likeness `52.494689`; vocals exact SDR `6.146467`,
  cosine `87.021447`; melody exact SDR `-1.147032`, cosine `31.494479`;
- `joint_v2_vocal_preserve`: mean reference likeness `52.399099`; vocals exact SDR `6.115054`,
  cosine `86.943976`; melody exact SDR `-1.147032`, cosine `31.494475`.

Decision: v2 reduced melody residual peak versus v1 (`1.325702 dBFS` to about `0.4485 dBFS`) but did not improve
delivery quality enough and worsened vocal exact-reference metrics. It must stay diagnostic. The next production-grade
step is a stronger model-stack candidate for melody/vocals, not further promotion of this v2 output.

## Registered no-reference JOINT VOCAL/MELODY v3 proxy router

`ml_worker_improved/tools/build_joint_vocal_melody_candidate_v3.py` adds `joint_vocal_melody_v3` as a multi-candidate
no-reference route. It consumes multiple no-reference model-stack manifests for the same `source_mix.flac`, validates
their contracts and source hashes, scores stems independently by proxy leakage, then writes separate `vocals` and
`melody` outputs. It does not read verified target references at application time.

Contract:

- source input: authoritative `source_mix.flac`;
- base inputs: two or more no-reference candidate manifests for the same source hash;
- verified target/reference stems are not read at application time;
- `reference_sum_mix.flac` is not read;
- vocals and melody can come from different candidate manifests;
- output remains diagnostic until objective gate and exact-reference audit both accept it.

Square Oxygen v3 checkpoint on 2026-05-18:

- artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v3_proxy_router_20260518`;
- listen artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v3_proxy_router_listen_20260518`;
- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_v3_proxy_router_benchmark_20260518`;
- selected proxy sources: vocals from `joint_vocal_melody_v1`, melody from `joint_vocal_melody_v2`;
- status: rejected by exact-reference audit, not production delivery.

RALPH-AI re-gate on 2026-05-18:

- benchmark artifact:
  `lab_artifacts/golden_audio/flstudio_square_oxygen_joint_vocal_melody_ralph_ai_gate_20260518`;
- overall status: `no_accepted_candidate`;
- selected candidate: `null`;
- all evaluated joint candidates have `presentation_status=blocked_from_delivery`;
- each rejected candidate has `ralph_ai_delivery_contract.status=blocked` and `result_delivery_allowed=false`.

Measured comparison:

- `joint_v1_adaptive`: mean reference likeness `54.1694385`; vocals exact SDR `6.612339`,
  cosine `88.438208`; melody exact SDR `-1.315371`, residual peak `1.325702 dBFS`;
- `joint_v2_vocal_preserve`: mean reference likeness `52.399099`; vocals exact SDR `6.115054`,
  cosine `86.943976`; melody exact SDR `-1.147032`, residual peak `0.448484 dBFS`;
- `joint_v3_proxy_router`: mean reference likeness `53.416235`; vocals exact SDR `6.593442`,
  cosine `88.384972`; melody exact SDR `-1.149307`, residual peak `0.417900 dBFS`.

Decision: v3 proves stem-specific candidate routing is functioning and improves melody residual peak versus v1/v2, while
keeping vocals close to v1. It still fails the exact-reference delivery gate by a large margin, especially melody
cosine near `31%`. Do not promote this route as a production stem generator. The remaining gap is model quality, not
only routing: the project needs a stronger melody/vocal model stack or training/teacher-distillation path before a
95% exact-reference claim is credible.

## Interpretation

Passing this benchmark means the current detector layer can identify the synthetic defects it claims
to detect. It does not prove production stem quality on arbitrary commercial tracks. Real-track
acceptance still requires listen packs and, when available, verified isolated ground truth.
