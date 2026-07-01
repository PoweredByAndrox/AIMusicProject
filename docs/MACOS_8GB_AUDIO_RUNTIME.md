# macOS 8GB Audio Runtime

This project can use a Mac as a constrained audio experiment host, but the
runtime contract is different from the current CUDA Docker path.

## Source-of-truth position

- Existing production separator routes are audio routes: Demucs, MDX/MDX23C,
  BS-RoFormer, MelBand/Mel-RoFormer, SCNet, and registered project adapters.
- Text vLLM/Qwen is not an audio separator and must not replace those routes.
- Apple `container` tooling is the approved Mac container orchestration route.
  Docker Desktop, Docker daemon, Docker VMM, and Docker Model Runner are not Mac
  project execution routes unless the user explicitly requests a separate
  non-audio controller experiment. None of these tools by itself proves that
  Demucs/MDX/MelBand/SCNet can use the Apple GPU inside a Linux container.
- The macOS audio GPU candidate path is native PyTorch MPS. It is production
  blocked unless `torch.backends.mps.is_available()` is true on the Mac and the
  actual separator route is explicitly adapted to `device=mps`.

Official references checked for this contract:

- Docker Desktop VMM: https://docs.docker.com/desktop/features/vmm/
- Docker Model Runner: https://docs.docker.com/model-runner/
- Apple container project: https://github.com/apple/container
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html

## Supported modes

### `native_mps_audio`

Use this only after the Mac preflight reports:

- macOS host;
- Apple Silicon `arm64`;
- at least 8 GiB unified memory;
- importable PyTorch;
- `torch.backends.mps.is_built() == true`;
- `torch.backends.mps.is_available() == true`.

This mode is for a single-model, single-route audio probe first. It is not a
full `model_split_v2` route zoo on an 8GB machine.

### `apple_container_orchestration_only`

Mac container orchestration must use Apple `container` CLI
(`/usr/local/bin/container`), not Docker Desktop, Docker daemon, Docker VMM, or
`docker compose`. Apple `container` can run project orchestration, manifests,
CPU diagnostics, and non-audio helpers. It must not be treated as GPU audio
separation until a specific container proves PyTorch MPS or another real audio
accelerator path.

### `docker_model_runner_controller_only`

Docker Model Runner is not part of the Mac project execution route unless the
user explicitly asks for a separate non-audio controller experiment. It is not a
Demucs/MDX/MelBand/SCNet runtime.

## 8GB constraints

The Mac profile must start stricter than the Windows 12GB CUDA profile:

- one audio model resident at a time;
- no text vLLM in the audio path;
- no QuerySep/long-tail route until memory is measured;
- no parallel separator workers;
- no hidden CPU fallback for production audio;
- `route_budget_profile=fast`;
- suggested `max_candidates=3` for the first probe;
- every run must write `progress.jsonl`, `manifest.json`, and `summary.md`.

## Preflight command

Run this on the Mac from the mounted project folder:

```bash
python3 ml_worker_improved/tools/build_macos_8gb_runtime_plan.py \
  --out artifacts/stem_eval/macos8_preflight \
  --echo-progress
```

The output is diagnostic only:

- `delivery_status=blocked`;
- `delivery_allowed=false`;
- `presentation_status=diagnostic_macos_runtime_plan_only`.

If the manifest says `ready_for_native_mps_audio_probe`, the next step is a
small native-MPS separator smoke test. If it says `blocked_macos_preflight`,
do not run the full audio search on that Mac.

## SSH / mounted disk shape

Expected layout when the Mac is connected over SSH and the project disk is
mounted locally on the Mac:

```text
<mounted-project>/AIMusicProject/
  ml_worker_improved/
  docs/
  artifacts/stem_eval/macos8_preflight/
```

Keep generated artifact path segments short. Store host-specific details in
`manifest.json`, not in directory names.

## Mac API over shared artifacts

The supported control shape is:

```text
Windows Codex -> SSH -> Mac command
Mac command -> /Users/andrejviluha/win_artifacts/<short-id>/
/Users/andrejviluha/win_artifacts -> SMB share -> D:\AIMusicProject\artifacts
```

Current verified mount contract:

- Windows path: `D:\AIMusicProject\artifacts`
- SMB share: `\\192.168.0.101\AIMusicArtifacts`
- Mac mount: `/Users/andrejviluha/win_artifacts`
- SSH target: `andrejviluha@192.168.0.167`

If the Mac-side SMB mount stops responding, long audio jobs must not keep
running as blind jobs against `/Users/andrejviluha/win_artifacts`. Copy the
required short input pack to a local Mac scratch root such as:

```text
/Users/$USER/aimusic_local_artifacts/
```

Run the Mac/MPS job locally, then copy the completed artifact root back into
`D:\AIMusicProject\artifacts`. The copied-back artifact must still keep
`delivery_allowed=false` unless the normal delivery contract later accepts it.

Use `ml_worker_improved/tools/run_macos_remote_api_command.py` from Windows to
send one bounded command to the Mac and force the Mac to write:

- `manifest.json`
- `progress.jsonl`
- `stdout.txt`
- `stderr.txt`

Example:

```powershell
py -3 ml_worker_improved/tools/run_macos_remote_api_command.py `
  --ssh-target andrejviluha@192.168.0.167 `
  --ssh-key "$env:USERPROFILE\.ssh\id_ed25519_phone" `
  --mac-artifact-root /Users/andrejviluha/win_artifacts `
  --artifact-subdir mac_api/c001 `
  --command "/usr/local/bin/container run --rm docker.io/library/alpine:latest uname -a"
```

The runner is diagnostic-only. It never writes listen packs and always keeps
`delivery_allowed=false`. This is a control-plane SSH launcher; it is not an
audio/model runtime and it does not relax the container-only contract for project
tests or audio inference. On Mac, that container route is Apple `container`, not
Docker Desktop or Docker daemon.

## Apple container status

`apple/container` is the Mac-native Linux container runtime to test first, not
text vLLM. Verified local setup on the Mac:

- CLI: `/usr/local/bin/container`
- Version: `0.12.0`
- System service: running
- Kernel: recommended Kata `3.28.0` arm64
- Smoke test: `container run --rm docker.io/library/alpine:latest uname -a`
  returned Linux `aarch64`

Diagnostic artifact:

```text
D:\AIMusicProject\artifacts\mac_api\container_smoke.json
```

The Linux container smoke proves remote API/container orchestration works. It
does not prove Metal/MPS GPU access for audio models inside the Linux container.
The first real audio accelerator path remains native PyTorch MPS unless a
separate container GPU proof is added.

## Native PyTorch MPS status

Native macOS MPS is the first confirmed accelerator path for audio tensor work
on this Mac.

Verified setup:

- venv: `/Users/andrejviluha/aimusic_mps_venv`
- Python: `3.9.6`
- PyTorch: `2.8.0`
- `torch.backends.mps.is_built() == true`
- `torch.backends.mps.is_available() == true`
- tensor probe device: `mps:0`
- audio tensor probe: stereo synthetic waveform -> MPS `conv1d` -> WAV

Diagnostic artifact:

```text
D:\AIMusicProject\artifacts\mac_api\mps_audio02\mps_audio_probe.json
D:\AIMusicProject\artifacts\mac_api\mps_audio02\audio\mps_probe.wav
```

This proves native audio tensor operations can run through MPS and write results
back to the Windows artifact folder. It is not a stem separation result and
remains `delivery_allowed=false`.

## MelBand pitch-robust vocal leak guard

The Mac MelBand route can be wrapped by `melband_pitch_guard_v1` when its sound
is the preferred diagnostic baseline. The wrapper runs the existing Mac API
`melband_roformer_vocals` export, then scans the resulting `instrumental.flac`
against transformed vocal probes. Pitch-shifted, lowered, muffled, or
formant-damaged vocal traces are still treated as vocal content.

The guard writes short-path artifacts under roots such as:

```text
D:\AIMusicProject\artifacts\mac_api\mel_guard001\
  manifest.json
  progress.jsonl
  audio\v.flac
  audio\i.flac
  audio\raw.flac
  score\pitch.json
  repair\v.flac
  repair\i.flac
```

The repair is mixture-consistent diagnostic evidence only: leaked vocal-like
patches are subtracted from instrumental and added to vocals so
`vocals + instrumental` stays aligned with `raw_ref`. It cannot create accepted
listen output and must keep `delivery_allowed=false`.

## Mac MPS stem refiner training

`tools/run_mac_stem_refiner_training_cycle.py` is the Mac-first smoke path for
training a real stem-region refiner checkpoint on Apple GPU. It builds a
synthetic clean non-vocal stem, injects pitch-shifted vocal contamination, trains
`StemRegionRepairNet` on MPS, then writes a checkpoint and before/after metrics.

Expected short-path output:

```text
D:\AIMusicProject\artifacts\mac_api\stem_refine001\
  manifest.json
  progress.jsonl
  history.json
  model.pt
  audio\clean.flac
  audio\bad.flac
  audio\src.flac
  audio\repaired.flac
```

The runner must fail closed when MPS is required but unavailable. A trained
artifact is still diagnostic-only: `delivery_allowed=false`,
`listen_pack_created=false`, and `presentation_status=training_diagnostic_only`.

## MelBand-conditioned refiner replay

`tools/apply_mac_stem_refiner_to_mel_guard.py` applies an existing trained
checkpoint to a real `melband_pitch_guard_v1` failure. It is a replay/audit
tool: it loads the guard manifest, builds pitch-shifted vocal source
conditioning from the failed regions, applies the refiner to instrumental, then
transfers the opposite delta into vocals so `vocals + instrumental` remains
aligned with `raw_ref`.

`tools/train_mac_refiner_from_mel_guard.py` trains a new MPS checkpoint on the
same real failure distribution using a pseudo-teacher target derived from the
guard evidence. This is intentionally not verified ground truth:

```text
label_type=pseudo_teacher
is_ground_truth=false
usable_for_promotion=false
delivery_allowed=false
presentation_status=training_diagnostic_only
```

Expected short-path output:

```text
D:\AIMusicProject\artifacts\mac_api\mel_train001\g1.5\
  manifest.json
  progress.jsonl
  history.json
  model.pt
  audio\bad.flac
  audio\target.flac
  audio\src.flac
  audio\v.flac
  audio\i.flac
  score\before\pitch.json
  score\after\pitch.json
```

Passing the pitch guard on one MelBand hardcase is useful training evidence, but
it is not release evidence. Promotion requires a separate multi-hardcase replay
benchmark with no fake ground truth, no per-track timestamps, and no listen
delivery bypass.

## Bounded MelBand refiner benchmark

`tools/run_mac_mel_refiner_benchmark.py` runs the fixed TOP-1 benchmark for
`mel_train001/g1.5`, `g2.0`, and `g2.5`. It has only three terminal outcomes:

```text
promote_to_next_diagnostic_stage
reject_as_overfit
model_gap_requires_new_training_objective
```

The benchmark is intentionally bounded: it cannot add retries, relax detector
thresholds, expand checkpoints, or create listen output. Rows that are intended
as leak hardcases but have no before-repair leak are recorded as invalid and
cannot support promotion.

The benchmark manifest also carries `failure_taxonomy_summary`, aggregated from
uncapped `total_fail_regions` and guard `failure_taxonomy` fields across
checkpoints and row types. This summary is diagnostic-only
(`presentation_status=benchmark_taxonomy_diagnostic_only`,
`delivery_allowed=false`) and is used to rank remaining leak modes/probes; it is
not release or listen-pack evidence.

Current diagnostic example:

```text
D:\AIMusicProject\artifacts\mac_api\mel_bench004\
  manifest.json
  rows.jsonl
  before_after.jsonl
  progress.jsonl
```

`mel_bench004` ended as `promote_to_next_diagnostic_stage` with `g1.5`
selected. All decision rows were valid (`invalid_row_count=0`), leak rows moved
to `after_regions=0`, and the clean negative row stayed at `0 -> 0`. This still
does not create listen output or production release permission.

## MelBand full-track diagnostic stage

`tools/run_mac_mel_refiner_fulltrack_stage.py` applies the selected bounded
benchmark checkpoint to a MelBand pitch-guard source. It is a diagnostic stage,
not a delivery stage:

```text
delivery_allowed=false
listen_pack_created=false
presentation_status=fulltrack_diagnostic_only
```

The full-track stage only accepts benchmark manifests that include
`failure_taxonomy_summary` (`mac_mel_refiner_benchmark_taxonomy_summary.v1`).
Its own manifest also exposes `failure_taxonomy_summary` so a passing bounded
benchmark cannot hide remaining full-track pitch-vocal regions.

TOP-1 promotion is not considered generalized until
`tools/validate_mac_mel_refiner_top1_holdout.py` runs the selected bounded
checkpoint against explicit full-track holdout guard roots. The validator
fails closed when the benchmark was not promoted, the benchmark lacks taxonomy
evidence, a holdout overlaps the benchmark source guard, or any holdout keeps
remaining pitch-vocal regions. Its output stays diagnostic-only:

```text
delivery_allowed=false
listen_pack_created=false
presentation_status=top1_holdout_validation_diagnostic_only
```

Current artifacts:

```text
D:\AIMusicProject\artifacts\mac_api\mel_full001\
```

This replay used `mel_bench004 -> g1.5` on the original 12 second regression
guard source and ended as `diagnostic_passed`: pitch regions moved `4 -> 0`.

Current TOP-1 holdout evidence:

```text
D:\AIMusicProject\artifacts\mac_api\it6b002\  canonical bounded benchmark, 36 -> 0
D:\AIMusicProject\artifacts\mac_api\it6f002\  full-track replay on unified_nt001_guard, 69 -> 46
D:\AIMusicProject\artifacts\mac_api\it6h001\  holdout validation blocked, remaining=46
D:\AIMusicProject\artifacts\mac_api\it6p001\  holdout-to-residual handoff plan, r001 remaining=46
D:\AIMusicProject\artifacts\mac_api\it6r001\  residual r001 training, 16 -> 25, still blocked
```

These artifacts are diagnostic-only. They prove the bounded checkpoint is useful
but not generalized enough for the current full-track hardcase.

`tools/plan_mac_mel_refiner_residual_handoff.py` converts a blocked TOP-1
holdout validation manifest into short residual-training work items such as
`r001`. The handoff is metadata-only: it records the failed full-track stage,
remaining failure taxonomy, expected Mac/MPS execution context, and the exact
`tools/train_mac_refiner_from_fulltrack_residuals.py` command. It does not
train, repair, publish, or create listen output.

The residual handoff is taxonomy-aware. When the failed full-track stage reports
remaining failures by probe or fail mode, it emits one short work item per
taxonomy batch and passes that batch to the trainer via `--region-batch-json`.
The manifest also records source failure coverage, including capped selected
region counts, so a run with `remaining_failure_count=46` is not silently treated
as a single fully covered 16-region training item. `tools/train_mac_refiner_from_fulltrack_residuals.py`
filters the residual guard regions by the requested batch and blocks a batch
with no matching detector-confirmed regions instead of training on the wrong
slice.
If a scoped batch passes its local pitch guard, the trainer now runs the
instrumental damage audit before reporting the batch-scoped diagnostic status.
Only a damage-pass batch may report `residual_batch_trained_passed_pitch_guard`.
If the damage audit fails, the batch stays diagnostic with
`residual_batch_trained_damage_failed`; `full_holdout_resolution_status` remains
`not_evaluated`, and a full holdout plus delivery gates must be rerun before any
promotion or listen-pack decision.
Fresh pitch-guard scans also materialize the uncapped detector failure list as
`score/regions.jsonl`. Residual training prefers this file over the capped
`score/pitch.json` `regions[]` field, so taxonomy batches can train on the full
detector-confirmed region surface instead of only the selected top-`max_regions`
subset. Older artifacts without `regions.jsonl` remain selected-region
diagnostics and must not be treated as full 46-region training coverage.
Large taxonomy buckets are split into deterministic `max_regions` chunks with
`chunk_start`, `chunk_limit`, and `chunk_expected_region_count`; a 39-region
`probe=0.0` bucket is therefore multiple work items, not one hidden 16-region
shortcut.

`it6r001/r001` executed that first work item on Mac MPS. It trained a
pseudo-teacher residual checkpoint from the selected failed regions, but the
after-score still failed and the detector count increased from `16` selected
regions to `25` total fail regions. This closes the current residual
projection-teacher path as insufficient for this hardcase; do not publish the
audio and do not repeat the same objective as a delivery route.

Updated taxonomy-aware rerun:

```text
D:\AIMusicProject\artifacts\mac_api\it7h001\  fresh holdout validation with score/regions.jsonl, 46 full regions
D:\AIMusicProject\artifacts\mac_api\it7p003\  chunked residual handoff, 6 chunks covering 46 regions
D:\AIMusicProject\artifacts\mac_api\it7r003\  chunked residual training sweep, all chunks still blocked
D:\AIMusicProject\artifacts\mac_api\it7cmp001\  diagnostic chunk composition, 46 -> 7
D:\AIMusicProject\artifacts\mac_api\it7rr001\  composition-stage residual pass, 7 -> 4
D:\AIMusicProject\artifacts\mac_api\it7rr002\  strong-projection objective pass, 4 -> 0
D:\AIMusicProject\artifacts\mac_api\it7dmg001..it7dmg004\  damage audits, still failed
D:\AIMusicProject\artifacts\mac_api\it7rr003\  damage-gated strong-projection rerun, 4 -> 0 but damage failed
```

`it7p003` splits the dominant `probe=0.0` bucket into three chunks
`16 + 16 + 7`, then keeps the smaller `-9.0`, `-5.0`, and `-12.0` buckets as
single chunks. The consistent-code `it7r003` sweep stayed diagnostic-only:
`r001..r004,r006` reduced total detector failures (`46 -> 21/23/36/45/45`),
while `r005` stayed `46 -> 46`; every chunk remained
`residual_trained_still_blocked` with `delivery_allowed=false`. This is useful
training evidence, not a listen result.

`tools/compose_mac_mel_residual_chunk_repairs.py` is the follow-up diagnostic
composition route. It does not run model inference; it reads the failed
full-track replay plus residual chunk manifests, validates that each chunk
points back to the same `source_failed_stage_dir`, applies only the masked
instrumental deltas from each chunk's recorded `applied_regions`, transfers the
opposite delta to vocals for mixture consistency, and reruns the full pitch
guard. Its progress log explicitly records:

```text
gpu_inference_skipped=true
gpu_inference_skipped_reason=chunk_repair_composition_uses_existing_residual_audio
acceleration_mode=cpu_dsp_scoring
```

The current `it7cmp001` run composed all six `it7r003` chunks and reduced the
uncapped full-track detector failures from `46 -> 7`, but the final pitch guard
still failed:

```text
status=composition_blocked_pitch_guard
failed_checks=["pitch_guard_failed"]
delivery_allowed=false
listen_pack_created=false
presentation_status=chunk_composition_diagnostic_only
after_failure_taxonomy.by_probe={"0.0":5,"-5.0":1,"7.0":1}
```

`it7cmp001/audio/i.flac` must not be treated as a listen result. A later
waveform-delta audit confirmed why it can sound unchanged despite the detector
count drop: two remaining regions had zero repair coverage, and the
`106.25-107.00` region had near-zero delta (`about -47 dB` versus its segment).

`it7rr001/r001` added a composition-stage source adapter for the residual
trainer and trained on the seven `it7cmp001` regions with Mac MPS. It stayed
diagnostic-only and reduced `7 -> 4`; the remaining four regions were all
`strong_projection_stft` at `probe=0.0`.

`it7rr002/r001` added backend objective routing through the residual wrapper
and ran the current `it7rr001/r001/train` state with:

```text
region_batch.fail_mode=strong_projection_stft
objective_profile=pitch_region_detector_objective_v3
probe_semitone=0.0
target_gain=4.0
resolved_device=mps
```

That pass reached the pitch-only wrapper status recorded by the original
`it7rr002` manifest:

```text
status=residual_batch_trained_passed_pitch_guard
training_status=trained_repair_passed_pitch_guard
before_region_count=4
after_region_count=0
after_score.verdict=pass
delivery_allowed=false
listen_pack_created=false
presentation_status=residual_training_diagnostic_only
```

This is still not accepted output. It proves the remaining
`strong_projection_stft` failure mode needs explicit objective routing rather
than another default `projection_teacher` pass. Production delivery still
requires the broader acceptance contract: exact/reference waveform audit,
damage audit, and accepted/listen-pack gates.

The follow-up Mac damage audits confirm why the pitch-only pass is still not a
usable result:

```text
it7dmg001: original it7h001 -> it7cmp001, fail, instrumental_region_overprocessed at 37.92-38.83
it7dmg002: original it7h001 -> it7rr002, fail, instrumental_region_overprocessed at 37.92-38.83
it7dmg003: it7cmp001 -> it7rr002, fail, instrumental_region_overprocessed at 22.17-23.08
it7dmg004: it7rr001 -> it7rr002, fail, instrumental_region_overprocessed at 22.17-23.33
```

The residual wrapper has been updated so future pitch-pass residual runs execute
`tools/mac_audio_damage_audit.py` automatically against the guard regions. A
candidate like `it7rr002` must therefore be marked damage-failed instead of
being summarized as a pitch-pass result.

`it7rr003/r001` is the replay of the same strong-projection batch with the
damage-gated wrapper:

```text
status=residual_batch_trained_damage_failed
training_status=trained_repair_passed_pitch_guard
before_region_count=4
after_region_count=0
damage_audit.verdict=fail
damage_audit.failed_region_count=2
damage_audit.blocker_reasons=["instrumental_region_overprocessed"]
failed_checks=["instrumental_damage_audit_failed"]
delivery_allowed=false
resolved_device=mps
```

The failing damage regions are `22.17-23.08` and `22.42-23.33`; their
`delta_ratio` values are `0.544467` and `0.72208` against the configured
`max_region_delta_ratio=0.42`. This run proved that the objective can remove the
detector's pitch regions, but overprocesses the instrumental and remains
blocked.

`it7rr004/r001` tested the first damage-guarded static objective profile:

```text
objective_profile=pitch_region_detector_damage_guarded_v1
status=residual_trained_still_blocked
training_status=trained_repair_still_blocked
before_region_count=4
after_region_count=1
delivery_allowed=false
resolved_device=mps
```

The follow-up `it7dmg005` damage audit still failed, and got worse than
`it7rr003` for this batch:

```text
verdict=fail
failed_region_count=4
blocker_reasons=["instrumental_region_overprocessed"]
max_failed_delta_ratio=1.689176
```

`it7sel001` is the current Pareto selection checkpoint over `it7rr003` and
`it7rr004`:

```text
status=blocked_no_pareto_candidate
blocker_reasons=["no_candidate_passes_pitch_and_damage"]
it7rr003: pitch_pass=true, damage_pass=false, after_region_count=0, max_failed_delta_ratio=0.72208
it7rr004: pitch_pass=false, damage_pass=false, after_region_count=1, max_failed_delta_ratio=1.689176
delivery_allowed=false
```

This closes the simple static damage-budget profile as insufficient. The next
valid TOP task is a real damage-aware objective/proxy loop: optimize pitch
region reduction and local instrumental-damage features together before
post-hoc damage audit, then use Pareto selection only as the diagnostic selector.

The next loop added a train-time instrumental damage proxy to
`StemRegionRepairTrainConfig` and objective profile
`pitch_region_detector_damage_aware_v2`. The proxy penalizes region delta RMS
above the damage-audit budget and is included in both training loss and
checkpoint selection loss. The Mac replay was:

```text
it7rr005/r001
objective_profile=pitch_region_detector_damage_aware_v2
target_gain=4.0
status=residual_batch_trained_damage_failed
training_status=trained_repair_passed_pitch_guard
before_region_count=4
after_region_count=0
damage_audit.verdict=fail
damage_audit.failed_region_count=1
max_failed_delta_ratio=0.445517
delivery_allowed=false
```

This is a real improvement over `it7rr003` (`failed_region_count=2`,
`max_failed_delta_ratio=0.72208`), but it is still blocked. The artifact is
diagnostic-only and must not be used as a listen result.

Two follow-up hypotheses were tested and rejected:

```text
it7rr006/r001: same v2 objective, target_gain=3.5, pitch regressed to 4 -> 1, no damage audit
it7rr007/r001: stricter pitch_region_detector_damage_margin_v3, pitch 4 -> 0 but damage failed worse, failed_region_count=3
```

The next structural fix preserved original guard/audit regions as separate
train-time damage constraint masks through the crop instead of collapsing them
into one merged `region_mask`. The Mac replay confirmed the wiring
(`training_constraint_region_count=4`) but did not yet pass:

```text
it7rr008/r001
objective_profile=pitch_region_detector_damage_aware_v2
target_gain=4.0
training_constraint_region_count=4
status=residual_batch_trained_damage_failed
training_status=trained_repair_passed_pitch_guard
before_region_count=4
after_region_count=0
damage_audit.verdict=fail
damage_audit.failed_region_count=1
max_failed_delta_ratio=0.492525
delivery_allowed=false
```

`it7sel002` reran Pareto selection across `it7rr003..it7rr008`:

```text
status=blocked_no_pareto_candidate
blocker_reasons=["no_candidate_passes_pitch_and_damage"]
delivery_allowed=false
```

Current strict-analysis result: `it7rr005` is the best diagnostic candidate so
far, but it is still one overprocessed guard region over budget. The next valid
TOP task is a generalized damage-limited repair loop: when a pitch-pass
candidate fails damage audit narrowly, attenuate only model delta inside
detector/audit regions, preserve mixture consistency, and rerun both pitch and
damage gates. It must remain diagnostic-only until a full holdout and delivery
contract accept it.

Implementation checkpoint:

```text
tools/train_mac_refiner_from_fulltrack_residuals.py now includes a diagnostic
damage-limited retry after initial pitch-pass/damage-fail. The retry writes
short-path `dlim/audio/{v,i,raw}.flac`, attenuates only failed damage-region
model delta, reruns pitch guard, then reruns damage audit over the original
damage-audit region surface. A retry pass changes the residual wrapper status
to `residual[_batch]_trained_damage_limited_passed`, but still keeps
delivery_allowed=false and presentation_status=residual_training_diagnostic_only.

`tools/apply_mac_residual_damage_limited_repair.py` is the reusable fast path
for an already-trained residual manifest that passed pitch but failed damage.
It avoids spending another training run when the checkpoint already exists and
records the source residual/guard/training manifests in the repair manifest.
It can also consume a `mac_residual_damage_audit_route.v1` manifest from the
post-hoc audit path, following `source_residual_manifest` and
`source_guard_manifest` instead of requiring the older residual manifest to be
mutated with embedded damage-audit data.

`tools/apply_mac_residual_damage_audit.py` is the paired fast path for an
already-trained residual manifest that passed pitch but was produced before a
damage audit was embedded in the residual manifest. It materializes the guard
score surface as short-path `score.json`, reruns
`tools/mac_audio_damage_audit.py`, and records the source residual/guard/training
manifests in a diagnostic-only `mac_residual_damage_audit_route.v1` manifest.
If damage passes, the selector may move the candidate to full-holdout
consideration; if damage fails, the selector routes it to the damage-limited
repair path. The route never creates listen output.

Validated locally in Docker:
tests/test_train_mac_refiner_from_fulltrack_residuals.py
tests/test_apply_mac_residual_damage_audit.py
tests/test_apply_mac_residual_damage_limited_repair.py

Mac diagnostic artifacts:
it7dl001/r001 direct helper replay of it7rr005, pitch=pass, damage=pass
it7dl002/r001 full wrapper retrain with it7rr005 args, still blocked 4 -> 5
it7dl003/r001 official apply_mac_residual_damage_limited_repair route,
  source=it7rr005/r001, status=damage_limited_repair_passed,
  pitch=pass, damage=pass, failed_damage_regions=0,
  source_delta_ratio=0.445517, applied_scale=0.914443,
  reconstruction_rms=0.0, delivery_allowed=false

No Mac artifact has been promoted by this checkpoint. `it7dl003/r001` is the
first diagnostic route in this loop to pass both local pitch and damage gates.

Follow-up selector/audit checkpoint:

```text
it7sel003: selected it7dl003 for full-holdout consideration
  status=pareto_candidate_selected_for_full_holdout
  candidate_count=7
  selected_schema=mac_residual_damage_limited_repair_route.v1
  selected_pitch_pass=true
  selected_damage_pass=true
  selected_full_holdout_resolution_status=not_evaluated
  delivery_allowed=false
  presentation_status=pareto_selection_diagnostic_only
```

Selector eligibility is intentionally stricter for
`mac_residual_damage_limited_repair_route.v1`: the route must keep
`status=damage_limited_repair_passed`,
`presentation_status=damage_limited_repair_diagnostic_only`,
`full_holdout_resolution_status=not_evaluated`, and both top-level and nested
damage-audit delivery/listen flags false. A route can therefore be selected for
the next diagnostic gate without becoming a listen result.

The user-facing complaint that `it7cmp001/audio/i.flac` sounds unchanged is
consistent with the objective evidence. `it7cmp001` is still
`status=composition_blocked_pitch_guard` with `46 -> 7` detector failures and
`delivery_allowed=false`. `it7dl003` differs from `it7cmp001` by only about
`-31.91 dB` relative diff RMS over the full 159.889 second file, with changes
localized mainly around the remaining/repair windows:
`22.0-23.5`, `52.5-54.0`, `106.0-107.5`, `142.5-144.5`, and `155.5-157.0`
seconds. This is diagnostic evidence, not an audible production transformation.

`tools/audit_mac_candidate_audibility_delta.py` now records this distinction as
a machine-readable diagnostic gate. For the current checkpoint:

```text
it7aud001/cmp: it7h001 replay instrumental -> it7cmp001, pass, diff=-20.89 dB, active_changed_sec=38.0
it7aud001/d003: it7cmp001 -> it7dl003, pass, diff=-31.91 dB, active_changed_sec=8.25
it7rd003/sel: audibility_audit_status=pass, candidate_ready_for_full_holdout=true, listen_result_allowed=false
```

An audibility pass only proves the candidate is not a no-op/near-identical
artifact. It does not override pitch, damage, full-holdout, reference, or
accepted-listen delivery gates.

`tools/validate_mac_residual_candidate_full_holdout.py` now resolves the
selected residual-candidate full-holdout gate as a metadata-only diagnostic
manifest. It does not run inference or read audio bytes; progress logs set
`gpu_inference_skipped=true`. Current evidence:

```text
it7fh001/d003: source=it7dl003/r001, status=validation_blocked_no_holdout_evidence,
  full_holdout_resolution_status=failed, delivery_allowed=false
it7fh001/d005: source=it7dl005/r001, status=validation_blocked_no_holdout_evidence,
  full_holdout_resolution_status=failed, delivery_allowed=false
it7rd004/sel: audibility_audit_status=pass,
  full_holdout_validation_status=validation_blocked_no_holdout_evidence,
  status=blocked_failed_full_holdout_validation,
  listen_result_allowed=false
```

So the current bottleneck is no longer "did the candidate change bytes"; it is
missing linked independent full-holdout evidence plus missing verified reference
for the actual `be-nicegreenhorn` source.

Source-of-truth reconciliation on Mac shows the `it7*` hardcase comes from:

```text
/Users/andrejviluha/aimusic_local_artifacts/mac_api/unified_nt001_guard/manifest.json
source_audio_path=/Users/andrejviluha/win_project/testing_stuff_music/ytmp3free.cc_be-nicegreenhorn-single-version-youtubemp3free.org.mp3
```

No verified isolated reference stems were found for this source in the current
project state. A temporary `it7gt001` bundle built from Square Oxygen FL exports
was rejected as a reference for this artifact because the source identity,
duration, and sample rate do not match (`it7dl003` is 159.889 sec at 48 kHz;
the Square Oxygen instrumental reference is about 259.556 sec at 44.1 kHz).
Exact-reference delivery audit is therefore blocked for `it7dl003` until a
verified reference bundle for the actual `be-nicegreenhorn` source exists.
```

```text
D:\AIMusicProject\artifacts\mac_api\mel_guard_full001\
D:\AIMusicProject\artifacts\mac_api\mel_full002\
```

`mel_guard_full001` is the full 30.025 second Mac MelBand guard run for the
same MP3. It correctly blocked with
`pitch_shifted_vocal_in_instrumental`. Applying `g1.5` in `mel_full002`
improved the full-track defect (`13 -> 4` pitch-guard regions) but still ended
as:

```text
status=diagnostic_blocked_pitch_guard
after_pitch_verdict=fail
failed_checks=["pitch_guard_failed"]
delivery_allowed=false
```

So the selected TOP-1 checkpoint is not enough for full-track delivery. The
next training objective must learn the remaining full-track leak regions rather
than publishing the repaired audio or relaxing thresholds.

Current full-track pitch-refiner gap:

```text
D:\AIMusicProject\artifacts\mac_api\mel_gap001\
  manifest.json
  summary.md
```

`mel_gap001` records three bounded Mac MPS attempts against the same full-track
hardcase. The detector-aware local patch refiner changed weights and reduced
some scalar pitch scores, but every attempt left `16 -> 16` pitch-vocal leak
regions. The stronger hard-region variant also failed the instrumental damage
audit with `instrumental_region_overprocessed`.

This is a model/objective gap, not a threshold gap. Do not keep increasing local
post-repair strength to force success. The next valid direction is a
separator-level or separator-adapter objective:

```text
melband_pitch_attribution_separator_objective_v1
```

The objective must train vocal attribution inside the separator family so
pitch-shifted vocal content is routed to `vocals` while instrumental detail,
stereo, and mixture consistency survive.

### MelBand pitch-attribution separator-adapter objective

`tools/melband_pitch_attribution_adapter.py` and
`tools/train_melband_pitch_attribution_adapter.py` define the first bounded
implementation of:

```text
melband_pitch_attribution_separator_objective_v1
```

This is not another delivery gate and not a listen route. It is a trainable
separator-adapter objective: the model learns a source-probe delta from the
pitch guard regions, adds that delta to `vocals`, and subtracts the same delta
from `instrumental`. Therefore `vocals + instrumental` is preserved by
construction while the objective explicitly trains weights for vocal
attribution.

Required contract:

```text
training_level=separator_adapter
must_train_weights=true
delivery_allowed=false
listen_pack_created=false
presentation_status=training_diagnostic_only
```

The guard trainer reuses the existing pitch-robust detector regions and writes
short-path artifacts:

```text
audio/v.flac
audio/i.flac
audio/raw.flac
audio/src.flac
audio/d.flac
score/before/pitch.json
score/after/pitch.json
model.pt
manifest.json
progress.jsonl
```

Separator-side guard trainers must derive diagnostic `before_best_score` and
`after_best_score` from `best_probe.score` when present, and otherwise from the
maximum selected `regions[].score`. Copied or staged guard manifests can be
region-only, so a missing `best_probe` must not be recorded as score `0.0`.
Strict target improvement is `after verdict=pass` or an uncapped
fail-region-count reduction. A scalar score drop with the same number of
remaining regions is useful diagnostic evidence, but it is a model/objective
gap, not an improvement gate.

Passing this stage only means the adapter objective is worth replaying or
distilling into the separator family. It does not create a production result or
override the unified pipeline delivery contract.

Current bounded Mac MPS artifacts:

```text
D:\AIMusicProject\artifacts\mac_api\mel_attr001\
D:\AIMusicProject\artifacts\mac_api\mel_attr_g15\
D:\AIMusicProject\artifacts\mac_api\mel_attr_g20\
```

All three trained adapter weights on Mac MPS and stayed blocked:

```text
mel_attr001:  target_gain=1.0, 16 -> 16 regions, best 0.702988 -> 0.620889
mel_attr_g15: target_gain=1.5, 16 -> 16 regions, best 0.702988 -> 0.605856
mel_attr_g20: target_gain=2.0, 16 -> 16 regions, best 0.702988 -> 0.606102
```

The result is useful because it proves trainable separator-adapter weights can
move the target score while preserving mixture consistency. It is still not
enough for delivery because no tested gain reduced the failing region count.
The next objective must use richer separator-side conditioning or distillation,
not publish these diagnostic files as listen output.

Follow-up conditioned adapter artifacts:

```text
D:\AIMusicProject\artifacts\mac_api\mel_attr_cond002\
D:\AIMusicProject\artifacts\mac_api\mel_attr_cond_g15\
D:\AIMusicProject\artifacts\mac_api\mel_attr_cond_g20\
D:\AIMusicProject\artifacts\mac_api\mel_attr_cond_gap001\
```

The conditioned route fixes the previous route drift: manifests now record
`adapter_mode=conditioned`, the trainer builds regionwise projection targets
from second-based guard regions, and `region_projection_coeffs` is populated.

Current result:

```text
mel_attr_cond002:   target_gain=1.0, 16 -> 16 regions, best 0.702988 -> 0.611743, damage pass
mel_attr_cond_g15:  target_gain=1.5, 16 -> 16 regions, best 0.702988 -> 0.606265, damage pass
mel_attr_cond_g20:  target_gain=2.0, 16 -> 16 regions, best 0.702988 -> 0.606655, damage fail
```

This closes the current local conditioned-adapter family as insufficient for
this hardcase. The next objective is not higher gain; `g2.0` already reaches
`instrumental_region_overprocessed` before reducing the region count. Continue
with richer separator-side distillation or real separator fine-tuning:

```text
melband_pitch_attribution_separator_distillation_v2
```

## Full-track residual refiner objective

`tools/train_mac_refiner_from_fulltrack_residuals.py` continues from a failed
full-track diagnostic stage. It does not rescan arbitrary candidates. It builds
a short-path residual guard from the previous replay audio and trains only on
the detector-confirmed remaining pitch-guard regions.

Current artifact:

```text
D:\AIMusicProject\artifacts\mac_api\mel_resid001\
  manifest.json
  progress.jsonl
  guard\manifest.json
  train\manifest.json
  train\model.pt
  train\audio\v.flac
  train\audio\i.flac
  train\score\before\pitch.json
  train\score\after\pitch.json
```

`mel_resid001` used `mel_full002` as its source failure, trained on the four
remaining full-track pitch regions, and moved the residual guard from `4 -> 0`.
The run used native Mac MPS and `target_gain=2.5`.

This is still not production delivery:

```text
label_type=pseudo_teacher
is_ground_truth=false
usable_for_promotion=false
delivery_allowed=false
listen_pack_created=false
presentation_status=residual_training_diagnostic_only
```

The audio under `mel_resid001\train\audio\` is diagnostic evidence for the
two-stage MelBand refiner chain. It must not be published as a listen/result
pack until a separate validation stage proves the chain on non-training
hardcases.

## TOP-2 chain validation harness

`tools/validate_mac_mel_refiner_chain.py` composes the two checkpoints:

```text
mel_train001/g1.5/model.pt -> mel_resid001/train/model.pt
```

It runs both stages on a guard source, writes separate stage manifests, then
checks the final pitch guard. It also checks whether the validation guard is the
same guard that produced the residual training checkpoint. If so, the run is
blocked as training overlap even when the final audio passes.

Current artifact:

```text
D:\AIMusicProject\artifacts\mac_api\mel_chainval001\
  manifest.json
  progress.jsonl
  s1\manifest.json
  s2\manifest.json
  s2\audio\v.flac
  s2\audio\i.flac
```

`mel_chainval001` result:

```text
stage1: 13 -> 4, pitch guard still fail
stage2: 4 -> 0, pitch guard pass
status=validation_blocked_training_overlap
failed_checks=["training_overlap"]
delivery_allowed=false
listen_pack_created=false
```

This reaches TOP-2 as a validation harness, but does not promote the chain. The
next validation must use a non-training hardcase/source guard before any listen
contract can be considered.

Non-training validation artifact:

```text
D:\AIMusicProject\artifacts\mac_api\mel_guard_nt001\
D:\AIMusicProject\artifacts\mac_api\mel_chainval_nt001\
```

`mel_guard_nt001` uses a different source track and is therefore not the
training source for `mel_resid001`. It is a real hardcase: the pitch guard found
16 pitch-shifted vocal leak regions in the MelBand instrumental.

`mel_chainval_nt001` result:

```text
training_overlap=false
stage1: 16 -> 16, pitch guard fail
stage2: 16 -> 16, pitch guard fail
status=validation_blocked_pitch_guard
failed_checks=["pitch_guard_failed"]
delivery_allowed=false
listen_pack_created=false
```

Additional residual checkpoints trained on this non-training hardcase did not
solve it:

```text
mel_resid002: target_gain=2.5, 16 -> 16
mel_resid003: target_gain=5.0, 16 -> 16
```

`mel_resid003` changed the waveform more strongly than `mel_resid002`, but the
same 16 pitch-leak regions remained. For this hardcase, the current
projection/pseudo-teacher residual objective is a model-gap, not a threshold or
weight-strength issue. The next change must alter the training objective or
model architecture, not keep increasing gain or publishing diagnostic audio.

Follow-up objective probes on `unified_nt001` confirmed the same gap:

```text
mel_obj003: target_gain=3.0, 16 -> 16, damage pass
mel_obj004: target_gain=6.0, 16 -> 16, damage fail
mel_obj006: target_gain=4.0, 16 -> 16, damage fail
```

Increasing target gain lowers the detector best score, but it does not reduce
the number of pitch-leak regions and begins to overprocess the instrumental.
This is evidence for a model/objective gap, not a reason to keep increasing
repair strength.

Route comparison also failed to find a clean replacement vocal/minus specialist
on this hardcase:

```text
D:\AIMusicProject\artifacts\mac_api\alt_vocal001\manifest.json
```

Summary:

```text
melband_roformer_vocals:      fail, 16 regions, best_score=0.702988
htdemucs_6s_sum_non_vocal:    fail, 16 regions, best_score=0.779278
mdx23c_vocals:                fail, 16 regions, best_score=0.923959
scnet_xl_ihf_musdb18:         infra_gap, MPS OOM
```

MelBand remains the best available route by this detector, but it is still
blocked. The next useful work is a new separator/refiner training objective for
`pitch_shifted_vocal_in_instrumental`, not another threshold adjustment or
listen-pack export.

## Instrumental damage audit

`tools/mac_audio_damage_audit.py` checks whether a diagnostic MelBand repair
damages the instrumental around pitch-guard regions. It compares the repaired
instrumental against the original MelBand instrumental with region-local
high-band, centroid, delta, RMS, and stereo-width evidence. This catches cases
where total loudness stays similar but the music is audibly overprocessed or
detail is removed.

Current artifact:

```text
D:\AIMusicProject\artifacts\mac_api\mel_damage003\
  damage.json
  progress.jsonl
```

`mel_damage003` compares `mel_resid003/audio/i.flac` against the source
MelBand instrumental and fails with `instrumental_region_overprocessed`. This
means `mel_resid003` is not only still blocked by pitch leak; its instrumental
repair is also explicitly marked as damaging and cannot be used as a listen
result.

## Unified Mac track pipeline

`tools/run_mac_unified_track_pipeline.py` is the first Mac staged track runner.
It keeps model lanes independent:

```text
vocal/minus lane: melband_roformer_vocals
stem lane:        htdemucs_6s
```

The contract is:

- vocal/minus and stem winners are independent;
- a stem-lane failure must not cause a passed vocal/minus lane to be
  regenerated;
- a passed vocal/minus lane must not be replaced by a failed stem sibling lane;
- all six stems are required from the stem lane:
  `vocals`, `drums`, `bass`, `guitar`, `piano`, `other`;
- MelBand pitch guard and instrumental damage audit are both run before any
  delivery contract can be considered;
- missing/malformed guard repair data or missing Mac API run directories block
  with explicit manifest reasons instead of silently skipping audit or
  crashing the pipeline;
- the runner is diagnostic-only and always writes `delivery_allowed=false`
  until a later production delivery contract accepts every lane.

Current artifact:

```text
D:\AIMusicProject\artifacts\mac_api\unified001\
  manifest.json
  progress.jsonl
  mel\guard\audio\v.flac
  mel\guard\audio\i.flac
  stm_api\run\vocals.flac
  stm_api\run\drums.flac
  stm_api\run\bass.flac
  stm_api\run\guitar.flac
  stm_api\run\piano.flac
  stm_api\run\other.flac
```

`unified001` produced all six stem files through `htdemucs_6s`, but the overall
run is blocked because the MelBand instrumental still has
`pitch_shifted_vocal_in_instrumental`. This is the intended lane separation:
stem artifacts exist for inspection and training evidence, while the
vocal/minus lane remains the blocker that must be fixed next.

The same runner was also replayed on the non-training hardcase that produced
`mel_resid003`:

```text
D:\AIMusicProject\artifacts\mac_api\unified_nt001\
  manifest.json
  progress.jsonl
  mel\guard\audio\v.flac
  mel\guard\audio\i.flac
  stm_api\run\vocals.flac
  stm_api\run\drums.flac
  stm_api\run\bass.flac
  stm_api\run\guitar.flac
  stm_api\run\piano.flac
  stm_api\run\other.flac
```

`unified_nt001` also produced all six stem files through `htdemucs_6s`; the
run remains blocked only because the MelBand instrumental still contains
`pitch_shifted_vocal_in_instrumental`. The pipeline must therefore preserve the
available stem-lane evidence and focus the next model-training work on the
vocal/minus leak objective instead of regenerating already available lanes.

## Not allowed

- Do not install or run Qwen/vLLM as if it were a music separator.
- Do not mark Docker Model Runner as audio GPU support.
- Do not use CPU fallback as a production result when MPS is unavailable.
- Do not publish listen packs from Mac experiments until the normal delivery
  contract accepts them.
