# Non-vocal stem quality control

Этот документ фиксирует production-safe слой контроля качества для уже экспортированных non-vocal stem-ов.

Related benchmark: `docs/STEM_QUALITY_BENCHMARK.md` documents the synthetic regression harness for
known stem-quality defect labels such as drum tonal leakage, proxy-copy stems, insufficient presence,
and source conservation errors. The benchmark fails on unexpected P1 labels unless a fixture explicitly
allows a secondary label.

## Cross-stem forensic delivery gate

`ml_worker_improved/tools/stem_forensic_crosscheck.py` is the forced sibling-stem forensic gate for
delivery candidates and final listen packs.

Contract:

- a candidate cannot enter listen only because the standalone contamination detector passes;
- every delivery candidate must run a cross-stem matrix over forbidden sibling relationships:
  `vocals` against `drums/bass/guitar/piano/other`, non-vocal stems against `vocals`, configured
  non-vocal cross-bleed pairs, and `minus` against `vocals`;
- `vocals:drums` is a special high-risk pair: aligned high-frequency transients, high-frequency
  flux/envelope correlation, transient-region STFT similarity, or source-relative projection can
  raise `vocal_contains_drum_transient_imprint`;
- gain/stress probes are diagnostic only. They can expose quiet coherent imprint, but they never
  modify delivery audio;
- unknown forensic outcomes block automatically as `blocked_unknown`;
- failed pair checks write pair JSON, matrix JSON, suspicious regions, and optional proxy audio under
  `score/forensic/` for candidate evidence only;
- after `build_listen_pack()` concatenates accepted windows, the final `listen/audio/*.flac` files
  must be revalidated with the effective forensic gate used by the accepted candidate set. If
  candidate-level configs conflict, the listen pack blocks fail-closed. If revalidation fails, the
  listen pack is deleted, `delivery_status=blocked`, `delivery_allowed=false`,
  `presentation_status=blocked_from_listen_pack`, and `failed_checks=["listen_pack_forensic_failed"]`;
- forensic proxy audio is never a result/listen artifact.

## Model-split v2 pair ranking and repair candidates

`model_split_v2` composite search must rank vocal/minus route pairs by the audio that would actually
be delivered after candidate copy/shape alignment.

Contract:

- route collection must not dedupe only by `route_id`; the same model id can produce different audio
  in different candidates, so identity includes the required source audio paths;
- pair ranking uses aligned `output_path` from `copied_outputs` when available and falls back to raw
  `source_path` only when no aligned output exists;
- `delivery_pair_residual` repair candidates such as `pr16` and `bs400_pr16` are allowed only as
  gated repair candidates. They can reduce `vocals + minus` residual, but they cannot publish unless
  damage, voice-presence, contamination, and cross-stem forensic gates also pass;
- `delivery_pair_residual` is not a fallback-as-success route. If it exposes vocal leakage, drum
  imprint, stereo collapse, or other stem damage, the candidate stays blocked and is training/root
  cause evidence only;
- bounded composite search must sample support lanes before exhausting all repair variants. In
  practice this means the first pass tries no-support/no-repair, then available `bass_support` and
  `non_vocal_support` routes without repair. The bounded pass must then try key support-plus-repair
  combinations (`pr16`, `bs400_pr16`, `vp100`, `bs400`) before lower-priority no-support variants,
  because support-only candidates can still expose reconstruction, stereo, or vocal-leak blockers;
- support lanes must be materialized by the default bounded recipe, not only listed in
  `target_specific_routes`. `bass_support` must have an executable producer, and `non_vocal_support`
  must not depend on a single fragile first candidate.

## Training-first stem-region repair

`ml_worker_improved/tools/build_stem_region_synthetic_pretrain_dataset.py` and
`ml_worker_improved/tools/train_stem_region_repair_model.py` are the current training-first repair
entry points for blocked candidate evidence.

Contract:

- blocked separator output is mining evidence, not ground truth;
- synthetic pretrain rows may be exact ground truth only because the clean target is generated before
  corruption;
- supported synthetic repair issues are `stereo_collapse`, `vocal_leak`, and `instrument_leak`;
- focused builders can filter by `--issue-type` and `--stem`, so repeated blockers such as
  `bass:vocal_leak`, `minus:vocal_leak`, and `other:instrument_leak` can produce separate one-row
  checkpoints;
- v1 trainer intentionally trains exactly one focused row per checkpoint. This keeps the first repair
  target small and prevents unrelated failures from being hidden inside one broad model;
- `model_split_v2` can inject multiple focused checkpoints as a single sequence repair option
  (`srseq_vp100_pr16`). The sequence applies several `stem_region_repair` patches before the
  existing debleed/residual repair steps, but it is still a gated repair candidate;
- if the sequence shifts failure from leakage into reconstruction or damage, the candidate remains
  blocked and the run must write a root-cause queue. It must not create a listen pack.
- if an applied `stem_region_repair` reports `diagnostics.target_issue_improved=false`, that repair
  is direct training evidence. The root-cause queue records `root_cause=repair_gap`,
  `training_action=retrain_failed_repair_checkpoint`, `failed_repair_id`, and
  `failed_repair_checkpoint` even when the blocker appears only once;
- residual localization may still be recorded for failed-repair rows, but it must not overwrite the
  failed repair stem/action. A failed `other:bad_reconstruction` checkpoint must remain an `other`
  training item even if the delivery-pair residual localizer points elsewhere;
- root-cause route policy can remove future repair options that reuse a failed repair id or failed
  checkpoint. This only prevents repeated harmful repair attempts; it cannot accept a candidate.
- `inspect_root_cause_route_policy.py` can run the same policy filtering as a GPU-free diagnostic.
  It writes `manifest.json`, `recipe_after_policy.json`, and `progress.jsonl` with
  `gpu_inference_skipped=true`; the artifact must stay `delivery_allowed=false` and
  `presentation_status=diagnostic_policy_only`. The manifest records
  `active_blocked_repair_option_count` so a launch-ready filtered recipe can prove no active repair
  option still references a blocked repair id/checkpoint;
- `build_stem_search_launch_manifest.py` converts a filtered recipe into a reproducible launch
  packet without running separation. It records the recipe sha256, exact in-container command,
  host Docker command, GPU preflight, target run root, `progress.jsonl`, top-level
  `gpu_preflight_summary`, `resource_blocker_summary`, `status_summary`, and `summary.md`. The
  packet is diagnostic-only (`delivery_allowed=false`, `presentation_status=diagnostic_launch_only`) and may say
  `launch_status=blocked_gpu_preflight` when VRAM is not ready;
- `run_stem_search_launch_manifest.py` executes a launch packet fail-closed. It verifies the
  filtered recipe sha256 before doing anything, rechecks GPU preflight, writes execution
  `progress.jsonl`, honors the packet `wait_for_gpu_free_sec` / `gpu_preflight_poll_sec` with
  `gpu_preflight_wait` heartbeats, and starts the saved in-container command only when preflight
  passes. A sha mismatch or still-busy GPU writes a diagnostic execution manifest and `summary.md`
  with `run_command_started=false`, `result_status=not_started`, the final free/used VRAM snapshot,
  and best-effort resource blocker evidence;
- `build_audio_iteration_analysis.py` records the strict source-of-truth pass and idea funnel for
  the current audio task. It links policy, launch, and launch-execution manifests, writes
  `ideas.jsonl` and `summary.md`, rejects forbidden ideas (`threshold_relaxation`, `human_review`,
  blocked-output delivery), and remains diagnostic-only. Once a launch-execution manifest already
  proves `blocked_gpu_preflight` with `run_command_started=false`, the next selected action becomes
  `gpu_resource_arbitration` instead of repeating the same wait-capable launch step;
- `build_gpu_resource_arbitration_plan.py` turns that selected action into a non-destructive
  arbitration artifact. It may report host/container evidence when available, but it must not stop
  containers, kill processes, run audio replay, use CPU fallback, or set `delivery_allowed=true`.
  Its output remains `presentation_status=diagnostic_gpu_resource_arbitration_only`. When an
  execution manifest already contains `resource_blocker_snapshot`, arbitration must carry that
  snapshot forward instead of collecting an empty replacement;
- standalone diagnostic `stem_region_repair_probe.v1` manifests can be ingested into the same
  root-cause queue with `--probe-manifest`. Probe-derived rows remain `blocked_candidate_mining`
  evidence with `delivery_allowed=false`; they exist to prevent harmful checkpoint reuse and to feed
  retraining, not to create listen output.

### Reconstruction-aware repair v2

`ml_worker_improved/tools/analyze_model_split_reconstruction_residual.py`,
`ml_worker_improved/tools/build_stem_repair_hardcase_benchmark.py`, and
`ml_worker_improved/tools/validate_stem_repair_checkpoint.py` move repeated blockers from
"more rejected candidates" into training-ready evidence.

Contract:

- `bad_reconstruction` rows must be localized against the actual audio residual when possible.
  The queue row records `affected_stem`, `residual_region`, `route_family`, and
  `residual_localization` instead of leaving the defect anonymous;
- infra-failed route families are marked `quarantined_for_next_run=true` so the next replay can
  avoid spending candidates on known unavailable routes;
- reconstruction-aware repair rows can carry `sibling_stem_paths`, `reconstruction_surface`,
  `target_surface`, and `root_cause_row_id`;
- repair training can use `mix_conservation_loss_weight`, `sibling_preserve_loss_weight`, and
  `stem_sum_loss_weight`, but these losses are training signals only. They do not relax delivery
  gates and do not authorize listen output;
- hardcase benchmark rows built from blocked candidates are mining/holdout evidence only:
  `is_ground_truth=false`, `usable_for_promotion=false`, `delivery_allowed=false`;
- checkpoint validation writes explicit before/after rows. Improvement on a target defect is not a
  release claim unless the normal delivery contract still passes afterward;
- generated separator output and blocked candidates cannot become exact ground truth.

## Learned route policy v1

`ml_worker_improved/tools/stem_route_policy_common.py`,
`ml_worker_improved/tools/build_stem_route_policy_dataset.py`,
`ml_worker_improved/tools/train_stem_route_policy_model.py`, and
`ml_worker_improved/tools/validate_stem_route_policy_checkpoint.py` define the first learned
route/action controller for stem separation.

Contract:

- the policy model never writes repaired waveform audio;
- the policy can only recommend bounded route actions such as `prioritize_route`,
  `quarantine_route`, lane switches, repair-option enablement, or template retuning;
- no policy action can set `delivery_allowed=true`;
- unknown or invalid policy actions resolve fail-closed to `block_unknown`;
- generated separator outputs and blocked candidates cannot become exact ground truth;
- route-changing oracle labels require counterfactual before/after evidence. Without that evidence,
  repeated defects such as `bad_reconstruction` remain `block_unknown` training rows;
- observed-run counterfactual evidence may be used only when an already executed alternative
  candidate in the same window reaches a later gate stage and has extractable route provenance.
  Missing route provenance does not create a route-switch label;
- runner mode `shadow` only records `route_policy_*` fields in candidate manifests;
- runner mode `rank` may reorder candidate attempts, but it cannot bypass reconstruction, damage,
  voice-presence, contamination, cross-stem forensic, or listen-pack revalidation gates;
- `rank` success is measured first as safer candidate ordering / reduced wasted attempts. It is not
  an audio acceptance claim unless the normal delivery contract accepts the final listen pack;
- `stem_region_repair` remains experimental/diagnostic unless separately promoted by its own
  validation contract.

## Persistent separator backend

`ml_worker_improved/tools/persistent_separator_backend.py` and
`ml_worker_improved/tools/serve_persistent_separator.py` define the vLLM-style runtime optimization
layer for audio separation routes.

Contract:

- text LLMs and literal vLLM serving are not the audio separator. Demucs, MDX23C, MelBand,
  BS-RoFormer, SCNet, and registered audio routes still produce stems;
- `run_fully_automatic_stem_search.py --separator-backend subprocess` remains the default path;
- `--separator-backend persistent-server` may send target-specific lane commands to
  `POST /separate` on the `separator-server` Docker profile;
- the persistent backend can cache route outputs and keep a long-lived serving process, but it
  cannot set `delivery_allowed=true` and cannot bypass reconstruction, damage, voice-presence,
  contamination, cross-stem forensic, or listen-pack revalidation gates;
- resident in-process execution is available only for explicitly registered route IDs. The current
  `resident_fixture_copy` route is a POC contract test: it proves warm executor metadata,
  fail-closed manifests, and non-subprocess execution, not production separation quality. Because it
  is a CPU-copy fixture, it blocks when `require_gpu=true` and records `gpu_inference_skipped=true`;
- real audio-separator resident execution is opt-in via `prefer_resident=true` for registered
  `audio_separator_*` vocal routes. The adapter keeps one long-lived JSONL worker process per
  model/parameter key, loads the Audio Separator model once inside that worker, and reuses it across
  requests. It still records `external_runtime_subprocess_used=true` because the Audio Separator
  package lives in its own Python runtime, but successful resident responses must report
  `persistent_worker_process_used=true` and `per_request_external_runtime_subprocess_used=false`;
- MelBand/Becruily resident execution is opt-in via `prefer_resident=true` for
  `melband_roformer_vocals` and `becruily_instrumental_vocals` route families. It keeps a
  long-lived MelBand worker process with the loaded torch model/config and reuses it across
  requests. It does not accept delivery and must still report `delivery_allowed=false`;
- `model_split_v2` default recipes set `prefer_resident=true` automatically for
  `audio_separator_bs_roformer_vocals`, `melband_roformer_vocals`, and
  `becruily_instrumental_vocals` vocal/instrumental lane routes;
- `model_split_v2` may explicitly opt in to fail fast after vocal/minus lane execution when
  `vocals + minus` already fails delivery-pair reconstruction and no repair/debleed/merge mutation
  or composite search path could change that pair.
  This is an execution shortcut only: it blocks the candidate with `delivery_allowed=false` and skips
  six-stem/support routes; it never accepts audio or relaxes later gates;
- per-route runtime budgets are part of the separator backend provenance. `mdx23c_vocals` and
  `mdx23c_vocals:instrumental` use a bounded 6-second-window budget so a slow subprocess becomes an
  explicit timeout/infra blocker instead of stalling the whole candidate loop;
- `--route-budget-profile full` is the default. `--route-budget-profile fast` is an explicit
  diagnostic/benchmark profile that caps known slow route families such as MDX23C, MelBand/Becruily,
  and SCNet more aggressively. It cannot accept a candidate, skip delivery gates, or mark timed-out
  audio as a result;
- unsupported route IDs remain explicit subprocess routes and must report
  `resident_backend_used=false`. Cache hits preserve the original backend provenance;
- `/health` exposes resident executor readiness and `model_load_count` so model warmup/cache state is
  observable before long candidate loops;
- cache keys are short hash aliases and must not contain raw track titles, raw paths, or long route
  identifiers in artifact path segments;
- when `require_gpu=true`, CPU fallback is forbidden. A missing or failing persistent server is
  recorded as `infra_gap` / `blocked_unknown`, not accepted fallback audio;
- the optional Docker service is profile-gated as `audio-persistent-separator` and is independent
  from the optional `llm-vllm` text-controller service.

## macOS 8GB runtime contract

`docs/MACOS_8GB_AUDIO_RUNTIME.md` documents the constrained Mac runtime path. Docker Desktop,
Docker VMM, Apple container tooling, and Docker Model Runner are not treated as Demucs/MDX/MelBand/
SCNet GPU support unless a concrete audio route proves accelerator access on that host. The first
supported Mac probe is native PyTorch MPS, recorded by
`ml_worker_improved/tools/build_macos_8gb_runtime_plan.py`, and remains diagnostic-only with
`delivery_allowed=false`.

## Runtime observability and GPU preflight

`run_fully_automatic_stem_search.py` must not run long audio routes as a black box.

Contract:

- CLI runs mirror `progress.jsonl` events to stderr by default. `--no-echo-progress` is allowed only
  for controlled cases where another monitor tails `progress.jsonl`;
- every run still writes `progress.jsonl` with `start`, `gpu_preflight`, candidate/window events, and
  final `end`;
- CLI runs use `--min-gpu-free-mib` with a conservative default before starting candidate
  generation. If `require_gpu=true` and free VRAM is below that value, the run blocks before the
  candidate loop with `root_cause=infra_gap`, `failed_checks=["gpu_preflight_failed"]`,
  `delivery_allowed=false`, and no listen pack;
- CLI runs may use `--wait-for-gpu-free-sec` with `--gpu-preflight-poll-sec` to wait for VRAM before
  candidate generation. Each wait poll writes `gpu_preflight_wait` to `progress.jsonl`/stderr.
  If the wait expires, the run still blocks fail-closed before candidates and records the final GPU
  preflight evidence;
- `run_stem_search_launch_manifest.py` may override a saved launch packet's wait with
  `--wait-for-gpu-free-sec` / `--gpu-preflight-poll-sec`. This is an auto-resume convenience only:
  it cannot relax `min_gpu_free_mib`, cannot start on CPU, and cannot mark blocked output as a
  result;
- launch and launch-execution artifacts must expose the current resource blocker without requiring
  manual JSONL parsing: `gpu_preflight_summary`, `resource_blocker_snapshot`,
  `resource_blocker_summary`, `status_summary`, and `summary.md` must say whether the command
  started, whether a result was produced, the final `free_mib/used_mib/total_mib` snapshot, and
  whether container-level Docker stats were available from the current execution environment;
- launch manifest creation must write `resource_snapshot_start` and `resource_snapshot_end` progress
  events, and launch execution must write `resource_snapshot_start` before collecting blockers, so
  slow container attribution cannot make the process look silent;
- when `ml-worker` cannot access host Docker CLI/socket, it must record
  `docker_container_stats_available=false` with a reason such as `docker_cli_unavailable`; this is
  evidence coverage, not a successful offender attribution;
- when Docker CLI is unavailable but `/var/run/docker.sock` is mounted, the resource snapshot may use
  Docker Engine API as `docker_container_stats_source=docker_engine_socket` to populate container
  memory names inside the Docker-only launch path;
- saved `docker_compose_command_host` entries in launch manifests must include the Docker socket
  mount so rerunning a saved launch keeps the same container attribution path;
- `build_host_resource_snapshot.py` is the diagnostic-only fallback for that case. It reads
  host-visible `nvidia-smi`, `docker stats`, and `docker ps`, writes `manifest.json`,
  `resource.json`, `progress.jsonl`, and `summary.md`, and stays
  `presentation_status=diagnostic_host_resource_snapshot_only`. It must not start audio replay,
  stop containers, kill processes, or set `delivery_allowed=true`;
- GPU preflight is runtime provenance only. It does not relax delivery gates and cannot turn a bad
  audio candidate into an accepted result.

## Trainable stem refiner cycle

`ml_worker_improved/tools/run_trainable_stem_refiner_cycle.py` is the minimal weight-changing
audio path. It creates a focused synthetic stem defect, trains `stem_region_repair_model` with a
real torch `state_dict`, applies the checkpoint, and writes explicit before/after evidence.

Contract:

- this stage changes model weights, not route ordering only;
- the cycle must write `model/model.pt`, `history.json`, `before_after.json`, and short audio aliases
  under `audio/`;
- success requires target-region error to decrease after applying the checkpoint while outside-region
  change stays near zero;
- generated repaired audio remains diagnostic only:
  `delivery_allowed=false`, `presentation_status=blocked_from_listen_pack`;
- this is the first training proof for the audio refiner, not a final production listen result.

## Teacher-mask checkpoint candidate routes

`ml_worker_improved/tools/apply_teacher_mask_checkpoint_candidate.py` lets a learned
teacher-mask separator checkpoint enter `model_split_v2` as a normal target-specific candidate route.

Contract:

- the checkpoint route applies `train_teacher_mask_separator.py` checkpoint weights to the current
  window without a teacher target;
- supported v1 route roles are `vocals`, `minus`, `drums_support`, `bass_support`,
  `guitar_support`, `piano_support`, and `other_support`;
- `run_fully_automatic_stem_search.py --teacher-mask-checkpoint role:stem:checkpoint_path`
  injects the route into the candidate pool;
- the route writes only candidate/debug metadata and sets `delivery_allowed=false`;
- the runner can accept audio from this route only after the existing reconstruction, damage,
  voice-presence, contamination, cross-stem forensic, and listen-pack revalidation gates pass;
- teacher-mask checkpoint application is not a fallback-as-success path and cannot bypass holdout
  or delivery contracts.

## Scope

`ml_worker_improved/tools/stem_quality_control.py` применяет только общие проверки и безопасный headroom:

- fail-fast validation: sample rate, channel shape, finite samples, minimum duration, minimum RMS;
- downward-only peak headroom до `--target-peak-dbfs`;
- запись `PCM_24` FLAC;
- manifest с before/after metrics;
- optional listen pack.

Tool не является repair-моделью, не подменяет stem fallback-ом и не усиливает тихие stem-ы. Если входной stem не проходит базовые проверки, выполнение падает.

## Docker-only usage

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/stem_quality_control.py `
  /app/lab_artifacts/golden_audio/<run>/hq_stems/drums.flac `
  /app/lab_artifacts/golden_audio/<run>/drums_qc `
  --stem-name drums `
  --target-peak-dbfs -1.0 `
  --raw-ref-path /app/lab_artifacts/golden_audio/<source>/instrumental_resurrection_minus.flac `
  --listen-dir /app/lab_artifacts/golden_audio/<run>/listen_drums_qc `
  --source-label 01_drums_original.flac `
  --output-label 02_drums_qc_headroom.flac
```

## Orchestral / long-tail stems

В основном `ml_worker_improved` route нет direct `orchestral`, `strings` или `brass_woodwinds` stem. Такие классы должны идти через отдельный long-tail/query separation backend, например `stemlab_multicontainer_blueprint_v2/services/querysep-worker`, с `banquet_byoq` backend и fail-fast поведением при отсутствии checkpoint/runtime.

Основной repeatable queue/runtime entrypoint для этого слоя — `querysep_long_tail_build`; подробный payload и shared-volume contract описаны в `docs/QUERYSEP_LONG_TAIL_BUILD.md`.

Запрещено считать `other`-filtered proxy полноценным orchestral stem. Proxy artifacts допустимы только как diagnostic listen material с явным manifest status.

`querysep_long_tail_build` использует `long_tail_model_router_v1`: production-eligible candidates
(`banquet_byoq` или явно переданные external model outputs) могут быть выбраны только после acoustic gate.
Если primary Banquet output rejected для harmonic long-tail классов, route может создать
`banquet_byoq_harmonic_residual_query`: отдельный Banquet BYOQ pass с auxiliary
`harmonic_residual` query bus. Это не selected stem и не fallback; selected может стать только
сам Banquet output второго pass после полного `long_tail_acoustic_quality_gate_v1`.
Для этого второго pass adapter отправляет `raw_step.query_mode="harmonic_residual_single"`;
`querysep-worker` обязан строить Banquet query exemplar только из `bus_harmonic_residual` и не смешивать
`other` / `guitar` / `piano` / `melody` buses в query ensemble.
Если production-eligible candidate содержит локальную proxy leakage (`windowed_proxy_correlation_too_high`) без
полного proxy-copy blocker-а, route может создать `windowed_proxy_decorrelated_candidate`. Это production candidate,
а не fallback: он обязан сохранять provenance исходного candidate и снова проходить полный acoustic gate.
Если candidate слишком близок к `other` bus, но не пересёк full-copy/windowed thresholds, acoustic gate
ставит отдельный blocker `possible_other_proxy_leakage`. Это запрещает принимать умеренно decorrelated
`other` proxy за `strings_orchestra` или другой long-tail stem только потому, что он уже ниже грубого
copy-correlation порога.
Bus-derived diagnostic candidates используются для отчёта и rejection evidence, но не выбираются как production stem.
Manifest обязан различать `selected_output_path` и `primary_querysep_output_path`, чтобы external model
candidate не выглядел как файл, созданный Banquet/querysep-worker.

Минимальный acceptance gate для long-tail stem:

- output не пустой и проходит базовые audio metrics;
- output проходит minimum presence: `rms_dbfs >= QUERYSEP_LONG_TAIL_MIN_RMS_DBFS` и
  `output/source rms ratio >= QUERYSEP_LONG_TAIL_MIN_SOURCE_RMS_RATIO_DB`;
- output не должен быть почти пустым при формально проходящем full-track RMS: active-window gate требует
  минимум `QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_WINDOWS` окон с
  `rms_dbfs >= QUERYSEP_LONG_TAIL_MIN_ACTIVE_PRESENCE_RMS_DBFS`;
- duration output не должен расходиться с source audio больше `QUERYSEP_LONG_TAIL_MAX_DURATION_DELTA_SEC`;
- `fallback_used=false` в manifest;
- backend явно `banquet_byoq`, если нужен настоящий query-separated stem;
- correlation с исходным `other` или старым proxy не должна быть высокой. Для diagnostic gate текущий reject threshold: `> 0.85`;
- local windowed correlation с proxy buses также не должна быть высокой: active окна с
  `rms_dbfs >= QUERYSEP_LONG_TAIL_MIN_WINDOWED_PROXY_RMS_DBFS` reject-ятся при
  `abs(correlation) > QUERYSEP_LONG_TAIL_MAX_WINDOWED_PROXY_CORRELATION`;
- `other` bus имеет дополнительный подозрительный диапазон `possible_other_proxy_leakage`: если
  `max(full_track_abs_correlation, windowed_abs_correlation)` выше
  `QUERYSEP_LONG_TAIL_MAX_POSSIBLE_OTHER_PROXY_CORRELATION` и output не слишком тихий относительно
  `other` (`QUERYSEP_LONG_TAIL_MIN_POSSIBLE_OTHER_PROXY_RMS_RATIO_DB`), stem отклоняется;
- если proxy reference имеет другой sample rate, gate должен отклонить результат или явно нормализовать sample rate
  до сравнения. Silent skip запрещён;
- при отсутствии isolated ground truth не писать SDR/SI-SDR, а явно маркировать это как unconfirmed.

Если long-tail output не проходит этот gate, результат должен получить rejected/failed manifest, а не `completed`.

## DRUM bus-projection body-gain candidate

`ml_worker_improved/tools/drum_bus_projection_body_gain.py` is the repeatable Docker-only
route for a DRUM post-candidate when a model drum stem has audible non-drum leakage and a
cleaned/debleed version risks losing drum body.

Contract:

- inputs are explicit model stems: base drums, body-reference drums, and one or more non-drum
  leakage buses such as `other`, `piano`, or `guitar`;
- the tool builds `drums_bus_projection_debleed_intermediate.flac` and
  `drums_bus_projection_dynamic_body_gain.flac`;
- non-drum leakage is reduced by complex STFT projection against the provided leakage buses;
- body gain is dynamic and gated by low-noise/percussive-body evidence, not by track names or
  fixed timestamps;
- `--validation-window label:start:end` values are report-only and must not affect output audio;
- the manifest must keep `accepted_by_listening=false`, `fallback_used=false`, and
  `not_timestamp_hardcoded=true`;
- final output metrics and `sha256` must be computed after re-reading the written FLAC, not from
  in-memory audio before peak/headroom handling;
- if dynamic body gain would require broad peak/headroom reduction, the tool must reject that
  body-gain candidate and select the debleed intermediate with an explicit `selection.reason`;
- the listen pack is diagnostic until human listening confirms that leakage is lower and drum
  body is not muted.

Example:

```powershell
docker compose run --rm --no-deps -T ml-worker python /app/tools/drum_bus_projection_body_gain.py `
  /app/lab_artifacts/golden_audio/<run>/htdemucs_6s/drums.flac `
  /app/lab_artifacts/golden_audio/<run>/hybrid_bs/drums.flac `
  /app/lab_artifacts/golden_audio/<out>/drum_body_route `
  --leakage-stem-path /app/lab_artifacts/golden_audio/<run>/htdemucs_6s/other.flac `
  --leakage-stem-path /app/lab_artifacts/golden_audio/<run>/htdemucs_6s/piano.flac `
  --leakage-stem-path /app/lab_artifacts/golden_audio/<run>/htdemucs_6s/guitar.flac `
  --raw-ref-path /app/lab_artifacts/golden_audio/<source>/instrumental_resurrection_minus.flac `
  --validation-window reported_leakage_60_70:60:70 `
  --validation-window reported_drum_attenuation_130_142:130:142 `
  --listen-dir /app/lab_artifacts/golden_audio/<out>/listen_drum_body_route
```

## DRUM model-zoo segment router

`ml_worker_improved/tools/route_drum_model_zoo_segments.py` is the repeatable Docker-only
route when one global drum model candidate is not stable across the whole track.

Contract:

- inputs are explicit model outputs, not source-track timestamps or per-track hardcodes;
- each candidate may provide its own non-drum leakage references;
- the router scores every fixed-size segment by percussive-vs-harmonic ratio, correlation
  to non-drum references, quiet-candidate penalty, and body energy;
- selected segments are crossfaded to avoid hard model-boundary clicks;
- the manifest must keep `accepted_by_listening=false`, `fallback_used=false`, and
  `not_timestamp_hardcoded=true`;
- the output is still a candidate until listening confirms lower melody leakage and preserved
  drum body.

Example:

```bash
docker compose run --rm --no-deps -T ml-worker python /app/tools/route_drum_model_zoo_segments.py \
  /app/lab_artifacts/golden_audio/<out>/segment_router \
  --candidate htdemucs_6s=/app/lab_artifacts/golden_audio/<run>/htdemucs_6s/drums.flac \
  --candidate mdx_extra=/app/lab_artifacts/golden_audio/<run>/mdx_extra/drums.flac \
  --leakage-ref htdemucs_6s=/app/lab_artifacts/golden_audio/<run>/htdemucs_6s/other.flac \
  --leakage-ref mdx_extra=/app/lab_artifacts/golden_audio/<run>/mdx_extra/other.flac \
  --listen-dir /app/lab_artifacts/golden_audio/<out>/listen_model_zoo_final
```
