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
