# Adaptive Vocal Isolation

## Назначение

`ml_worker_improved/tools/adaptive_vocal_isolation.py` - Docker-only controller для
извлечения отдельного vocal stem без ручного выбора одного пресета.

Production-adaptive contract: controller должен работать как общий route для разных треков.
Запрещены per-track hardcodes, fixed timestamps и special-case ветки в production path. Если
качество недостаточно, расширяйте candidates/router/scoring/diagnostics/feedback contract, а не
добавляйте ручной фикс под один файл.

Цель controller:

- сохранить окончания слов, дыхание, согласные и хвосты фраз;
- уменьшить backing bleed, kick/punch и низкочастотный мусор в vocal stem;
- выбрать лучший candidate по repeatable proxy metrics;
- использовать issue-aware segment selection, если detector нашёл окна, где другой candidate лучше;
- строить segment issue repair candidate для unresolved detector labels без fixed timestamps;
- сохранить feedback-ready manifest для следующих запусков.

Это не fine-tune нейросети и не objective SDR/SI-SDR benchmark без verified isolated
ground-truth vocals.

## Docker-only запуск

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/adaptive_vocal_isolation.py `
  /host_audio/input.mp3 `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation `
  --feedback-store /app/lab_artifacts/golden_audio/adaptive_vocal_feedback.jsonl
```

Если файл лежит вне project mount, добавьте read-only bind mount:

```powershell
docker compose run --rm --no-deps `
  -v "C:\Users\PBAX\Downloads:/host_downloads:ro" `
  --entrypoint python ml-worker `
  /app/tools/adaptive_vocal_isolation.py `
  /host_downloads/source.mp3 `
  /app/lab_artifacts/golden_audio/source_adaptive_vocal_isolation `
  --listen-dir /app/lab_artifacts/golden_audio/source_listen_adaptive_vocal_isolation `
  --feedback-store /app/lab_artifacts/golden_audio/adaptive_vocal_feedback.jsonl
```

## Candidate profiles

Controller по умолчанию запускает:

- `hybrid_tail_preserve` - главный кандидат для сохранения окончаний слов и хвостов;
- `hybrid_balanced_clean` - компромисс между сохранением vocal tails и чисткой backing bleed;
- `ft_tail_safe` - Demucs-first continuity candidate, если строгие модели съедают окончания;
- `bs_artifact_clean` - BS-RoFormer-first clean candidate, если нужен более сухой vocal.

После базовых candidates controller строит derived-кандидаты без повторного model inference:

- `ft_tail_safe_anchor_clean` - берёт приятную vocal continuity из `ft_tail_safe`, но подавляет
  uplifter/noise/synth/robot-like leakage через `bs_artifact_clean` как vocal anchor;
- `ft_tail_safe_anchor_strict` - более строгая версия, если clean-вариант всё ещё тянет non-vocal
  textures.

После anchor-suppressed candidates controller строит `ft_anchor_clean_vocal_body_repair`.
Этот candidate берёт `ft_tail_safe_anchor_clean` как чистую базу и дозаполняет просевшее тело вокала
consensus-донором из `bs_artifact_clean`, `hybrid_tail_preserve` и `hybrid_balanced_clean`.
Заполнение ограничено vocal-core band, чтобы не возвращать high-band uplifter/noise/robot textures.

Для загруженных drop-участков controller также строит `ft_anchor_clean_phase_safe_drop_repair`.
Это section-aware/frame-aware candidate: он поднимает vocal-core magnitude только в местах,
где чистая база просела, а donor consensus подтверждает vocal body. Phase при этом сохраняется
от `ft_tail_safe_anchor_clean`, поэтому candidate не смешивает complex phase разных моделей и
снижает риск watery/flanger-like артефактов. High-band noise не бустится этим слоем.

После этого controller строит segment-router candidates. Это не новый ML inference, а
derived routing поверх уже готовых vocal candidates:

- clean base: `ft_tail_safe_anchor_clean`;
- strict/noise guard: `ft_tail_safe_anchor_strict` + `bs_artifact_clean`;
- body/drop repair: `ft_anchor_clean_phase_safe_drop_repair`;
- tail/continuity donor: `hybrid_tail_preserve`.

Router принимает решение на time-frequency уровне. В vocal-core band он мягко подмешивает
body-preserve magnitude только там, где clean base просела, а strict/anchor candidates подтверждают
vocal body. Raw busy/drop activity сама по себе не считается доказательством вокала: если support
от anchor слабый, busy/drop участок damped и donor energy не вытягивается в vocal stem.
В high band он, наоборот, тянет magnitude к strict/anchor candidates, если энергия не подтверждается
clean vocal anchor. Phase остаётся от clean base, поэтому router не делает complex phase crossfade
между разными моделями.

Сейчас строятся два router-профиля для A/B:

- `segment_adaptive_vocal_router` - baseline busy-safe route, более консервативен против synth/noise bleed;
- `busy_vocal_supported_router` - recovery route для dense drop, где strict/anchor всё ещё несут
  partial speech evidence; он сильнее восстанавливает vocal body, но сохраняет unsupported busy/noise damping.

2026-05-05 consistency update:

- `quality_consistent_ir_router` - IR-residual-supported router для более ровного качества по всему треку.
  Он использует более высокий strict/anchor support floor, не делает raw-busy boost, сильнее damped unsupported
  busy/noise и мягче подмешивает body repair, чтобы агрессивный drop-профиль не портил спокойные и средние секции.
- `--segment-consistency-sec` включает per-window consistency scoring. Значение по умолчанию `5.0`; `0` отключает
  этот слой только для synthetic tests или узкой отладки.
- После primary scoring controller строит `issue_aware_segment_selection` как explicit derived candidate.
  Он не знает имён треков и не использует fixed timestamps: окна берутся только из `segment_issue_report`.
  Для каждого problematic window layer выбирает существующий candidate с меньшим adjusted issue score,
  склеивает выбранные окна с crossfade, пишет `window_decisions` в manifest, затем заново считает proxy
  metrics/issues и участвует в обычном ranking. Если улучшенного окна нет, candidate фиксируется как skipped.
- После issue-aware selection controller может построить `segment_issue_repair` как второй explicit
  derived candidate. Он берёт unresolved labels из текущего best candidate, выбирает donor по severity parts
  для vocal recovery, а bleed/noise-only окна отправляет через strict/anchor-supported debleed route без
  donor body fill. Core vocal-loss labels включают controlled body/tail recovery; bleed/noise labels
  запускают core debleed и high-band strict cleanup даже без core vocal-loss. Если оба типа labels есть в
  одном окне, router совмещает мягкое восстановление вокала с более осторожным debleed, чтобы не превращать
  dropout в hard gate. `backing_vocal_loss_risk` остаётся diagnostic risk без isolated backing ground truth и
  сам по себе не запускает donor repair. Candidate выигрывает только после повторного пересчёта proxy
  metrics/issues.
- После `segment_issue_repair` controller может построить `no_vocal_leakage_cleanup` как explicit derived
  candidate для окон, где candidate consensus не даёт надёжного vocal evidence. Это не fixed timestamps:
  решения берутся из `segment_issue_report`, severity consensus и low-energy/low-kick bridge-окон, соседних с consensus
  no-vocal окном. Layer подавляет только выбранный vocal stem в таких окнах, пропускает
  `backing_vocal_loss_risk`; `local_vocal_body_dip` сам по себе не считается no-vocal сигналом.
- `--reuse-existing-candidates` теперь отклоняет stale manifests, если `audio_path`, source SHA-256,
  `preview_window_sec`, candidate mode, tuning, локальный `stems.vocals.path` или dependency fingerprints
  не совпадают с текущим запросом. Для derived/body/router candidates fingerprint включает SHA-256
  dependency manifests и vocal files, поэтому downstream reuse не переживает скрытую перезапись upstream
  candidate на том же пути. Отклонения пишутся в `adaptive_vocal_manifest.json.reuse_rejected_profiles`.

Отключить этот слой можно флагом:

```powershell
--disable-anchor-suppression
--disable-body-repair
--disable-segment-router
--disable-issue-aware-segment-selection
--disable-segment-issue-repair
--disable-no-vocal-leakage-cleanup
```

Ограничить search space можно повторяющимся флагом:

```powershell
--profile hybrid_tail_preserve --profile hybrid_balanced_clean
```

Если часть candidates уже посчитана и нужно расширить тот же run, используйте:

```powershell
--reuse-existing-candidates
```

## IR Residual And External Candidates

2026-05-05 update: `adaptive_vocal_isolation.py` теперь может расширять search space без нового hardcoded inference:

- `--ir-instrumental-minus-path <path>` строит `ir_residual_vocals` по формуле
  `source_window - instrumental_resurrection_minus_window`. Это proxy vocal candidate, а не verified isolated vocal.
- `--external-vocal-candidate NAME=PATH` добавляет precomputed vocal stem как candidate. Используйте это для
  MDX/MelBand/model-zoo outputs, когда они уже посчитаны отдельными Docker-only wrappers.
- `ir_residual_anchor_clean` строится как debleed gate поверх `ir_residual_vocals` через `bs_artifact_clean`.
- `ir_residual_supported_body_repair` использует IR residual как donor для dense/drop vocal-body recovery, но сохраняет
  clean-base phase и BS support в donor consensus.
- `ir_residual_supported_router` добавляет IR-residual-supported repair leg в segment router. Если IR residual не
  передан, этот router profile фиксируется как skipped dependency, а базовый вокальный pipeline работает как раньше.

Docker-only пример с IR residual:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/adaptive_vocal_isolation.py `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_clean/raw_ref.flac `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation_ir_residual `
  --ir-instrumental-minus-path /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_clean/instrumental_resurrection_minus.flac `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_ir_residual `
  --feedback-store /app/lab_artifacts/golden_audio/adaptive_vocal_feedback.jsonl
```

Docker-only пример с готовыми MDX/MelBand candidates:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/adaptive_vocal_isolation.py `
  /host_downloads/source.mp3 `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation_model_zoo `
  --external-vocal-candidate mdx23c=/app/lab_artifacts/golden_audio/<track>_mdx23c/vocals.flac `
  --external-vocal-candidate melband=/app/lab_artifacts/golden_audio/<track>_melband/vocals.flac `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_model_zoo
```

`adaptive_vocal_manifest.json.external_candidate_profiles` фиксирует такие inputs отдельно от встроенных BS/FT/hybrid
profiles. IR residual имеет специальные derived profiles для debleed/body/router. Generic external candidates вроде
MDX/MelBand участвуют в общем scoring/listen pack напрямую; для их участия в derived router/debleed path нужен отдельный
явный derived profile. Этот слой не превращает IR residual, MDX или MelBand в ground truth.

## Metrics и выбор

`adaptive_vocal_manifest.json` содержит:

- `candidate_profiles`;
- `external_candidate_profiles`;
- sorted `candidates`;
- `selected_profile`;
- `selected_score`;
- `selected_artifacts`;
- `learning_contract`;
- `body_repair_candidate_profiles`;
- `skipped_body_repair_profiles`;
- `segment_router_candidate_profiles`;
- `skipped_segment_router_profiles`;
- `issue_aware_segment_selection_profile`;
- `skipped_issue_aware_segment_selection`;
- `segment_issue_repair_profile`;
- `skipped_segment_issue_repair`;
- `post_gate_residual_issue_repair_profile`;
- `skipped_post_gate_residual_issue_repair`;
- `no_vocal_leakage_cleanup_profile`;
- `skipped_no_vocal_leakage_cleanup`;
- `selected_segment_issue_report`;
- `detected_issue_action_contract`;
- `defect_effectiveness_gate`;
- `issue_effectiveness_ledger`;
- `reuse_rejected_profiles`.

Proxy metrics:

- `dropout_frame_ratio`;
- `tail_dropout_ratio`;
- `candidate_to_source_rms_db`;
- `low_band_ratio_db_35_180`;
- `kick_punch_ratio_db_250_900`;
- `presence_ratio_db_1800_4500`;
- `air_ratio_db_4500_9000`;
- `unsupported_vs_anchor_ratio_db`;
- `high_unsupported_vs_anchor_ratio_db`;
- `high_noise_flatness_6000_16000`;
- `stereo_side_mid_ratio_db`;
- `segment_consistency_worst_dropout_frame_ratio`;
- `segment_consistency_worst_tail_dropout_ratio`;
- `segment_consistency_dropout_std`;
- `segment_consistency_tail_dropout_std`;
- `segment_consistency_rms_ratio_range_db`;
- `source_stereo_side_mid_ratio_db`;
- `candidate_stereo_side_mid_ratio_db`;
- `backing_vocal_side_loss_db`;
- `segment_issue_max_score`;
- `segment_issue_mean_top3_score`;
- `segment_issue_count`;
- `segment_issue_window_ratio`;
- `segment_issue_repair_window_count`;
- `segment_issue_repair_recovery_improvement_sum`;
- `segment_issue_repair_issue_improvement_sum`;
- `no_vocal_cleanup_window_count`;
- `no_vocal_cleanup_confidence_sum`;
- `no_vocal_cleanup_suppressed_duration_sec`.

2026-05-05 issue diagnostics update: controller теперь пишет `segment_issue_report`
для каждого candidate и `selected_segment_issue_report` для выбранного vocal stem.
Отчёт ранжирует problematic windows и размечает labels: `vocal_dropout`,
`vocal_tail_loss`, `thin_or_overcleaned`, `leaky_full_mix_risk`,
`low_band_bleed`, `kick_punch_bleed`, `unsupported_anchor_leakage`,
`high_band_unsupported_leakage`, `high_noise_or_robotic_artifact`,
`excessive_reverb_tail_smear`, `wide_stereo_non_vocal_leakage`,
`backing_vocal_loss_risk`, `local_vocal_body_dip`, `presence_loss`,
`air_detail_loss`. `excessive_reverb_tail_smear` помечает окна, где tail/air context
не выглядит как hard dropout, но high-band noise и wide stereo residue дают слышимый
reverb/tail smear. Такие окна защищены от no-vocal cleanup suppression и направляются
в guard/repair/router actions. Backing vocals считаются vocal content; `backing_vocal_loss_risk`
помечает окна, где source имеет широкую side-информацию, а selected vocal stem
становится существенно уже. Это proxy-риск, а не доказательство isolated backing vocal
без ground truth.
`--segment-issue-report-max-windows` ограничивает число окон в manifest,
а `--segment-issue-listen-clip-count` пишет top selected-candidate windows
в `listen_dir/segment_issues/` как пары raw/vocal clips.

2026-05-05 issue-aware selection update: diagnostics теперь также управляют
`issue_aware_segment_selection`. Это не post-hoc silent repair выбранного файла:
controller пишет новый candidate в `candidates/issue_aware_segment_selection/`,
фиксирует base profile, selected profiles, `window_decisions`, dependency fingerprints,
crossfade и затем пересчитывает metrics/issues. Финальный `01_adaptive_selected_vocals.flac`
может указывать на этот candidate только если он выиграл обычный scoring.

2026-05-06 segment issue repair multi-issue update: если после issue-aware selection остаются
problematic windows, controller может построить `segment_issue_repair`. Окна и действия
берутся только из `segment_issue_report.labels`: core vocal-loss labels включают controlled
donor body/tail recovery. `local_vocal_body_dip` помечает supported local body/tail dip, где окно
не выглядит как unsupported leakage, но vocal stem локально просел относительно source. Bleed/noise labels
теперь также создают `bleed_control` decisions для окон без core vocal-loss: body/tail donor fill выключается,
а STFT-router тянет unsupported mid/low/high leakage к strict/anchor reference. Окна с recovery и leakage
получают `combined_recovery_and_bleed_control`: body/tail recovery остаётся включённым, но core debleed
ограничивается более мягкой силой. `backing_vocal_loss_risk` не считается isolated backing ground truth и сам
по себе не запускает repair. Это explicit candidate в `candidates/segment_issue_repair/`, не silent mutation
и не per-track timestamp rule.

2026-05-06 no-vocal leakage cleanup update: если после repair текущий best candidate всё ещё содержит
окна, где detector помечает vocal-loss/noise, но consensus всех candidates не даёт надёжного vocal evidence,
controller строит explicit candidate `no_vocal_leakage_cleanup` в `candidates/no_vocal_leakage_cleanup/`.
Он не знает названия трека и не использует fixed timestamps: решения берутся из `segment_issue_report`,
severity consensus и low-energy/low-kick bridge окон, соседних с consensus no-vocal окном. Layer подавляет vocal stem
только в таких окнах, фиксирует `window_decisions`/gain/crossfade в manifest и затем пересчитывает
metrics/issues. `backing_vocal_loss_risk` защищает реальные/возможные backing vocal parts, а
`local_vocal_body_dip`, `vocal_dropout` и `vocal_tail_loss` не запускают no-vocal suppression; они защищают
окно от cleanup, потому что означают риск потери вокального содержимого.
Scoring даёт этому candidate ограниченный no-vocal cleanup bonus, потому что intentional silence в таких окнах
иначе выглядит для общего dropout detector как ухудшение.

Эти метрики нужны для repeatable ranking, а не для заявления studio-grade objective
quality без oracle stems.

## Vocal Harshness Guard

`ml_worker_improved/tools/apply_vocal_harshness_guard.py` - Docker-only postprocess layer
для vocal candidate, когда base сохраняет тело вокала, но в отдельных diagnostic windows слышен
режущий high-band / robotic noise.

`adaptive_vocal_isolation.py` также строит `vocal_harshness_guard` как ranked derived candidate
после `no_vocal_leakage_cleanup`. Поэтому detector-marked high-band issues больше не остаются
только отчётом: они создают candidate, участвуют в scoring и попадают в `candidate_XX_*` listen pack.

Contract:

- это explicit derived artifact/candidate, а не silent mutation основного selected output;
- решения берутся только из `segment_issue_report` выбранного base candidate;
- target labels: `high_noise_or_robotic_artifact`, `high_band_unsupported_leakage`,
  `unsupported_anchor_leakage`, `excessive_reverb_tail_smear`;
- vocal-loss labels (`vocal_dropout`, `vocal_tail_loss`, `local_vocal_body_dip`,
  `backing_vocal_loss_risk`) уменьшают силу guard, чтобы не превращать просевший вокал в hard gate;
- unlabelled windows не трогаются, поэтому участки, где base звучит лучше alternate candidate,
  не заменяются целиком другим профилем;
- обработка идёт только по high band, phase сохраняется от base candidate.

## Detected Issue Action Contract

`adaptive_vocal_manifest.json.detected_issue_action_contract` фиксирует правило:
каждый detected issue label выбранного result должен иметь либо action candidate, либо explicit
rejection reason.

`adaptive_vocal_manifest.json.decision_contract` is the control-plane decision
derived from `detected_issue_action_contract`, `defect_effectiveness_gate`, and
`vocal_next_action_plan`. The same contract is mirrored into
`selected_delivery_gate_manifest.json.decision_contract` so downstream tools do
not need to reinterpret accepted/rejected/hard-case state from raw fields.

Action candidates:

- vocal recovery / body dips: `segment_issue_repair`, `issue_aware_segment_selection`;
- bleed / kick / low-band / wide leakage: `segment_issue_repair`, `issue_aware_segment_selection`;
- high-band harshness / robotic / unsupported / reverb-tail-smear leakage: `vocal_harshness_guard`,
  `segment_issue_repair`, `issue_aware_segment_selection`;
- no-vocal leakage: `no_vocal_leakage_cleanup`, когда есть consensus no-vocal evidence.

Bounded thin/over-cleaned repair:

- `--bounded-thin-repair-only` is the explicit runtime contract for
  `thin_or_overcleaned` / `presence_loss` follow-up runs;
- it disables anchor-suppression, segment router, issue-aware segment selection,
  segment issue repair, no-vocal leakage cleanup, vocal harshness guard,
  post-gate residual repair, gate-blocker best-segment splice, and clustered
  issue-window ratio repair;
- it keeps only direct body repair
  `ft_tail_safe_direct_body_presence_repair` and gate-blocker
  `gate_blocker_presence_band_recovery`;
- the direct body repair uses `ft_tail_safe` as the base and does not require
  `ft_tail_safe_anchor_clean`, so the route does not need a stricter
  anchor-clean/debleed candidate to restore vocal body;
- manifests write `bounded_thin_repair_only=true` and
  `bounded_thin_repair_contract`; this still does not make an artifact
  deliverable unless the final selected gate has `delivery_allowed=true` and
  `presentation_status=accepted_for_listen_pack`.

Post-gate residual repair:

- after the first scoring pass, the controller builds `defect_effectiveness_gate` for the selected
  candidate;
- if the gate returns `diagnostic_only`, the controller creates one more candidate,
  `post_gate_residual_issue_repair`, from the actually selected blocked candidate instead of a fixed
  profile or timestamp;
- the candidate uses the same segment issue reports, action labels and dependency fingerprints, plus
  the selected profile, score and gate blockers fingerprint;
- the controller then recalculates ranking, `detected_issue_action_contract`,
  `defect_effectiveness_gate` and `issue_effectiveness_ledger`;
- the ledger now also records per-window `action_effectiveness`: best action candidate, severity delta,
  resolved labels and introduced labels. This prevents a detected issue from being treated as handled
  only because a candidate existed; the candidate must measurably improve the issue window or remain
  diagnostic.
- if the repair is not run, the manifest writes an explicit reason in
  `skipped_post_gate_residual_issue_repair`: `selected_candidate_release_ready`,
  `disabled_by_segment_issue_repair_flag`, `segment_consistency_disabled`,
  `missing_candidate_issue_reports` or `no_post_gate_repairable_issue_window`.

2026-05-10 debugging/performance update:

- `adaptive_vocal_manifest.json.stage_profiler` records per-stage timings and heartbeat events for
  runtime diagnosis. Heartbeats are JSON lines on stderr and can be disabled with
  `--disable-stage-heartbeat`; the manifest field is still written.
- `adaptive_vocal_manifest.json.analysis_cache` records per-run source-window and candidate-audio
  cache reads/hits. The cache is provenance-bound to file path, size and mtime, and is used only inside
  one controller run.
- Proxy metrics, segment consistency reports, segment issue reports and anchor-support checks can share
  decoded audio through the cache. This reduces repeated disk reads while preserving the same scoring
  contracts.

Protective rejection:

- `backing_vocal_loss_risk` не запускает suppression сам по себе, потому что backing vocals считаются
  vocal content. Без isolated backing-vocal ground truth это diagnostic/protective label, а не сигнал
  на удаление.

Если action candidate не построился, contract пишет `candidate_unavailable` с причиной из
`skipped_*`, например `no_harshness_guard_window`, `missing_candidate_issue_reports` или
`disabled_by_flag`. Это не fallback и не per-track timestamp rule.

Runtime statuses:

- `pass` - все detected labels имеют action candidate, explicit protective rejection или documented
  `candidate_unavailable` reason;
- `missing_issue_report` - выбранный result не имеет segment issue report, поэтому labels не могут
  быть reconciled;
- `uncovered_labels` - detector выдал label, который не привязан ни к action candidate, ни к
  protective rejection. Такой label должен получить production action mapping или explicit rejection
  reason перед тем, как route считать покрытым.

Docker-only пример:

```powershell
docker compose run --rm --no-deps -T --entrypoint python ml-worker `
  /app/tools/apply_vocal_harshness_guard.py `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation/adaptive_vocal_manifest.json `
  /app/lab_artifacts/golden_audio/<track>_vocal_harshness_guard `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_vocal_harshness_guard `
  --base-profile issue_aware_segment_selection `
  --anchor-profile ft_tail_safe
```

`--anchor-profile` является high-band reference, а не full-window replacement. Если full-track
candidate глушит vocal body в конкретном окне, его всё ещё можно использовать как high-band ceiling
только в detector-marked harshness windows, не подменяя весь участок.

## Feedback

Feedback store - JSONL. Следующий запуск с тем же `--feedback-store` изменит profile
priors:

```json
{"profile":"hybrid_tail_preserve","rating":"accept","score":0.8}
{"profile":"bs_artifact_clean","rating":"reject","swallowed_endings":true}
{"profile":"hybrid_balanced_clean","rating":"reject","backing_bleed":true}
{"profile":"ft_tail_safe","rating":"reject","backing_vocal_loss":true}
{"profile":"hybrid_balanced_clean","rating":"reject","watery_artifacts":true}
{"profile":"ft_tail_safe","rating":"reject","non_vocal_bleed":true,"uplifter_bleed":true,"synth_bleed":true,"robotic_drop_bleed":true}
{"profile":"ft_tail_safe_anchor_clean","rating":"accept","score":0.8}
{"profile":"bs_artifact_clean","rating":"better","score":1.0,"listener_verified":true,"confidence":1.0,"source_audio_sha256":"<source-sha256>","window_sec":{"start":0.0,"end":30.0}}
```

Feedback меняет selection policy, но не обучает веса модели.

Обычный feedback остаётся слабым profile prior. `listener_verified` / `human_verified`
используется только как сильный perceptual signal, когда запись привязана к конкретному
`source_audio_sha256` и точному `window_sec`. Если hash или окно не совпали с текущим
запуском, такая запись игнорируется. Это защищает controller от глобального хардкода
профиля под один трек и сохраняет repeatable provenance в `adaptive_vocal_manifest.json`
через `source_audio_sha256`, `feedback_store_path` и `feedback_adjustments`.

## Artifacts

```text
adaptive_vocal_manifest.json
01_adaptive_selected_vocals.flac
candidates/<profile>/vocals.flac
candidates/<profile>/stem_manifest.json
candidates/issue_aware_segment_selection/vocals.flac
candidates/issue_aware_segment_selection/stem_manifest.json
```

Legacy calibration backups must not stay under `candidates/`. Move old
`segment_adaptive_vocal_router_pre_support_gate_*` snapshots to
`archived_candidates/` so broad candidate/report scans only see active profiles.

Listen pack:

```text
00_raw_ref.flac
01_adaptive_selected_vocals.flac
candidate_01_<profile>.flac
candidate_02_<profile>.flac
candidate_map.json
segment_issues/segment_issue_clips.json
segment_issues/issue_01_<labels>_raw_<start>_<end>.flac
segment_issues/issue_01_<labels>_selected_vocals_<start>_<end>.flac
```

Выбранный `01_adaptive_selected_vocals.flac` можно использовать как вход для
`build_vocal_guided_minus.py`, если нужно построить minus по схеме `raw mix - selected vocals`
с optional blend к существующему base minus. Это отдельный minus-layer contract: vocal controller
не меняет минус сам и не подменяет `export_instrumental_resurrection_minus.py`.

Для focused A/B проверки проблемного участка можно сохранять отдельный listen pack с 30-секундными
clips в `lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_first30_<date>`.
Такие clips являются diagnostic artifacts; они не должны превращаться в per-track production
thresholds или hardcoded routing.
Такой pack должен содержать raw reference, selected candidate и ближайшие альтернативы, а также
`first30_metrics.json` с dropout/leakage proxy metrics.

## Model-zoo first30 rebuild

Если adaptive pack даёт почти одинаковые варианты или не чистит non-vocal мусор, следующий
repeatable шаг - построить model-zoo кандидатов из реально разных backbones:

- `export_mdx23c_vocal_preview.py` - MDX23C vocal checkpoint;
- `export_melband_vocal_preview.py` - Mel-Band RoFormer vocal checkpoint;
- `build_vocal_model_zoo_candidates.py` - postprocess поверх готовых first30 outputs:
  - сохраняет старый `bs_artifact_clean` как baseline;
  - добавляет MDX23C и MelBand;
  - строит MelBand debleed через instrumental-guided DSP;
  - строит multi-model support-gate candidates, где primary vocal подавляется в TF-зонах,
    которые не подтверждаются другими моделями.

Пример postprocess после отдельных inference-запусков:

```powershell
docker compose run --rm --no-deps `
  -v "C:\Users\PBAX\Downloads:/host_downloads:ro" `
  --entrypoint python ml-worker `
  /app/tools/build_vocal_model_zoo_candidates.py `
  --source-audio-path /host_downloads/source.mp3 `
  --raw-ref-path /app/lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_first30_<date>/00_raw_ref_first30.flac `
  --output-dir /app/lab_artifacts/golden_audio/<track>_vocal_model_zoo_first30_<date>/postprocessed `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_vocal_model_zoo_post_first30_<date> `
  --bs-vocals-path /app/lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_first30_<date>/03_bs_artifact_clean_first30.flac `
  --mdx-vocals-path /app/lab_artifacts/golden_audio/<track>_vocal_model_zoo_first30_<date>/mdx23c/vocals.flac `
  --melband-vocals-path /app/lab_artifacts/golden_audio/<track>_vocal_model_zoo_first30_<date>/melband_plain/vocals.flac `
  --melband-instrumental-path /app/lab_artifacts/golden_audio/<track>_vocal_model_zoo_first30_<date>/melband_plain/instrumental.flac
```

Выход:

```text
model_zoo_candidate_manifest.json
model_zoo_metrics.json
model_zoo_delta_metrics.json
01_old_bs_artifact_clean_first30.flac
02_mdx23c_first30.flac
03_melband_plain_first30.flac
04_melband_debleed_first30.flac
05_bs_supported_by_mdx_first30.flac
06_bs_supported_by_mdx_melband_first30.flac
07_melband_supported_by_bs_mdx_first30.flac
08_melband_drop_safe_first30.flac
09_melband_drop_body_first30.flac
10_melband_drop_air_guard_first30.flac
11_10_focus_25_30_vocal_air_detail.flac
12_10_focus_25_30_vocal_air_plus.flac
13_11_focus_25_30_vocal_body_plus.flac

```

`08`-`10` сохраняют направление `07` (MelBand primary + BS/MDX support), но поднимают
active/vocal-core floors и снижают mask aggressiveness. Их назначение - сложные drop-участки,
где `07` звучит чисто, но слишком глушит голос.

Ограничение: этот слой не обучает веса модели. Он расширяет adaptive search space реальными
backbones и повторяемым DSP/TF gating. Если `BS-Mamba2` checkpoint есть, но контейнер падает на
`ModuleNotFoundError: No module named 'mamba_ssm'`, это Docker dependency defect, а не audio-quality
результат.
## 2026-05-06 no-vocal cleanup safety boundary

`no_vocal_leakage_cleanup` не должен использовать `vocal_dropout`, `vocal_tail_loss`,
`local_vocal_body_dip` или `backing_vocal_loss_risk` как основание для подавления
vocal stem. Эти labels означают риск потери вокального содержимого, а не доказательство
того, что вокала в окне нет.

No-vocal suppression разрешён только для окон с независимым low-energy + low-kick
evidence при отсутствии vocal-loss protection labels. `thin_or_overcleaned`,
`presence_loss` и `air_detail_loss` сами по себе не доказывают отсутствие вокала:
они лишь участвуют в cleanup, если окно уже прошло low-energy/low-kick gate.
Это защищает congested/drop участки от эффекта hard gate, когда просевший голос
ошибочно подавлялся до `suppression_gain`.
