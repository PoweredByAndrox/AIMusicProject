# Flexible Minus Repair

## Назначение

`ml_worker_improved/tools/apply_flexible_minus_repair.py` - Docker-only post-processing stage для instrumental/minus-кандидата, когда модель уже хорошо убрала вокал, но во vocal-window участках слышны провалы ambience, background noise, uplifter/downlifter или kick punch/attack.

Это не objective SDR/SI-SDR gate и не замена source separation модели. Скрипт пишет repeatable FLAC/JSON артефакты и focus clips для A/B прослушки.

## Safety contract

- Clean-minus baseline по умолчанию должен быть `pure-base`: `02` равен `base_minus`, без ambience repair,
  texture floor inpainting, raw-kick repair и любых synthetic/noise layers.
- `--pure-base` является repeatable Docker-only gate для чистого минуса без добавленного шума. Если пользователь
  слышит artifact/noise layer в `03` или `04`, такой repair нельзя публиковать как clean-minus baseline.
- Default profile является vocal-safe self-only: repair строится из `base_minus`, без подмешивания `raw full mix`.
- Ambience repair и texture floor inpainting используют только `base_minus`.
- Ambience repair и texture floor inpainting теперь opt-in repair-кандидаты, а не clean-minus default.
- `raw full mix` можно использовать только через явный экспериментальный флаг `--enable-raw-kick-repair`.
- Если в `03_added_layer_only.flac` слышен вокал, raw-kick profile нельзя считать безопасным baseline.
- Без verified isolated stems этот workflow не заявляет SDR/SI-SDR/SIR/SAR.
- Exit code отражает runtime-ошибки, а не subjective quality verdict.

## Clean-minus baseline guard

`clean-minus` baseline строится через `export_instrumental_resurrection_minus.py`, то есть через
`bs_roformer_instrumental_resurrection_unwa` с `target_instrument=other`. После этого допускается только self-only
artifact repair из `base_minus`.

2026-05-05 production-adaptive update: clean-minus baseline больше не должен добавлять synthetic
ambience/texture layer автоматически на любом треке. Для чистого минуса используйте
`apply_flexible_minus_repair.py --pure-base` или adaptive profile `pure_base`. `strict_no_texture`,
`soft_delta` и `vocal_safe_balanced` остаются audit/experiment кандидатами для A/B, но не заменяют
pure-base baseline без явного решения.

Исторические упоминания конкретных треков в этом документе являются examples/checkpoints только для
регрессии и прослушки. Они не являются нормативными per-track правилами и не разрешают hardcoded
timestamps, EQ или special-case ветки в production route.

`build_vocal_guided_minus.py` не является clean-minus baseline. Он использует `selected_vocals_path` и строит
`raw mix - selected vocals`, поэтому пропущенные vocal-фрагменты в upstream vocal stem останутся в минусе. Этот слой
разрешён только как отдельный audit/experiment workflow и не должен заменять Instrumental Resurrection baseline.

## Docker-only запуск

Vocal-safe profile:

Pure-base clean baseline:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/apply_flexible_minus_repair.py `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_clean/raw_ref.flac `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_clean/instrumental_resurrection_minus.flac `
  /app/lab_artifacts/golden_audio/<track>_clean_minus_pure_base `
  --pure-base `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_clean_minus_pure_base `
  --listen-label 02_clean_minus_no_repair.flac
```

Experimental repair profile example:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/apply_flexible_minus_repair.py `
  /app/lab_artifacts/golden_audio/falling_instrumental_resurrection_isolated_20260502/raw_ref.flac `
  /app/lab_artifacts/golden_audio/falling_instrumental_resurrection_isolated_20260502/instrumental_resurrection_minus.flac `
  /app/lab_artifacts/golden_audio/falling_flexible_minus_repair_vocal_safe_20260502 `
  --listen-dir /app/lab_artifacts/golden_audio/falling_listen_flexible_repair_vocal_safe_20260502 `
  --listen-label 02_flexible_vocal_safe_minus.flac `
  --focus-window intro:0:35 `
  --focus-window intro_dropout_4_5:4:5 `
  --focus-window kick_130_140:90:100
```

Raw-kick recovery оставлен только как диагностический эксперимент:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/apply_flexible_minus_repair.py `
  /app/lab_artifacts/golden_audio/falling_instrumental_resurrection_isolated_20260502/raw_ref.flac `
  /app/lab_artifacts/golden_audio/falling_instrumental_resurrection_isolated_20260502/instrumental_resurrection_minus.flac `
  /app/lab_artifacts/golden_audio/falling_flexible_minus_repair_raw_kick_experimental_20260502 `
  --listen-dir /app/lab_artifacts/golden_audio/falling_listen_flexible_repair_raw_kick_experimental_20260502 `
  --listen-label 02_flexible_raw_kick_experimental_minus.flac `
  --enable-raw-kick-repair `
  --kick-strength 1.2 `
  --kick-max-blend 0.18 `
  --focus-window intro:0:35 `
  --focus-window intro_dropout_4_5:4:5 `
  --focus-window kick_130_140:90:100
```

## Что смотреть в output

Основной JSON:

```text
lab_artifacts/golden_audio/<run>/stem_manifest.json
```

Ключевые поля:

- `repair.tuning.enable_raw_kick_repair`
- `metrics.rms_drift_db_vs_base`
- `metrics.added_layer_vs_base_rms_db`
- `metrics.presence_band_ratio_drift_db`
- `metrics.air_band_ratio_drift_db`
- `focus_metrics.intro_dropout_4_5.drift_vs_base.*`
- `focus_metrics.kick_130_140.drift_vs_base.*`

Listen directory:

```text
00_raw_ref.flac
01_base_minus.flac
02_flexible_*.flac
03_added_layer_only.flac
04_added_layer_delta_raw.flac
focus_intro_*.flac
focus_intro_dropout_4_5_*.flac
focus_kick_130_140_*.flac
candidate_map.json
```

`03_added_layer_only.flac` и `focus_*_added_layer.flac` - monitor-friendly слой для прослушки риска. Он band-limited и не используется для сборки `02`.

`04_added_layer_delta_raw.flac` и `focus_*_added_layer_delta_raw.flac` - точный residual `repaired - base_minus` для аудита. Он может звучать фазово/шумно, потому что это не музыкальный stem.

## Archived Falling checkpoint

Этот checkpoint хранит старый regression/listening context и не является production rule.
Для `ytmp3free.cc_falling-youtubemp3free.org.mp3` старые listen packs `safe`, `kick_plus` и `kick_recovery`
считаются raw-risk, если в `03_added_layer_only.flac` или `focus_kick_130_140_added_layer.flac`
слышен вокал.

Новый baseline должен строиться через vocal-safe command выше и обязательно включать focus window `intro_dropout_4_5:4:5`, потому что широкий `intro:0:35` скрывает локальный провал на `00:04-00:05`.

Созданный 2026-05-02 vocal-safe pack:

```text
lab_artifacts/golden_audio/falling_listen_flexible_repair_vocal_safe_20260502
```

Ключевые факты из manifest:

- `repair.name = flexible_minus_repair_v2`
- `repair.tuning.enable_raw_kick_repair = false`
- `metrics.rms_drift_db_vs_base = 0.077122`
- `metrics.added_layer_vs_base_rms_db = -28.314938`
- `artifacts.added_layer_flac` - monitor-friendly artifact
- `artifacts.added_layer_delta_raw_flac` - exact delta artifact
- `focus_metrics.intro_dropout_4_5.drift_vs_base.presence_band_ratio_db = 0.303815`
- `focus_metrics.intro_dropout_4_5.drift_vs_base.air_band_ratio_db = 0.330038`
- `focus_metrics.intro_dropout_4_5.drift_vs_base.added_layer_vs_base_rms_db = -40.112668`

## Adaptive controller

`ml_worker_improved/tools/adaptive_minus_repair.py` - Docker-only orchestration layer над
`apply_flexible_minus_repair.py`. Он не заменяет separation model и не заявляет SDR/SI-SDR без
verified ground truth. Его задача - убрать ручной выбор repair-профиля:

1. прогнать несколько vocal-safe candidate profiles;
2. прочитать `stem_manifest.json` каждого кандидата;
3. посчитать conservative proxy score по drift/added-layer/focus metrics;
4. выбрать лучший кандидат;
5. записать `adaptive_manifest.json`;
6. при наличии JSONL feedback store скорректировать будущий выбор profile.

Первый production-safe feedback loop работает на уровне выбора профиля, а не обучения весов
нейросети. Это намеренное ограничение: без verified isolated stems нельзя честно fine-tune/accept
модель по SDR/SI-SDR.

Docker-only пример:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/adaptive_minus_repair.py `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/raw_ref.flac `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/instrumental_resurrection_minus.flac `
  /app/lab_artifacts/golden_audio/<track>_adaptive_minus_repair `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_adaptive_minus_repair `
  --feedback-store /app/lab_artifacts/golden_audio/adaptive_minus_feedback.jsonl `
  --focus-window intro:0:30 `
  --focus-window problem_kick:90:100
```

Feedback record example:

```json
{"profile":"soft_delta","rating":"accept","score":0.8}
{"profile":"vocal_safe_balanced","rating":"reject","artifact_noise":true}
{"profile":"strict_no_texture","rating":"reject","too_dry":true}
```

Main artifacts:

```text
adaptive_manifest.json
02_adaptive_selected_minus.flac
candidates/<profile>/stem_manifest.json
candidates/<profile>/02_flexible_minus_repaired.flac
```

`adaptive_manifest.json` содержит `candidate_profiles`, sorted `candidates`, `selected_profile`,
`selected_artifacts` и `learning_contract`. Если пользователь помечает профиль как noisy/dry/good,
следующий запуск с тем же `--feedback-store` изменит profile priors без изменения бизнес-логики
нижнего repair-скрипта.

Delivery lock:

- `adaptive_minus_repair.py` is a candidate generator/controller, not the final delivery route.
- `selected_minus_release_gate` remains an internal proxy gate for ranking and downstream evidence.
- The selected adaptive artifact is written as `candidate_status=selected_candidate_only`,
  `delivery_status=candidate_only`, `delivery_allowed=false`,
  `presentation_status=blocked_until_downstream_delivery_gates`,
  `release_status=not_release_ready`, and `requires_post_judge=true`.
- The controller must not copy `02_adaptive_selected_minus.flac` into the listen directory as an
  accepted artifact. A later self-judge/result-judge/listen-preview pipeline must accept it before
  any listen-pack delivery.

## Vocal-guided minus layer

`ml_worker_improved/tools/build_vocal_guided_minus.py` - Docker-only слой для случая, когда уже есть
выбранный vocal stem. Он строит `raw mix - selected vocals`, а если передан `--base-minus-path`,
делает bounded blend между base minus и raw-minus по vocal activity envelope. Это переносит удачный
vocal-selection результат в minus-ветку без изменения существующих separation entrypoints.

Safety contract:

- `01_selected_vocals.flac` копируется как есть после ресемплинга/выравнивания с raw mix;
- `02_raw_minus_from_selected_vocal.flac` является audit-кандидатом `raw - vocal`;
- `03_vocal_guided_minus.flac` является final minus только внутри audit/experiment workflow и не является clean-minus baseline;
- `vocal_guided_minus_manifest.json` теперь явно пишет `candidate_only=true`, `delivery_status=candidate_only`,
  `delivery_allowed=false`, `presentation_status=blocked_until_judged`, `release_status=not_release_ready`
  и `requires_post_judge=true`;
- route пишет structured progress events и помечает `gpu_inference_skipped=true`, потому что использует уже готовые
  vocal/minus candidates и CPU/DSP subtraction, а не neural inference;
- если `--base-minus-path` задан, final minus не заменяет base полностью, а смешивается через
  `base_blend_inactive` и `base_blend_active`;
- без verified isolated stems этот слой остаётся proxy/listening workflow, не SDR/SI-SDR verdict.

Docker-only пример:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/build_vocal_guided_minus.py `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/raw_ref.flac `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation/01_adaptive_selected_vocals.flac `
  /app/lab_artifacts/golden_audio/<track>_vocal_guided_minus `
  --base-minus-path /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/instrumental_resurrection_minus.flac `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_vocal_guided_minus
```

Main artifacts:

```text
vocal_guided_minus_manifest.json
00_raw_ref.flac
01_selected_vocals.flac
02_raw_minus_from_selected_vocal.flac
03_vocal_guided_minus.flac
04_base_minus_reference.flac
```

Manifest metrics:

- `metrics.reconstruction_error_rms_dbfs` измеряет pre-limiter контракт raw subtraction;
- `metrics.post_limiter_reconstruction_error_rms_dbfs` оставляет видимым post-limiter reconstruction drift;
- `metrics.post_limiter_delta_rms_dbfs` показывает, насколько peak limiting изменил raw-minus audit candidate.
