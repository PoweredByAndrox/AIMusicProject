# StemLab Multi-Container Blueprint v2

Каркас проекта для локальной системы тестирования и оркестрации music source separation, fine-grained routing и lyrics sync.

## Что изменилось во второй версии

Во второй версии я не просто оставил orchestration-стенд, а добавил три практических улучшения:

- исправил `bsroformer-worker`, чтобы он работал через **isolated single-file input**, а не сканировал всю папку `/app/data/inputs`;
- добавил отдельный `lyrics-worker` для **автоматической расшифровки вокала** и **таймкодов**;
- протянул во фронтенд **artifact browser + audio player + lyrics viewer**, чтобы можно было слушать stem'ы и сверять текст/тайминги в одном интерфейсе.

## Что это такое

Это **монорепо-скелет** для локальной лаборатории, где:
- разные семейства моделей запускаются в **разных Docker-контейнерах**;
- у всех сервисов есть **единый API и единый UI**;
- можно собирать **иерархический pipeline**: coarse split -> targeted stem routing -> restoration -> lyrics sync -> A/B-тесты;
- можно назначать **свою архитектуру для каждого стэма**;
- можно держать рядом как production-ready сервисы, так и research-ветки.

## Что уже заложено

- `apps/api` — FastAPI gateway + pipeline orchestrator + uploads + artifact browser
- `apps/frontend` — React/Vite UI для тестов, прослушивания и просмотра lyrics
- `services/demucs-worker` — обёртка над Demucs
- `services/bsroformer-worker` — обёртка над BS-RoFormer-Infer
- `services/audio-separator-worker` — **honest_name = bus-mixer-worker**
  (R12). Этот сервис **НЕ** оборачивает PyPI-пакет `audio-separator`
  и не подгружает UVR-модели — это детерминистический ffmpeg-микшер
  стэмов с per-bus EQ. Для реального ML-ensemble см.
  `restoration-worker.stem_ensemble` (R9) и `wiener_postfilter` (R8),
  для long-tail — `querysep-worker` (Banquet, R11).
- `services/querysep-worker` — интерфейс под query-based extraction (Banquet / QSCNet / TUSS)
- `services/restoration-worker` — интерфейс под targeted restoration / cleanup
- `services/lyrics-worker` — локальный ASR/alignment слой для word-level lyrics
- `configs/presets` — пресеты пайплайнов
- `docs` — объяснение архитектуры, плана, фиксов и ограничений

## Что важно понять сразу

Этот архив — **не упакованные веса моделей** и не готовый “one-click magic separator”.
Это **проектная база**, которую можно:
1. поднять в Docker,
2. быстро начать dry-run и частично real-run тесты,
3. дальше расширять под свои модели и свои чекпоинты.

## Что реально работает уже сейчас

### Готово для orchestration и UI
- единый gateway
- jobs
- artifact indexing
- загрузка audio/text через API
- frontend с polling, плеером и lyrics-панелью

### Готово как честный локальный real-run путь
- `lyrics-worker` preset `lyrics_sync_from_vocals` при `DRY_RUN=false`
  (WhisperX `large-v3` по умолчанию — R5)
- `bsroformer-worker` single-file staging bugfix; SCNet medium как
  опциональный третий backbone (R10)
- `demucs-worker` / `bsroformer-worker` реальный subprocess path,
  TTA-обёртка (R6), адаптивные activity-маски (R-ACT), `htdemucs_ft`
  (R2), `num_overlap=4` + честный `-ar 44100` (R1+R4)
- `restoration-worker` — больше не stub: `other_residual_rebuild`
  (`direct`/`residual` режимы, R7), MWF post-filter (R8) и FFT
  per-stem ensemble (R9, `median_fft` / `min_fft` / `max_fft` /
  `avg_fft` / `min_mag`)
- `querysep-worker` — Banquet (`query-bandit`) реально подключён в
  `lyrics_sync_from_track_tandem_banquet_other.yaml` (R11) для
  long-tail (`strings/synth/brass/organ/fx`) экстракции из `other_direct`
- regression-сетка: `scripts/baseline_snapshot.py` +
  `scripts/baseline_compare.py` (R13) для отлова unintended drift,
  `scripts/sisdr_compare.py` (R14) для SI-SDR против ground truth
  (MUSDB18-HQ и т.п.)

### Всё ещё остаётся research/optional зоной
- собственные fine-tuning / LoRA / oracle-routing
- веса моделей (надо положить в `./models` руками: SCNet medium,
  Banquet checkpoints и т.д.)
- `audio-separator-worker` — это **bus-mixer-worker** (см. R12 ниже),
  PyPI-пакет `audio-separator` сюда не подключён

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build
```

После старта:
- API: `http://localhost:8000`
- UI: `http://localhost:5173`
- health: `http://localhost:8000/health`

## Рекомендуемый первый запуск

1. В UI загрузи `vocals.wav` или исходный трек.
2. Выбери preset `lyrics_sync_from_vocals`.
3. Включи `lyrics-worker`.
4. При желании загрузи `lyrics.txt` / `.lrc` / `.srt` как reference text.
5. Для реального прогона поставь `DRY_RUN=false`.
6. После job открой артефакты и сравни:
   - `lyrics_json`
   - `lyrics_lrc`
   - `lyrics_srt`
   - аудио stem

## S1/S2/S3/S5 fix-up (instrumental hybrid)

В апрельском проходе по `stem-tandem-improvement-roadmap` был
закрыт долгоживущий баг и добавлен safety-net в hybrid:

- **S1** — `bsroformer-worker.runtime` больше не использует
  `instruments[0]` для residual-subtraction. Теперь
  `_resolve_instrumental_strategy(instruments, target_instrument)`
  выбирает между `residual`, `direct` и `sum` режимами в зависимости
  от того, что модель реально предсказывает. Без этого
  `*_instrumental.wav` мог содержать всё, кроме `other`, а pipeline
  дальше работал не с инструменталом, а почти с full-mix.
- **S2** — `restoration-worker.run_tandem_instrumental_hybrid`
  считает `_vocal_contamination_db` для каждого BS-кандидата и
  обнуляет его вес в hybrid'е, если он хуже Demucs base больше чем
  на `bs_overlay_safety_margin_db` (default **3.0 dB**). Это
  предотвращает регрессию `final.bleed_db_rel_total` от -28 дБ до
  -10 дБ на агрессивных треках.
- **S3** — default-пресет в фронтенде переключён на
  `configs/presets/lyrics_sync_from_track_tandem_residual_other_center_hard.yaml`.
  Это даёт ~5 дБ выигрыша по `vocal_bleed.final_vocals.bleed_db_rel_total`
  на Nine Thou validation (-22.71 dB → **-28.11 dB**). Старый
  `lyrics_sync_from_track_tandem.yaml` остаётся доступным как
  "openness over cleanliness" вариант.
- **S5** — metadata шага hybrid'a теперь честно отражает
  применённые веса (`bs_seed_weight`, `bs_raw_weight`) с учётом
  safety-net override и пишет `bs_overlay_disabled_reason`.

Всё это покрыто unit-тестами:
`services/bsroformer-worker/tests/test_runtime_instrumental_strategy.py`,
`services/restoration-worker/tests/test_hybrid_safety_net.py`,
а также интеграционным тестом `tests/integration/test_full_pipeline_against_reference.py`,
который синтезирует контрольный mix/vocals/instrumental и
проверяет инвариант "новый default ≥ legacy default по cleanliness".

## Главная идея проекта

Не пытаться выжать из одной fixed 6-stem модели всё сразу.

Вместо этого:
1. сделать **coarse 6-stem разложение**;
2. для каждого coarse stem или residual bus выбрать **свою лучшую архитектуру**;
3. long-tail классы (`strings`, `synth_lead`, `synth_pad`, `brass_woodwinds`, `organ_reeds`, `fx`) извлекать **не fixed decoder-ом**, а query/prompt-based веткой;
4. после separation делать **restoration**, а не считать separation финалом;
5. lyrics и таймкоды держать как **отдельный слой артефактов**, а не как побочный текстовый лог.

## Что нельзя обещать нечестно

Даже с отдельным lyrics-worker нельзя гарантировать автоматическое **1-в-1 совпадение текста** для любого вокала, языка и качества источника.

Что этот проект делает честно:
- автоматическую расшифровку с таймкодами;
- word-level / segment-level синхронизацию;
- лучший путь к “почти 1-в-1” через **reference lyrics alignment**.

## Куда смотреть первым делом

- `docs/00_START_HERE_RU.md`
- `docs/01_ARCHITECTURE_RU.md`
- `docs/02_MODEL_STRATEGY_RU.md` (включая R7/R8/R9/R10/R11)
- `docs/03_BUILD_JOURNAL_RU.md`
- `docs/04_FIX_GUIDE_RU.md` (включая раздел про baseline / SI-SDR
  регрессию)
- `docs/09_LYRICS_WORKER_RU.md`
- `docs/10_V2_CHANGELOG_RU.md` (история P0–P3 sweep, R1–R14)
- `configs/presets/lyrics_sync_from_vocals.yaml`
- `configs/presets/lyrics_sync_from_track_tandem_banquet_other.yaml`
- `docker-compose.yml`

## Лицензии и модели

Проверь лицензии датасетов, чекпоинтов и репозиториев перед коммерческим использованием.
