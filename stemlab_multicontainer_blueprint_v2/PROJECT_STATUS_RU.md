# Статус проекта

## Что уже есть в архиве
- docker-compose
- gateway
- frontend
- worker wrappers
- YAML presets
- документация
- uploads API для audio/text
- artifact browser API
- media serving `/media/...`
- `lyrics-worker`
- UI-плеер для stem'ов
- UI-панель синхронизированного текста
- исправление single-file routing для `bsroformer-worker`

## Что это уже позволяет
- поднять единый UI
- смотреть preset-driven pipeline
- создавать jobs
- загружать audio и reference lyrics
- гонять orchestration между контейнерами
- получать артефакты job как API-список
- слушать результаты stem'ов прямо во фронтенде
- читать `json / txt / lrc / srt` lyrics рядом с аудио
- использовать реальный локальный lyrics pipeline при `DRY_RUN=false`

## Что уже не является проблемой из первой версии
- stale `queued` во фронте теперь можно диагностировать через polling и artifacts view
- `bsroformer-worker` больше не обязан читать всю папку inputs
- job может публиковать стабильные alias-ключи (`vocals`, `lyrics_json`, `instrumental_mix` и т.д.)

## Что добавлено P0–P3 sweep'ом (R1–R14)
- `num_overlap=2 → 4` + честный 44100 Hz во ffmpeg (R1+R4)
- `htdemucs_6s → htdemucs_ft` для drums/bass (R2)
- успокоены агрессивные `*_unmuffle` пресеты (R3)
- исправлено Mel-Band instrumental subtraction (R7b)
- промежуточный аудио-формат `pcm_s16le → pcm_f32le` (R-PCM)
- WhisperX `large-v2 → large-v3` (R5)
- TTA-обёртка (polarity-flip, L/R swap) для separation (R6)
- адаптивные пороги `_activity_mask` (R-ACT)
- прямой `other` от ML-модели (`mode: direct|residual|auto`, R7)
- MWF post-filter в `restoration-worker` (R8)
- per-stem ensemble (`median_fft`/`min_fft`/`max_fft`/`avg_fft`/`min_mag`,
  R9)
- SCNet medium как опциональный третий backbone (R10)
- Banquet (`querysep-worker`) подключён в production preset для
  long-tail (R11)
- честное переименование `audio-separator-worker → bus-mixer-worker`
  (`honest_name`, R12)
- regression-набор: `scripts/baseline_snapshot.py` /
  `scripts/baseline_compare.py` (R13)
- SI-SDR scorer: `scripts/sisdr_compare.py` (R14) — single-track или
  manifest-aggregate

## Что ещё нужно для production-like MSS
- реальные веса моделей в `./models` (SCNet medium, Banquet checkpoints,
  resurrection-модели)
- собственные fine-tuning / LoRA на MoisesDB
- waveform preview во фронтенде
- acceptance metrics на основе SI-SDR (R14) / WER (lyrics)
- очередь и лимиты по GPU/VRAM
- oracle-like section routing

## Самый готовый сценарий сейчас
1. поднять стек;
2. загрузить isolated vocals stem;
3. запустить `lyrics_sync_from_vocals`;
4. получить `lyrics_json/lrc/srt`;
5. сверять текст и аудио во фронтенде.
