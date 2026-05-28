# С чего начать

## 1. Что у тебя уже есть сейчас

В этой версии архив уже умеет не только строить orchestration pipeline, но и:
- принимать uploads через API;
- индексировать артефакты job;
- отдавать media URL;
- показывать stem'ы и lyrics в одном UI;
- запускать реальный локальный lyrics pipeline, если отключить dry-run.

## 2. Что сделать в первую очередь

1. Поднять `docker compose up --build`
2. Проверить:
   - `GET /health`
   - `GET /api/v1/catalog/presets`
   - `GET /api/v1/catalog/services`
   - `GET /api/v1/jobs`
3. Открыть UI на `localhost:5173`
4. Загрузить audio
5. При необходимости загрузить `lyrics.txt`
6. Выбрать preset `lyrics_sync_from_vocals`
7. Для честного прогона поставить `DRY_RUN=false`
8. Создать job и открыть вкладку артефактов

## 3. Самый полезный первый test-case

Не начинай сразу с full fine-grained 14-stem пайплайна.

Первый честный smoke test:
- input: isolated vocals stem или raw track
- preset: `lyrics_sync_from_vocals`
- output:
  - `lyrics_json`
  - `lyrics_txt`
  - `lyrics_lrc`
  - `lyrics_srt`

Это быстрее всего показывает, что:
- uploads работают,
- job orchestration работает,
- артефакты публикуются,
- frontend умеет их читать,
- player и текст синхронизированы.

## 4. Что editable первым делом

- `configs/presets/lyrics_sync_from_vocals.yaml`
- `configs/presets/fine_grained_14stem.yaml`
- `configs/presets/lyrics_sync_from_track_tandem_residual_other.yaml`
  (R7 `mode: residual` для совместимости со старым поведением)
- `configs/presets/lyrics_sync_from_track_tandem_banquet_other.yaml`
  (R11: Banquet long-tail + R9 `stem_ensemble` финал)
- `configs/model_registry.yaml`
- `services/lyrics-worker/app/main.py` (R5: WhisperX `large-v3`)
- `services/querysep-worker/app/main.py` (R11: Banquet wired up)
- `services/restoration-worker/app/main.py` (R7/R8/R9/R-ACT)
- `services/bsroformer-worker/app/runtime.py` (R10 SCNet detect/load)
- `services/audio-separator-worker/app/main.py`
  (R12: `honest_name = bus-mixer-worker`)
- `scripts/baseline_snapshot.py`, `scripts/baseline_compare.py` (R13)
- `scripts/sisdr_compare.py` (R14)

## 5. Что менять НЕ сразу

Не надо первым делом переписывать:
- UI layout;
- docker-compose;
- gateway orchestration;
- artifact schema.

Сначала добей:
- реальные веса;
- реальные subprocess outputs;
- стабильный input/output mapping для каждого worker.

## 6. Самый важный practical note

Для lyrics у тебя теперь два режима:

### A. Automatic pass
- без reference текста;
- даёт лучший автоматический текст + таймкоды.

### B. Exact-text alignment
- с `lyrics_reference_path`;
- текст берётся из reference файла;
- модель только привязывает этот текст к времени.

Если тебе нужен результат максимально близкий к “1-в-1”, почти всегда правильнее идти через режим **B**, а не надеяться, что ASR сам идеально угадает все слова в песне.
