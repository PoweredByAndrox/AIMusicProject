# Как фиксить проект

## Если separation снова “ничего не делает”

Сначала проверь не UI, а факт выполнения.

Смотри:
- `GET /api/v1/jobs/<job_id>`
- `GET /api/v1/jobs/<job_id>/artifacts`
- `data/jobs/<job_id>/job.json`
- `data/outputs/<job_id>/job_manifest.json`

Если там только команды и planned outputs, а новых `.wav/.flac/.json` нет — скорее всего у тебя снова активен dry-run.

## Если всё работает только в dry-run

Это нормально для первого старта.

Что сделать:
1. скачать веса моделей;
2. положить их в `./models`;
3. поставить `DRY_RUN=false` в `.env`;
4. пересобрать контейнеры;
5. проверить каждый worker по отдельности.

## Если BS-RoFormer снова берёт не тот input

Во второй версии worker уже staging'ит одиночный файл в `_isolated_input/`.

Если хочешь проверить:
- открой `services/bsroformer-worker/app/main.py`;
- смотри `stage_single_input()`;
- смотри `metadata["staged_input"]` в `run_log`.

Если этого staging больше нет, значит кто-то откатил фикс.

## Если UI показывает queued, а backend уже completed

Теперь смотреть надо не только список jobs, но и:
- polling `/api/v1/jobs`;
- `/api/v1/jobs/<job_id>/artifacts`;
- `run_log` внутри job.

Если backend completed, а UI старый:
- проверь `apps/frontend/src/App.tsx`;
- проверь polling interval;
- проверь, не упал ли запрос на artifacts.

## Если lyrics не совпадают с песней 1-в-1

Это не всегда баг.

Проверь по порядку:
1. был ли input именно `vocals`, а не полный микс;
2. правильно ли указан `lyrics_language`;
3. есть ли reference text;
4. не слишком ли грязный/ревербный вокал;
5. не слиплись ли слова из-за даблов, эффектов или хора.

Если нужен максимально точный текст, не полагайся только на автоматический ASR.
Дай `lyrics_reference_path` и используй alignment-проход.

## Если lyrics-worker не стартует

Смотри:
- `docker compose logs -f lyrics-worker`
- наличие `ffmpeg`
- наличие CUDA/torch при GPU режиме
- значения:
  - `LYRICS_DEVICE`
  - `LYRICS_COMPUTE_TYPE`
  - `LYRICS_WHISPERX_MODEL`
  - `LYRICS_ALIGN_MODEL`

Первый безопасный fallback:
- `LYRICS_DEVICE=cpu`
- `LYRICS_COMPUTE_TYPE=int8`

## Если API жив, а UI не может слушать артефакты

Проверь:
- смонтирован ли `./data:/app/data` у API;
- работает ли `GET /media/...`;
- правильно ли строится `media_url`;
- нет ли файлов вне `DATA_ROOT`.

## Если хочешь добавить новый stem

Пример:
- `choir`
- `backing_vocals`
- `synth_bass`
- `strings_short`
- `strings_long`

Что сделать:
1. добавить target в preset;
2. добавить service + task;
3. научить worker публиковать `named_outputs`;
4. проверить, что gateway индексирует артефакт;
5. только потом добавлять отдельную карточку/визуализацию в UI.

## Если query-based модель ещё не готова

С R11 `querysep-worker` уже умеет реальный Banquet, но если у тебя
нет весов или ты пробуешь свою query-модель:
- выделяй `harmonic_residual` (или `other_direct` из R7);
- сохраняй его как отдельный bus;
- тестируй на нём research модели отдельно;
- финал собирай через `restoration-worker.stem_ensemble` (R9), чтобы
  один кривой кандидат не убивал общий `other`.

## Если хочешь поймать «у меня раньше звучало лучше»-регрессию

После любого изменения, влияющего на DSP/MS/Wiener-параметры или
веса моделей, прогоняй:

```powershell
py scripts\baseline_snapshot.py --root data\outputs\<job_id> --out tests\baselines\after.json
py scripts\baseline_compare.py --baseline tests\baselines\before.json --candidate tests\baselines\after.json --out tests\baselines\delta.json
```

`baseline_compare.py` (R13) различает:
- `IDENTICAL` — байт-в-байт,
- `TOLERATED` — sha256 разный, но все численные метрики в пределах
  tolerance,
- `CHANGED` / `MISSING` / `NEW` — настоящая регрессия / расхождение.

Это ловит **unintended drift**, но не качество как таковое.

## Если хочешь померить настоящее качество (SI-SDR)

Используй R14:

```powershell
py scripts\sisdr_compare.py --estimates data\outputs\<job_id>\stems --reference data\reference\<track>\stems --out tests\sisdr\report.json
```

Или через manifest для агрегата по нескольким трекам:

```json
{
  "tracks": [
    { "track_id": "musdb_track_01", "estimates": "...", "reference": "..." },
    { "track_id": "musdb_track_02", "estimates": "...", "reference": "..." }
  ]
}
```

```powershell
py scripts\sisdr_compare.py --manifest tests\sisdr\manifest.json --out tests\sisdr\aggregate.json
```

SI-SDR в dB — то самое число, на которое смотрят MDX/MUSDB
бенчмарки. Растёт после фикса — значит фикс реально помог
качеству, а не просто сместил отпечаток.

## Если `audio-separator-worker` "не разделяет" / "не использует UVR-модели"

Это by design. С R12 он явно называет себя `bus-mixer-worker`
(см. `/capabilities → honest_name`). Он делает только ffmpeg-микс
шин с EQ и не подтягивает PyPI-пакет `audio-separator`.

Что делать:
- для **learnable ensemble** нескольких ML-кандидатов используй
  `restoration-worker.stem_ensemble` (R9);
- для **post-cleanup** ML-стэмов — `restoration-worker.wiener_postfilter`
  (R8);
- для **long-tail** инструментов — `querysep-worker` (Banquet, R11).
