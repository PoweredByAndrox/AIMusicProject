# StemLab Multi-Container Blueprint v2 — параллельный research-стенд

Этот документ — короткий **pointer**: подсистема `stemlab_multicontainer_blueprint_v2/` живёт в репозитории, но представляет собой **отдельный проект** со своим Docker-стеком, своим API/UI, своими docker-compose, env, моделями и собственной полной документацией.

Главный AI-music-стек (Laravel + Vue + ML-worker) и blueprint v2 — **независимы**. Они не делят между собой Docker-сеть, контейнеры, базу или Redis. Связь только на уровне общих исходных аудиофайлов и общего vendor-каталога Mel-Band Roformer (read-only bind-mount).

## 1. Назначение

Blueprint v2 — **исследовательская лаборатория** для music source separation, fine-grained routing и lyrics sync. Решает задачу, которую `ml-worker` основного проекта решает «в одну модель» (Demucs / BS-Mamba2 / MDX23C / MelBand-Roformer внутри `stem_source_build`/`stem_tuning_render`):

- **coarse 6-stem split** через Demucs/BS-RoFormer;
- **targeted stem routing** для каждого стэма своей моделью (vocals → resurrection, dereverb → bs-roformer-de-reverb, sw → BS-Rofo-SW и т.д.);
- **long-tail классы** (`strings`, `synth_lead`, `synth_pad`, `brass_woodwinds`, `organ_reeds`, `fx`) — **query-based** (Banquet/QSCNet/TUSS) поверх `other_direct`;
- **restoration** (MWF post-filter + per-stem FFT ensemble: `median_fft`/`min_fft`/`max_fft`/`avg_fft`/`min_mag`);
- **lyrics sync** (WhisperX large-v3 + alignment) как **отдельный слой артефактов**.

Подробное обоснование архитектуры — `stemlab_multicontainer_blueprint_v2/README.md` и `stemlab_multicontainer_blueprint_v2/docs/01_ARCHITECTURE_RU.md`.

## 2. Где живут собственные docs

В `stemlab_multicontainer_blueprint_v2/docs/`:

| Файл | Что внутри |
|------|------------|
| `00_START_HERE_RU.md` | первый запуск, рекомендуемый сценарий |
| `01_ARCHITECTURE_RU.md` | контейнеры, очередь, формат job |
| `02_MODEL_STRATEGY_RU.md` | стратегия по моделям (R7/R8/R9/R10/R11) |
| `03_BUILD_JOURNAL_RU.md` | что было сделано в P0–P3 sweep |
| `04_FIX_GUIDE_RU.md` | что и как чинилось, baseline / SI-SDR регрессия |
| `05_ROADMAP_RU.md` | план развития |
| `06_LIMITATIONS_RU.md` | честные ограничения |
| `07_CURRENT_LAB_MAPPING_RU.md` | соответствие моделей и пресетов |
| `08_REFERENCES_RU.md` | внешние источники |
| `09_LYRICS_WORKER_RU.md` | детали `lyrics-worker` (WhisperX + alignment) |
| `10_V2_CHANGELOG_RU.md` | история P0–P3 sweep, R1–R14 |

В корне подсистемы:

- `stemlab_multicontainer_blueprint_v2/README.md` — обзор «что есть, что работает, что research-only».
- `stemlab_multicontainer_blueprint_v2/PROJECT_STATUS_RU.md` — статус и список добавленного.
- `stemlab_multicontainer_blueprint_v2/docs/architecture.svg` — диаграмма.
- `stemlab_multicontainer_blueprint_v2/docs/TREE.txt` — снапшот дерева файлов.

> **Правило**: тонкости стенда (CLI флаги, формат preset.yaml, поведение каждого worker'а, метрики SI-SDR и WER) НЕ дублируются в основной `docs/` папке — единственный источник правды по blueprint v2 — его собственная `docs/` папка. Этот файл (`docs/STEMLAB_BLUEPRINT_V2.md`) лишь даёт контракт «как стенд соотносится с основным проектом» и куда смотреть.

## 3. Состав Docker-стека blueprint_v2

Источник: `stemlab_multicontainer_blueprint_v2/docker-compose.yml`. **11 сервисов**:

| Сервис | Build context | Назначение | Порт (host:container) |
|--------|---------------|------------|----------------------|
| `api` | `apps/api/` | FastAPI gateway + pipeline orchestrator + uploads + artifact browser | `${API_PORT:-8000}:8000` |
| `frontend` | `apps/frontend/` | React/Vite UI: загрузка треков, выбор пресета, плеер стемов, lyrics-панель | `${FRONTEND_PORT:-5173}:5173` |
| `demucs-worker` | `services/demucs-worker/` | Demucs (`htdemucs_ft`, `htdemucs_6s`) | внутренний `8000` |
| `bsroformer-worker` | `services/bsroformer-worker/` | BS-RoFormer-Infer (vocals/instrumental resurrection by unwa, опционально SCNet medium) | внутренний `8000` |
| `bsroformer-sw-worker` | `services/bsroformer-worker/` | Тот же образ, но preset на `BS-Rofo-SW-Fixed` (jarredou) | внутренний `8000` |
| `bsroformer-dereverb-worker` | `services/bsroformer-worker/` | Тот же образ, preset на `dereverb_bs_roformer_anvuew_sdr_22.5050` | внутренний `8000` |
| `audio-separator-worker` | `services/audio-separator-worker/` | **Не оборачивает** PyPI `audio-separator`. Это **bus-mixer-worker** (R12, `honest_name`) — детерминированный ffmpeg-микшер с per-bus EQ. Для реального ML-ensemble см. `restoration-worker.stem_ensemble` | внутренний `8000` |
| `querysep-worker` | `services/querysep-worker/` | Banquet (CLAP-HTSAT-unfused), checkpoint `query-bandit/ev-pre-aug.ckpt` (4dfb91d6d27c2dfd4992a15070915541) — для long-tail (strings/synth/brass/organ/fx) | внутренний `8000` |
| `restoration-worker` | `services/restoration-worker/` | `other_residual_rebuild` (`direct`/`residual`/`auto` режимы, R7), MWF post-filter (R8), FFT per-stem ensemble (R9) | внутренний `8000` |
| `lyrics-worker` | `services/lyrics-worker/` | WhisperX `large-v3` + alignment (R5). Поддерживает reference text для «почти 1-в-1» alignment | внутренний `8000` |

Все воркеры — **GPU-бэкенд** (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`). Между собой общаются через единую docker-network проекта; `api` оркестрирует jobs.

> **Конфликт портов с основным проектом**: `api` слушает `8000` (как и `app` у Laravel), `frontend` — `5173` (как и `node` у Vite). Запускать оба стека одновременно нельзя без явного `API_PORT`/`FRONTEND_PORT` override в `.env`.

## 4. Точки соприкосновения с основным проектом

| Аспект | Где | Поведение |
|--------|-----|-----------|
| Общие исходные аудиофайлы | `stemlab_multicontainer_blueprint_v2/data/inputs/*.mp3` (заранее подложены) и тестовый клип `falling_44_3_52_3.wav` | Не пересекается с `docker_data/storage/audio/` основного проекта. Стенд работает с собственным набором сэмплов. |
| Общий vendor-каталог Mel-Band Roformer | `bsroformer-worker` mount: `../ml_worker_improved/vendor/mel_band_roformer_vocal_model:/app/vendor/mel_band_roformer_vocal_model:ro` (`docker-compose.yml:107`) | Read-only. Если в основном проекте пересобирается vendor — стенд автоматически видит новый код, но **только для этого одного бэкбона**. |
| Общая Docker-сеть | НЕТ | Каждый стек в своей сети, изоляция полная. |
| Общая Postgres / Redis | НЕТ | Стенд не использует БД и Redis вообще; жобы хранятся в файлах под `data/jobs/{jobId}/job.json`. |

## 5. Формат job (для интеграции из вне)

Источник: `stemlab_multicontainer_blueprint_v2/examples/create_job.json` и `examples/create_job_curl.sh`. Подробности — в `stemlab_multicontainer_blueprint_v2/docs/01_ARCHITECTURE_RU.md`.

Каждый job получает 12-символьный hex `jobId`, и в `data/jobs/{jobId}/job.json` пишется состояние; артефакты — в `data/outputs/{jobId}/job_manifest.json` (плюс реальные `.wav`-стемы в подпапках).

> Для просмотра «по факту» того, какие job'ы реально проходили на стенде, см. `data/outputs/*/job_manifest.json` и параллельные `.tmp/job-diagnostics/{jobId}/diagnostics.{json,md}`.

## 6. Чек-лист быстрого запуска (полные подробности — в `docs/00_START_HERE_RU.md` стенда)

```bash
cd stemlab_multicontainer_blueprint_v2
cp .env.example .env
docker compose up --build

# UI: http://localhost:5173
# API: http://localhost:8000
# health: http://localhost:8000/health
```

> Перед запуском **остановите основной проект** (`docker compose down` в корне `D:\AIMusicProject`), иначе порты `8000` и `5173` будут заняты.

> Веса моделей (SCNet medium, Banquet checkpoint) НЕ запакованы в репозиторий. Положите их руками в `stemlab_multicontainer_blueprint_v2/models/` или подключите prefetch (`*_PREFETCH_ON_START=true` в env).

## 7. Что НЕ нужно делать

- Не запускайте `docker compose` стенда **из корня основного проекта** — это поднимет основной `docker-compose.yml` (Laravel + ml-worker), а не стенд.
- Не правьте `stemlab_multicontainer_blueprint_v2/docs/*.md` под общую конвенцию основной документации — там собственный язык и собственная история.
- Не интегрируйте стенд в основной runtime бесшумно. Он по дизайну отдельный — для интеграции потребуется новая `App\Services\*` обёртка, новый task type в `MLQueueService`, и согласованный путь хранения.

## 8. Кросс-ссылки

- Основной стек stem-обработки (внутри `ml-worker`, единый процесс): см. [`ML_WORKER.md`](./ML_WORKER.md) §3.1 «Зарегистрированные типы задач», task types `stem_source_build` и `stem_tuning_render`.
- Конфиг Tuning Lab (какие треки доступны): см. [`API.md`](./API.md) §2.3 «Stem Tuning Lab» и `config/stem_tuning_lab.php` в корне основного проекта.
- Скрипт-заказ stem-tuning из CLI основного проекта (отдельные от стенда): `scripts/test-stem-tuning-api.mjs`, `scripts/batch-stem-tuning-api.mjs`, `scripts/vocals-macro-batch-search.mjs`, `scripts/vocals-macro-fine-search.mjs`, `scripts/compare-stem-render.py`, `scripts/define-track-segments.py` — см. [`OPERATIONS.md`](./OPERATIONS.md) §13 «Utility-скрипты в `scripts/`».
