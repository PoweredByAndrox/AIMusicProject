# Архитектура AI Music Project

Документ описывает реальную архитектуру системы по состоянию на текущий снимок репозитория. Все утверждения сверены с исходниками; в конце каждого блока указаны файлы-источники.

## 1. Контейнерная топология

Базовый стек поднимается через `docker-compose.yml` (8 сервисов, одна сеть `music-recommendations-network`):

| Сервис | Образ / Dockerfile | Внешний порт | Назначение |
|--------|--------------------|--------------|------------|
| `app` | `docker/app/Dockerfile` (php:8.2-fpm-alpine + nginx + supervisord) | `${APP_PORT:-8000}:80` | Laravel API + Nginx, отдаёт `/api/*` и SPA |
| `ml-worker`, `ml-worker-2`, `ml-worker-3` | `aimusicproject-ml-worker:latest` (`docker/ml-worker/Dockerfile`, база `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04`) | — | 3 параллельных GPU-воркера, читают задачи из Redis |
| `postgres` | `postgres:16-alpine` | `${DB_PORT:-5432}:5432` | Основная БД |
| `redis` | `redis:7-alpine` | (только внутри сети) | Очередь ML-воркеров и кэш |
| `ollama` | `ollama/ollama:latest` | `${OLLAMA_PORT:-11434}:11434` | Локальный LLM для AI Music Chat |
| `vllm` | `vllm/vllm-openai:latest` | `${VLLM_PORT:-8001}:8000` | Опциональный OpenAI-compatible LLM backend, profile `llm-vllm` |
| `node` | `node:20-alpine` | `${VITE_PORT:-5173}:5173` | Vite dev-сервер + WebSocket-сервер public-radio (`scripts/public-radio-ws-server.mjs`, порт 8080 внутри контейнера) |

Ключевые особенности:

- **GPU обязателен по умолчанию.** В базовом `docker-compose.yml` каждый `ml-worker*` получает `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` и `REQUIRE_CUDA=1`, `ALLOW_CPU_STRUCTURE_FALLBACK=0`. Запуск без NVIDIA Container Toolkit упадёт. Файл `docker-compose.gpu.yml` — это **дополнительный override** для специфичных GPU-настроек, а не «включалка GPU» (`docker-compose.yml:107-118, 180-189, 251-260`).
- **`tmpfs` для тяжёлых временных файлов.** ML-воркеры монтируют `/app/storage/tmp` как tmpfs (4 ГБ для `ml-worker`, по 2 ГБ для `ml-worker-2/3`), чтобы Demucs не писал многогигабайтные WAV через медленный 9P-bridge WSL (`docker-compose.yml:61-62, 136-137, 207-208`).
- **`LD_PRELOAD` для cuDNN.** Принудительно подгружается системный cuDNN ≥ 9.3 (`/usr/lib/x86_64-linux-gnu/libcudnn.so.9`), иначе Discogs-EffNet падает с «No DNN in stream executor» из-за конфликта версий с pinned cuDNN от PyTorch (`docker-compose.yml:103, 176, 247`).
- **Лимит памяти 12 ГБ на каждый ML-воркер** — защищает Docker daemon от падения, когда WSL начинает свопиться в `pagefile.sys` на C: (`docker-compose.yml:111-114`).
- **`HF_HUB_OFFLINE=1` + предзагруженные модели.** ML-образ скачивает Essentia-модели на этапе сборки (`docker/ml-worker/Dockerfile`), в рантайме загрузки из сети нет. `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME` указывают в `/app/storage/cache/*` для переиспользования между перезапусками.
- **Композ-том `app_vendor`.** Vendor PHP кэшируется в Linux-volume, а не в bind-mount, чтобы автозагрузка не тормозила на WSL (`docker-compose.yml:13`).

Источники: `docker-compose.yml`, `docker-compose.gpu.yml`, `docker/app/Dockerfile`, `docker/ml-worker/Dockerfile`, `docker/ml-worker/requirements.txt`.

## 2. Слои приложения

```
+-------------------+         HTTP (port 8000)       +-----------------------------+
|  Vue 3 SPA        |  -------------------------->   |  Nginx + PHP-FPM (Laravel)  |
|  (resources/js)   |  /api/...                      |  app container              |
|  + Vite dev:5173  |  WebSocket /radio/ws (8080)    |  - REST /api                |
+-------------------+  ----- via node container ---> |  - SPA shell /              |
                                                    |  - Public radio /radio/...   |
                                                    +-----+----------------+------+
                                                          |                |
                                                          v                v
                                                  +--------------+  +-------------+
                                                  |  PostgreSQL  |  |   Redis     |
                                                  |  (postgres)  |  |   (queue +  |
                                                  +------+-------+  |    cache)   |
                                                          |          +------+------+
                                                          |                  |
                                                          |   reads/writes   | LPUSH/RPUSH ⇆ BLPOP
                                                          v                  v
                                                  +-----------------------------+
                                                  |  ML-workers (1..3) Python   |
                                                  |  - structure_analyzer       |
                                                  |  - audio_analyzer           |
                                                  |  - dj_analyzer              |
                                                  |  - preference_model (BPR)   |
                                                  |  - segment_index (FAISS)    |
                                                  |  - stem_source_build /      |
                                                  |    stem_tuning_render       |
                                                  +-------------+---------------+
                                                                |
                                                                v
                                                  +-----------------------------+
                                                  |  Ollama / vLLM (local LLM)  |
                                                  |  для ChatController         |
                                                  +-----------------------------+
```

## 3. Роутинг и точки входа Laravel

- HTTP-фронт: `bootstrap/app.php:9-15` собирает приложение через `Application::configure()->withRouting(web: routes/web.php, api: routes/api.php, commands: routes/console.php, health: '/up')`.
- Web-маршруты (`routes/web.php`):
  - `GET /radio/{slug}` и `GET /embed/radio/{slug}` — публичное радио (`resources/views/public-radio.blade.php`).
  - `GET /{any?}` (где `where('any','.*')`) — отдаёт `resources/views/app.blade.php` (контейнер для Vue-роутера).
- API-маршруты (`routes/api.php`) разбиты на три уровня:
  1. **Публичные** (`api` middleware): `auth/register`, `auth/login`, потоковая отдача треков `tracks/{id}/stream`, `playback-options`, HLS (`hls/master.m3u8`, варианты, сегменты), waveform, public-radio.
  2. **Авторизованные** (`auth.token`): всё пользовательское — треки, fragments, лайки, рейтинги, плейлисты, рекомендации, AI DJ, AI Music Chat, pattern search, segment labeling, listening history, smart radio, stem tuning lab.
  3. **Админские** (`auth.token` + `role:admin`): `queue-status/*`, `admin/overview`, `admin/training-queue/repair`, `admin/users/{user}/preference-model/retrain`, `debug/tracks`, `debug/system`.
- Health-check: `GET /up` зарегистрирован Laravel автоматически (`bootstrap/app.php:14`), и `GET /api/health` явно объявлен в `routes/api.php:46-51`.

Полный реестр endpoint-ов — в [`API.md`](./API.md).

## 4. Аутентификация

Реализована **самописная** Bearer-token схема, без `laravel/sanctum`/`passport`:

- Таблицы `users` и `api_tokens` создаются миграцией `database/migrations/2026_03_22_000012_create_users_and_api_tokens_tables.php`. Хранится только `token_hash` (SHA256, длина 64), сам токен не сохраняется. Поля: `name`, `last_used_at`, `expires_at`, FK `user_id` с `cascadeOnDelete`.
- `app/Http/Middleware/AuthenticateApiToken.php`:
  - Принимает токен из `Authorization: Bearer <token>` или query-параметра `?token=`.
  - Хеширует SHA256 и ищет в `api_tokens`.
  - Кэширует резолюцию в Redis на `AUTH_CACHE_TTL_SECONDS` (30 с по умолчанию).
  - Обновляет `last_used_at` не чаще раза в `LAST_USED_TOUCH_INTERVAL_SECONDS` (300 с по умолчанию), чтобы не молотить БД.
  - Проверяет `expires_at` (если задано). При невалидности — `401 Unauthorized`.
- Алиасы middleware: `auth.token => AuthenticateApiToken`, `role => RequireRole` зарегистрированы в `bootstrap/app.php:17-20`.
- `App\Models\User` (`app/Models/User.php`) содержит `ROLE_ADMIN/ROLE_USER`, метод `isAdmin()`, отношения `hasMany` к `apiTokens`, `trackRatings`, `playlists`, `friendships`, `directMessages`.

## 5. Очередь ML-задач (Redis ↔ Python)

Это центральный мост между Laravel и Python-воркерами.

### Со стороны Laravel — `app/Services/MLQueueService.php`

- Константа `QUEUE_NAME = 'ml_worker:tasks'` — это **базовое имя**; Laravel-Redis сам приклеивает префикс из `REDIS_PREFIX` (по умолчанию `ai_music_recommendations_database_`).
- Методы низкого уровня:
  - `flushQueue()`, `queueLength()`, `reconcileQueue()` — обслуживание.
  - `push($payload)` (RPUSH в хвост), `pushFirst($payload)` (LPUSH в голову), `pushPriority($payload)` (RPUSH + последующий promote — см. ниже).
  - `removeQueuedTaskById($taskId)` — атомарная вырезка одной задачи по `task_id` с сохранением порядка остальных.
- Доменные хелперы:
  - `pushTrackAnalysis` — использует **`pushFirst`** (трек-анализ всегда в голову очереди).
  - `pushFragmentAnalysis`, `pushModelTraining`, `pushStemTuningRender`, `pushStemSourceBuild` — используют `pushPriority`.
  - `pushDjAnalysis` — обычный `push` (хвост).
  - `pushSegmentAnalysis` (фоновая) и `pushPrioritySegmentAnalysis` (ручной запрос пользователя) — оба `push`, но второй потом вызывает `prioritizeQueuedManualSegmentAnalysisTasks`.
  - `requestStemTuningRenderCancellation`, `clearStemTuningRenderCancellation`, `readStemTuningRenderCancellation` — управляют control-ключом `stem_tuning_render_control:{taskId}`.
  - `getQueuedTrainingTaskIds`, `reconcilePendingTrainingTasks`, `reconcilePendingTrackAnalysisTasks` — поддержка реконсиляции из админки.
- Все payload-ы — JSON с полями `type`, `task_id`, `created_at` и доменно-специфичными ключами.

#### Priority queueing (`PRIORITY_TASK_TYPES` + `pushPriority`)

`MLQueueService::PRIORITY_TASK_TYPES` (`app/Services/MLQueueService.php:26-34`) — закрытый список типов, которые **не должны простаивать за длинным сегментным backlog**:

```
['track_analysis', 'fragment_analysis', 'model_training',
 'embedding_recommendations', 'similarity_search',
 'stem_source_build', 'querysep_long_tail_build', 'stem_tuning_render']
```

Логика `pushPriority($payload)` (`MLQueueService.php:122-127`):

1. Делает обычный `RPUSH` в хвост.
2. Сразу вызывает `prioritizeQueuedTaskTypes(PRIORITY_TASK_TYPES)`, которая:
   - читает всю очередь `LRANGE 0 -1`;
   - стабильно (FIFO внутри партиций) разделяет элементы на «приоритетные» и «остальные»;
   - в одной MULTI/EXEC-транзакции делает `DEL` + `RPUSH` сначала приоритетных, потом остальных.

Возврат — `['queue_length' => N, 'promoted' => K]`, эти числа всплывают в логах хелперов и в ответах админ-эндпоинтов.

> **Важно**: «приоритет» — это **переупорядочивание FIFO**, а не отдельная очередь. Один список Redis, без partitions/sorted-set; гарантия — все приоритетные типы окажутся ближе к голове, чем `analyze_segments`/`backfill_pattern_example_embeddings`/`dj_analysis`.

Аналогично работает `prioritizeQueuedManualSegmentAnalysisTasks()` (`MLQueueService.php:653-701`): отдельно поднимает только `analyze_segments` с `priority=manual` (то есть пользовательские запросы из UI) над фоновым backlog'ом. Эта мутация вызывается из `pushPrioritySegmentAnalysis` и из `promoteAndDeduplicateQueuedSegmentAnalysisTask`.

### Со стороны Python — `ml_worker_improved/worker/queue_worker.py`

- Подключается к Redis по `REDIS_HOST`/`REDIS_PORT`/`REDIS_PASSWORD`, читает из `{REDIS_PREFIX}{QUEUE_NAME}` через `BLPOP` (`queue_worker.py:755`, timeout 5 с). Приоритезация выполняется **на стороне Laravel** через переупорядочивание списка, не на стороне воркера.
- Подключается к PostgreSQL через `psycopg2`, выполняет апдейты в `queue_tasks`, `tracks`, `fragments`, `pattern_examples`, `user_pattern_example_ratings`, статус сегментов в Redis (`setex`).
- Регистрирует обработчики через `register_handler(task_type, fn)`. Маппинг типов задач на код находится в `ml_worker_improved/worker/main.py` (импортирует `AudioAnalyzer`, `PreferenceModel`, `AIDJAnalyzer`, `StructureAnalyzer`, `SegmentIndexManager`, `SegmentPreferenceModel`, `build_track_playback_stems`, `render_stem_tuning`).
- Защитные механизмы:
  - `task_timeout` (контекст-менеджер) — мягкий таймаут.
  - `_start_hard_timeout_watchdog` — отдельный поток с `os._exit(124)` на жёсткие случаи.
  - `_requeue_timed_out_segment_task` — возвращает зависшие задачи сегментного анализа в очередь.
  - `_cleanup_process_owned_analysis_locks` — освобождает Redis-локи, привязанные к pid воркера.
  - Микро-батчинг для `track_analysis`: `_collect_track_analysis_batch` собирает до `TRACK_ANALYSIS_BATCH_SIZE` задач за `TRACK_ANALYSIS_BATCH_WAIT_MS` (значения берутся из env, в compose выставлены в `1` и `3000` соответственно).
  - GPU-cleanup: `_cleanup_gpu_memory` / `_deep_cleanup_gpu_memory` (вызывают `torch.cuda.empty_cache()`).
  - Circuit breaker в `run()` для устойчивых ошибок Redis/DB.

### Типы задач (`app/Models/QueueTask.php` и хелперы `MLQueueService`)

Подтверждённые в коде типы:

- `analysis` (`TYPE_ANALYSIS`) — полный анализ трека.
- `fragment_analysis` (`TYPE_FRAGMENT_ANALYSIS`) — эмбеддинг отдельного фрагмента.
- `model_training` / `training` (`TYPE_TRAINING`) — обучение `PreferenceModel`.
- `recommendation` (`TYPE_RECOMMENDATION`) — пересчёт рекомендаций.
- `stem_source_build` (`TYPE_STEM_SOURCE_BUILD`) — генерация исходных стемов через Demucs/HQ-сепаратор.
- `querysep_long_tail_build` (`TYPE_QUERYSEP_LONG_TAIL_BUILD`) — long-tail stem build через `querysep-worker` и `banquet_byoq` с fail-fast проверкой shared output; подробный контракт см. `docs/QUERYSEP_LONG_TAIL_BUILD.md`.
- `stem_tuning_render` (`TYPE_STEM_TUNING_RENDER`) — рендер микса стемов по preset из UI.

Дополнительно через очередь проходят сегментные операции (см. `ml_worker_improved/worker/segment_index.py`, `segment_preference_model.py`, `segment_melodic_refinement.py`).

Статусы (`QueueTask::STATUS_*`): `pending → processing → completed | failed | cancelled`. Хелперы `markAsCompleted`, `markAsFailed`, `markAsCancelled`.

## 6. Фоновые задачи Laravel (`app/Console/Kernel.php`)

| Расписание | Команда / closure | Назначение |
|------------|-------------------|------------|
| `everyMinute()` | inline closure: `Track::where('analysis_status','pending')` → `TrackAnalysisJob::dispatch` | Дотягивает до 10 «застрявших» pending-треков и кладёт их в очередь джобом |
| `everyFiveMinutes()` | `segments:enqueue-missing --limit=20` | Подсегментирует библиотеку малыми батчами, не перегружая GPU |
| `hourly()` | `segments:coverage --json` → `storage/app/logs/segments_coverage.log` | Снимок покрытия эмбеддингов сегментов |
| `everyFiveMinutes()` | `segments:backfill-embeddings --limit=200 --chunk=50` → `storage/app/logs/segments_backfill.log` | Дозаполняет эмбеддинги для существующих `pattern_examples` без пересегментации |

Все задачи помечены `withoutOverlapping()`. Реализация Artisan-команд — в `app/Console/Commands/` (`SegmentsCoverage.php`, `EnqueueMissingSegmentAnalysis.php`, `BackfillSegmentEmbeddings.php`, `AnalyzePendingTracks.php`, `ImportOrphanFiles.php`, `ReanalyzeAllTracks.php`, `ReparseYouTubeArtists.php`, `RestoreTrackMetadataFromLogs.php`, `RetryStuckTasks.php`, `WriteTrackSidecars.php`).

Scheduler и стандартные Laravel queue-воркеры запускаются под supervisord в контейнере `app`. См. `docker/app/supervisord.conf`:

| Программа supervisor | Команда | Кол-во процессов |
|----------------------|---------|------------------|
| `laravel-scheduler` | `while true; do php artisan schedule:run --verbose --no-interaction; sleep 60; done` (под `www-data`) | 1 |
| `laravel-worker-youtube` | `php artisan queue:work --queue=youtube --sleep=3 --tries=3 --max-time=3600` | 8 |
| `laravel-worker-analysis` | `php artisan queue:work --queue=analysis,default --sleep=3 --tries=3 --max-time=3600` | 2 |
| `laravel-worker-radio-prewarm` | `php artisan queue:work --queue=radio-prewarm ...` | 1 |
| `laravel-worker-radio-hls-prewarm` | `php artisan queue:work --queue=radio-hls-prewarm ...` | 1 |

Это значит, что параллельно с собственной ML-очередью (Redis-список `ml_worker:tasks`, потребляется Python-воркерами) в системе работают **обычные Laravel-очереди** (`youtube`, `analysis,default`, `radio-prewarm`, `radio-hls-prewarm`) — они тоже идут через Redis (`QUEUE_CONNECTION=redis`), но имена ключей и протокол — стандартные laravel-queue. Именно через `queue=youtube` уходят `App\Jobs\*` для импорта YouTube. Логи всех воркеров — в `storage/logs/worker-*.log`.

## 7. Стемы и Stem Tuning Lab

- Контроллер `app/Http/Controllers/StemTuningLabController.php` отдаёт UI-данные, отдаёт стримом исходные стемы (с поддержкой `Range` и `X-Accel-Redirect` для Nginx-туннеля), кладёт задачи `stem_source_build` и `stem_tuning_render` в очередь.
- Конфиг `config/stem_tuning_lab.php` объявляет «лабораторные» сессии на конкретных треках (на сейчас зарегистрирован только `track 15`) — пути `source_stem_manifest`, `render_latest_pointer`, `raw_reference_path`, дефолтный preset (`stem_drums`, `stem_bass`, `stem_other`, `stem_vocals`, `stem_guitar`, `stem_piano` с нулевыми `gainDb`/`transientDb`/`bodyDb`/`harmonicDuckDb`/`bleedDuckDb`).
- Артефакты лежат в `lab_artifacts/` (монтируется в `app` и во все `ml-worker*`).
- Статус рендера читается из Redis-ключа `stem_tuning_render_status:{taskId}`; отмена — `stem_tuning_render_control:{taskId}` (см. `MLQueueService::requestStemTuningRenderCancellation`).

Со стороны Python: `ml_worker_improved/worker/stem_source_build.py`, `stem_tuning_render.py`, `high_quality_stem_export.py`, `stem_separator_defaults.py`, плюс модели в `ml_worker_improved/vendor/` (`bs_mamba2_vocals`, `mdx23c_vocals_official`, `mel_band_roformer_vocal_model`).

## 8. Public Radio (WebSocket)

- Web-маршруты `GET /radio/{slug}` и `GET /embed/radio/{slug}` рендерят `public-radio.blade.php` (отдельная Vite entry — `resources/js/public-radio.ts`).
- WebSocket-сервер запускается в контейнере `node`: команда `npm run dev:docker` исполняет `npm run radio:ws & npm run dev`. `radio:ws` — это `node scripts/public-radio-ws-server.mjs`, слушает `RADIO_WS_PORT=8080` внутри контейнера, `RADIO_APP_ORIGIN=http://app` (общается с Laravel через сервисную сеть) (`docker-compose.yml:347-352`).
- Контроллер `App\Http\Controllers\PublicRadioController` отдаёт snapshot/историю/диагностику; `RadioAccessController` управляет токенами доступа и персональными настройками.
- Полный реестр сообщений (3 эндпоинта, формы payload, Redis-ключи, polling Laravel-API) — отдельный документ [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md).

## 9. Frontend

- Точки входа Vite (`vite.config.ts`): `resources/css/app.css`, `resources/js/app.ts`, `resources/js/public-radio.ts`. Манифест и хеш-имена включены, проксирование `/api` на Laravel.
- Vue Router (`resources/js/router/index.ts`) — основные пути: `/login`, `/register` (`guestOnly`), `/` (`library`), `/account`, `/upload`, `/track/:id`, `/track/:id/tuning-lab`, `/track/:id/segments`, `/artist/:slug`, `/recommendations`, `/playlists`, `/playlists/:id`, `/patterns`, `/patterns/:id`, `/chat`, `/friends`, `/debug` и `/admin` (последние два — `requiresAdmin`). Публичное радио (`/radio/:slug`, `/embed/radio/:slug`) идёт без app-chrome.
- `router.beforeEach` (`resources/js/router/index.ts:147-200`) проверяет `localStorage[AUTH_TOKEN_STORAGE_KEY]` и роль; при необходимости синхронизирует сессию с сервером через `useAuthStore().initialize()`.
- Состояние: Pinia с `pinia-plugin-persistedstate`. Сетевые вызовы — Axios. Аудио — Wavesurfer.js, HLS.js. WebSocket — браузерный API против `node`-контейнера.

См. [`FRONTEND.md`](./FRONTEND.md) для деталей.

## 10. Где живут данные

| Что | Где |
|-----|-----|
| Загруженные аудио / sidecar-файлы | `docker_data/storage/audio/...` (bind в `app:/var/www/html/storage/app/audio`) |
| Логи Laravel | `docker_data/storage/logs/...` (через `STORAGE_PATH=/var/www/html/storage/app`) |
| Логи Nginx/PHP-FPM | `docker_data/container_logs/app/` |
| Логи ML-воркеров | `docker_data/storage/logs/ml_worker[_2|_3].log`, `*_errors.log`, `*_disk.log` (см. `main.py:43-87`) |
| Логи PostgreSQL | `docker_data/container_logs/postgres/postgresql.log` |
| Логи Redis | `docker_data/container_logs/redis/redis.log` |
| Логи Ollama | `docker_data/container_logs/ollama/ollama.log` |
| Логи Vite/npm | `docker_data/container_logs/node/{vite.log, npm-install.log}` |
| FAISS-индекс | `docker_data/faiss_index/` (тома `ml-worker*`) |
| Веса ML-моделей | `docker_data/models/` |
| Demucs-выходы | `docker_data/demix/` |
| Lab-артефакты (стемы, рендеры) | `lab_artifacts/` (общий bind для `app` и всех `ml-worker*`) |
| PostgreSQL data | named volume `music-postgres-data` |
| Redis AOF | named volume `music-redis-data` |
| Ollama models | named volume `music-ollama-data` |
| node_modules | named volume `music-node-modules` |
| Composer vendor | named volume `music-app-vendor` |

## 11. Внешние сервисы

- **Ollama** — локальный LLM по умолчанию (слушает `0.0.0.0:11434`). Используется `ChatController::aiChat` / `aiHealth`, если `LLM_PROVIDER=ollama`. Конкретная модель задается `OLLAMA_MODEL` или `LLM_MODEL`; её нужно подтянуть командой `docker exec music-ollama ollama pull <model>`.
- **vLLM** — опциональный OpenAI-compatible LLM backend, profile-gated через `docker compose --profile llm-vllm up -d vllm`. Приложение использует его при `LLM_PROVIDER=vllm`, `VLLM_URL=http://vllm:8000/v1`, `VLLM_MODEL=<hf-model>`. Это ускоряет LLM/chat/controller inference, но не заменяет Demucs/BS-RoFormer/SCNet/audio-separator waveform inference.
- **Cloudflare Quick Tunnel** — `cloudflare_tunnel/README.md` описывает временный туннель, проксирующий `localhost:8000` наружу через `cloudflared tunnel --url http://localhost:8000`.
- **YouTube** — импорт через `yt-dlp` (установлен и в `app`-контейнере, и в `ml-worker`). Эндпоинты `POST /api/tracks/youtube` и `POST /api/tracks/youtube/batch`.

## 12. Связанные внешние артефакты в репозитории

В корне лежат каталоги `stemlab_multicontainer_blueprint_v1/` и `stemlab_multicontainer_blueprint_v2/` — это **отдельные blueprint'ы лаборатории сепарации стемов**, не подключаются к runtime текущего стека.

- `stemlab_multicontainer_blueprint_v2/` — активная research-версия с собственным `docker-compose.yml` (11 сервисов: FastAPI gateway, React/Vite UI, отдельные воркеры Demucs, BS-RoFormer (×3 пресета), audio-separator/bus-mixer, querysep, restoration, lyrics) и собственной полной документацией в `stemlab_multicontainer_blueprint_v2/docs/`. Высокоуровневый pointer для основного проекта — [`STEMLAB_BLUEPRINT_V2.md`](./STEMLAB_BLUEPRINT_V2.md).
- `stemlab_multicontainer_blueprint_v1/` — более ранний прототип, оставлен в репо для сравнения, новых правок не получает.

Никакой из blueprint'ов не запускается «по умолчанию» при `docker compose up` в корне `D:\AIMusicProject` — у каждого свой `docker-compose.yml` внутри своей папки. Они используют **те же** или близкие модели (например, оба читают `ml_worker_improved/vendor/mel_band_roformer_vocal_model/` через read-only bind), но не делят сеть, БД или Redis с основным runtime.

## Источники, прочитанные для составления документа

- `README.md`, `AGENTS.md`, `TASKS.md`
- `composer.json`, `package.json`, `.env.example`, `phpunit.xml`, `vite.config.ts`, `vitest.config.ts`, `tsconfig.json`
- `docker-compose.yml`, `docker-compose.gpu.yml`
- `docker/app/Dockerfile`, `docker/app/start.sh`
- `docker/ml-worker/Dockerfile`, `docker/ml-worker/requirements.txt`
- `bootstrap/app.php`, `routes/api.php`, `routes/web.php`, `app/Console/Kernel.php`
- `app/Http/Middleware/AuthenticateApiToken.php`
- `app/Models/Track.php`, `app/Models/Fragment.php`, `app/Models/QueueTask.php`, `app/Models/User.php`
- `app/Services/MLQueueService.php`
- `app/Http/Controllers/StemTuningLabController.php`
- `config/stem_tuning_lab.php`
- `ml_worker_improved/worker/queue_worker.py`, `ml_worker_improved/worker/main.py` (первые 200 строк)
- `structure_analyzer_snapshot.txt` (исходник `StructureAnalyzer`)
- `database/migrations/*` (полный список 37 `.php`-миграций; см. table-by-table reference в `DATABASE.md`)
- `resources/js/router/index.ts`
- `.kiro/specs/ai-music-recommendations/design.md`, `.kiro/specs/advanced-music-player/requirements.md`
- `.kiro/steering/language-russian.md`
- `cloudflare_tunnel/README.md`
- `stemlab_multicontainer_blueprint_v2/README.md`
