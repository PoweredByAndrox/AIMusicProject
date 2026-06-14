# Эксплуатация

Документ объясняет, как запускать, останавливать, диагностировать и сопровождать стек в день-в-день. Все команды проверены против реальных контейнеров и `start.sh`/`supervisord.conf`.

## 1. Команды compose

```powershell
# Старт/стоп
docker compose up -d
docker compose stop
docker compose down                      # сохраняет volumes
docker compose down -v                   # ВНИМАНИЕ: удалит данные Postgres/Redis/Ollama/node_modules/vendor

# Только рестарт одного сервиса
docker compose restart app
docker compose restart ml-worker ml-worker-2 ml-worker-3

# Пересборка после изменений в Dockerfile
docker compose build app ml-worker
docker compose up -d --no-deps --force-recreate ml-worker
```

## 2. Что внутри контейнера `app`

`docker/app/start.sh` (entrypoint):

1. Создаёт `/var/log/app`, `/var/www/html/storage/{logs,app/audio,framework/{cache,sessions,views}}` и `/var/www/html/bootstrap/cache`.
2. Считает `sha1(composer.lock + composer.json)`, сравнивает с `vendor/.composer-lock-hash`. Если не совпало — `composer install --no-interaction --prefer-dist` (лог в `/var/log/app/composer-install.log`).
3. `fix_tree` ставит `chown www-data:www-data` + `chmod 775` (директории) / `664` (файлы) для `storage/logs`, `storage/framework`, `bootstrap/cache`.
4. Принудительно `chmod 0777 /var/www/html/storage/app/audio` — без этого YouTube-воркеры (запускающиеся под другим UID на WSL bind-mount) не смогут писать.
5. Запускает `exec /usr/bin/supervisord -c /etc/supervisor.d/supervisord.ini` (форвардит сигналы).

Под supervisor крутятся (см. `docker/app/supervisord.conf`):

| Программа | Команда | numprocs |
|-----------|---------|----------|
| `php-fpm` | `php-fpm -F` | 1 |
| `nginx` | `nginx -g "daemon off;"` | 1 |
| `laravel-worker-youtube` | `php artisan queue:work --queue=youtube --sleep=3 --tries=3 --max-time=3600` | **8** |
| `laravel-worker-analysis` | `php artisan queue:work --queue=analysis,default --sleep=3 --tries=3 --max-time=3600` | 2 |
| `laravel-worker-radio-prewarm` | `php artisan queue:work --queue=radio-prewarm ...` | 1 |
| `laravel-worker-radio-hls-prewarm` | `php artisan queue:work --queue=radio-hls-prewarm ...` | 1 |
| `laravel-scheduler` | `while true; do php artisan schedule:run --verbose --no-interaction; sleep 60; done` | 1 |

Все воркеры пишут в `storage/logs/worker-*.log` (8 youtube-процессов параллельно тянут импорты!). Scheduler запускается каждую минуту, чтобы определения из `app/Console/Kernel.php` действительно срабатывали.

> ML-задачи (анализ треков, сегменты, стемы) идут **не** через эти Laravel-очереди, а через собственный Redis-список `ml_worker:tasks` напрямую к Python-воркерам. См. [`ARCHITECTURE.md` §5](./ARCHITECTURE.md#5-очередь-ml-задач-redis-↔-python).

## 3. Логи

| Источник | Где смотреть из Windows | Команда |
|----------|--------------------------|---------|
| Laravel | `docker_data\storage\logs\laravel.log` | `Get-Content docker_data\storage\logs\laravel.log -Wait -Tail 100` |
| Laravel queue: youtube | `docker_data\storage\logs\worker-youtube.log` | (8 процессов, агрегированный лог) |
| Laravel queue: analysis,default | `docker_data\storage\logs\worker-analysis.log` | |
| Laravel queue: radio-prewarm | `docker_data\storage\logs\worker-radio-prewarm.log` | |
| Laravel queue: radio-hls-prewarm | `docker_data\storage\logs\worker-radio-hls-prewarm.log` | |
| Laravel scheduler | `docker_data\storage\logs\scheduler.log` | |
| Nginx | `docker_data\container_logs\app\nginx.stdout.log`, `nginx.stderr.log` | |
| PHP-FPM | `docker_data\container_logs\app\php-fpm.stdout.log`, `php-fpm.stderr.log` (плюс `docker compose logs app`) | |
| Composer install | `docker_data\container_logs\app\composer-install.log` | |
| ML-worker | `docker_data\storage\logs\ml_worker.log` (+ `ml_worker_2.log`, `ml_worker_3.log`) | `Get-Content docker_data\storage\logs\ml_worker.log -Wait -Tail 100` |
| ML errors | `docker_data\storage\logs\ml_worker_errors.log` (и аналоги) | |
| ML disk telemetry | `docker_data\storage\logs\ml_worker_disk.log` (и аналоги) | |
| PostgreSQL | `docker_data\container_logs\postgres\postgresql.log` | |
| Redis | `docker_data\container_logs\redis\redis.log` | |
| Ollama | `docker_data\container_logs\ollama\ollama.log` | |
| Vite + public-radio WS | `docker_data\container_logs\node\vite.log` | Один файл на оба процесса: `dev:docker` запускает `npm run radio:ws & npm run dev`, и весь stdout/stderr идёт в `vite.log`. Полезные строки: `[public-radio-ws] listening on 0.0.0.0:8080`, `VITE v… ready in …`. |
| `node` npm install | `docker_data\container_logs\node\npm-install.log` | Лог отдельной фазы `npm install` (запускается, если изменился `package-lock.json`). |
| Очередь сегментов (cron) | `docker_data\storage\logs\segments_coverage.log`, `segments_backfill.log` | |

Дополнительно — стандартные docker-логи:

```powershell
docker compose logs -f app
docker compose logs -f ml-worker ml-worker-2 ml-worker-3
docker compose logs -f node
```

## 4. Артisан-команды

`app/Console/Commands/`:

| Команда | Назначение |
|---------|------------|
| `php artisan migrate` | Применить миграции |
| `php artisan tinker` | REPL — для создания пользователей и Bearer-токенов |
| `php artisan segments:enqueue-missing --limit=20` | Поставить в очередь сегментный анализ для треков, у которых нет сегментов |
| `php artisan segments:coverage --json` | JSON-снимок покрытия эмбеддингов |
| `php artisan segments:backfill-embeddings --limit=200 --chunk=50` | Дозаполнить эмбеддинги существующих сегментов |
| `php artisan tracks:analyze-pending` (`AnalyzePendingTracks`) | Принудительно запихнуть pending-треки в очередь |
| `php artisan tracks:retry-stuck` (`RetryStuckTasks`) | Перезапустить зависшие задачи |
| `php artisan tracks:reanalyze-all` (`ReanalyzeAllTracks`) | Полный пересчёт |
| `php artisan tracks:import-orphans` (`ImportOrphanFiles`) | Импорт файлов, лежащих в `storage/app/audio` без записи в БД |
| `php artisan tracks:reparse-youtube-artists` (`ReparseYouTubeArtists`) | Переразбор `artist_credits` для YouTube-импортов |
| `php artisan tracks:restore-metadata-from-logs` (`RestoreTrackMetadataFromLogs`) | Восстановление метаданных из старых логов |
| `php artisan tracks:write-sidecars` (`WriteTrackSidecars`) | Записать sidecar'ы для существующих треков |

Точные названия можно узнать из `signature` в каждом классе (`docker compose exec app php artisan list`).

## 5. Расписание (`app/Console/Kernel.php`)

```
* * * * * (each minute)         queue-pending-tracks            (closure)
*/5 * * * *                     enqueue-missing-segments        (segments:enqueue-missing --limit=20)
0  * * * *                      segments-coverage               (segments:coverage --json → segments_coverage.log)
*/5 * * * *                     backfill-segment-embeddings     (segments:backfill-embeddings --limit=200 --chunk=50 → segments_backfill.log)
```

Все — с `withoutOverlapping()`. Запуск планировщика обеспечен программой `[program:laravel-scheduler]` в `docker/app/supervisord.conf` (бесконечный цикл `schedule:run` + `sleep 60`). Проверить, что он жив:

```powershell
docker compose exec app sh -lc "ps aux | grep -E 'schedule:run|supervisord'"
docker compose exec app tail -n 50 /var/www/html/storage/logs/scheduler.log
```

## 6. Очередь Redis

```powershell
# Длина очереди ML
docker compose exec redis redis-cli LLEN ai_music_recommendations_database_ml_worker:tasks

# Заглянуть в первые 5 задач
docker compose exec redis redis-cli LRANGE ai_music_recommendations_database_ml_worker:tasks 0 4

# Очистить очередь (БЕЗ удаления BD-записей queue_tasks)
docker compose exec redis redis-cli DEL ai_music_recommendations_database_ml_worker:tasks
```

> Префикс `ai_music_recommendations_database_` — дефолт Laravel-кэша/queue для имени БД `music_recommendations`. Если переопределите `REDIS_PREFIX` в `.env`, поменяйте и в командах.

Из Laravel удобнее работать через `MLQueueService::flushQueue()`, `queueLength()`, `removeQueuedTaskById($id)` (см. `app/Services/MLQueueService.php`).

### 6.1 Redis-ключи public-radio WS (используются `scripts/public-radio-ws-server.mjs`)

WS-сервер хранит в Redis ровно **два** типа ключей (полный реестр и поведение — в [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md)):

```powershell
# 1) STRING-счётчик слушателей конкретной станции (INCR/DECR/SET ... EX 120)
#    Источник: scripts/public-radio-ws-server.mjs:76-78, :303
docker compose exec redis redis-cli GET ai_music_recommendations_database_public_radio_listener_count:{slug}

# Сбросить счётчик (если зомби-клиенты не отвалились по staleClientTimeoutMs=25 s)
docker compose exec redis redis-cli DEL ai_music_recommendations_database_public_radio_listener_count:{slug}

# 2) LIST истории чата (LPUSH + LTRIM до chatHistoryLimit=50, EXPIRE 604800 = 7 суток)
#    Источник: scripts/public-radio-ws-server.mjs:84-86
docker compose exec redis redis-cli LRANGE ai_music_recommendations_database_public_radio_chat:{slug} 0 -1

# Очистить чат конкретной станции
docker compose exec redis redis-cli DEL ai_music_recommendations_database_public_radio_chat:{slug}
```

> Список «кто сейчас слушает» по конкретным `client_id` хранится **только в памяти WS-процесса** (`Map publicClients`, см. `scripts/public-radio-ws-server.mjs:29`), в Redis такого ключа нет. После рестарта `node`-контейнера счётчик восстановится из подключений заново, история чата переживает рестарт.
>
> Префикс берётся из `REDIS_PREFIX` (см. `docker-compose.yml:349`). Менять только синхронно с Laravel — иначе backend и WS-сервер будут видеть разные пространства ключей.

## 7. Управление пользователями и токенами

Создание, рассказано в [`INSTALLATION.md` §4](./INSTALLATION.md#4-миграции-и-первый-администратор). Ротация токена:

```php
// в php artisan tinker
$u = App\Models\User::where('email','admin@example.local')->first();
$u->apiTokens()->update(['expires_at' => now()]);   // отозвать старые
$plain = bin2hex(random_bytes(32));
$u->apiTokens()->create([
    'name' => 'cli',
    'token_hash' => hash('sha256', $plain),
]);
echo "$plain\n";
```

После любого ALTER в `api_tokens` сбросьте Redis-кеш авторизации:

```powershell
docker compose exec redis redis-cli FLUSHDB
```

Реальные ключи (см. `app/Http/Middleware/AuthenticateApiToken.php`):

- `api_token:auth:{sha256-токена}` — резолюция токена, TTL `AUTH_CACHE_TTL_SECONDS = 30 с`.
- `api_token:last_used_touch:{tokenId}` — антидребезг для обновления `last_used_at`, TTL `LAST_USED_TOUCH_INTERVAL_SECONDS = 300 с`.

## 8. Бэкап и восстановление

```powershell
# Дамп БД
docker compose exec postgres pg_dump -U postgres music_recommendations > backup.sql

# Восстановление
Get-Content backup.sql | docker compose exec -T postgres psql -U postgres music_recommendations

# Бэкап загруженных треков и sidecar'ов
Compress-Archive -Path docker_data\storage -DestinationPath storage_backup.zip
Compress-Archive -Path lab_artifacts -DestinationPath lab_artifacts_backup.zip

# Бэкап FAISS-индекса
Compress-Archive -Path docker_data\faiss_index -DestinationPath faiss_backup.zip
```

> Backup моделей (`docker_data\models\`) обычно не нужен — они скачиваются заново при сборке.

## 9. Туннель наружу

`cloudflare_tunnel/README.md`:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Cloudflare выдаст временный `https://*.trycloudflare.com`. Полезно для демо и быстрого шаринга. Не использовать для продакшна — это эфемерный URL без аутентификации со стороны Cloudflare.

## 10. Мониторинг здоровья

```powershell
# Health Laravel
curl http://localhost:8000/up
curl http://localhost:8000/api/health

# Health Ollama (нужен Bearer admin/user)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/chat/ai/health

# SSE-стрим очереди (admin)
curl -N -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8000/api/queue-status/stream

# Жив ли Postgres
docker compose exec postgres pg_isready -U postgres -d music_recommendations

# Жив ли Redis
docker compose exec redis redis-cli PING
```

## 11. Типичные сценарии разрешения инцидентов

| Инцидент | Где смотреть | Что делать |
|----------|--------------|------------|
| Очередь не двигается | `ml_worker.log`, `redis-cli LLEN ...` | Перезапустить `ml-worker*`; проверить `REDIS_PREFIX` в обоих процессах |
| Все ML-задачи падают на CUDA OOM | `ml_worker_errors.log` | Снизить `TRACK_ANALYSIS_BATCH_SIZE`; убедиться, что нет других процессов на GPU; перезапустить воркеры |
| `ml-worker` рестартится с exit 124 | watchdog | Увеличить `MAX_SEGMENT_ANALYSIS_DURATION` или разобраться, почему трек обрабатывается > 15 мин |
| Stem render висит «forever» | Redis ключ `stem_tuning_render_status:{taskId}` | Из UI вызвать `cancelRender`; либо `MLQueueService::requestStemTuningRenderCancellation` |
| Frontend не видит обновлений | `vite.log`, `npm-install.log` | Перезапустить `node`-контейнер; проверить, что `node_modules` тома живы |
| 401 на `/api/...` | `laravel.log` | Сбросить Redis-кеш токенов (`FLUSHDB`); пересоздать токен |
| Public-radio WS не подключается (`/ws/radio-listeners`, `/ws/radio-private`, `/ws/friends`) | `vite.log` (строка `[public-radio-ws] listening on 0.0.0.0:8080`); `docker compose logs node` | Перезапустить `node`-контейнер: `docker compose restart node`. Проверить, что Laravel доступен из `node` (`docker compose exec node wget -qO- http://app/api/health`). Полный протокол — [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md). |
| Слушатели «зависли» в счётчике станции | `vite.log`; ключи `redis-cli KEYS "ai_music_recommendations_database_public_radio_listener_count:*"` | Зомби-клиенты отрубаются по `staleClientTimeoutMs=25 s`. Если счётчик всё равно завышен — `redis-cli DEL ai_music_recommendations_database_public_radio_listener_count:{slug}`. |

## 12. Гигиена диска

Главные тяжелые места (быстро вырастают):

- `docker_data/storage/audio/` — оригиналы треков (бывает > 100 ГБ).
- `docker_data/demix/` — Demucs-выходы.
- `lab_artifacts/` — рендеры Stem Tuning Lab (FLAC+JSON), быстро накапливается.
- `docker_data/storage/logs/` — ротация уже включена (`RotatingFileHandler`), но 5 backup × 10 МБ для каждого воркера всё же занимают место.

Диск ML-воркера семплируется отдельно (`disk_telemetry.py` → `*_disk.log`). Если log показывает, что свободного места < 5 ГБ — пора чистить или мигрировать на больший том.

## 13. Utility-скрипты в `scripts/`

В корневой папке `scripts/` лежит набор инструментов, которые **не запускаются автоматически** ни при `docker compose up`, ни планировщиком — это руками-запускаемые утилиты. Часть исполняется на Windows-хосте (PowerShell, Node.js), часть — внутри контейнера `ml-worker` (Python, потому что использует librosa/numpy и пути `/app/...`). Источник правды для каждой утилиты — её собственный `--help`/первые ~50 строк (см. колонку «Точка входа»).

### 13.1 Сводная таблица

| Скрипт | Где исполнять | Назначение | Output по умолчанию |
|--------|---------------|------------|---------------------|
| `public-radio-ws-server.mjs` | внутри контейнера `node` (запускается из `npm run dev:docker` через `npm run radio:ws`) | WebSocket-сервер public radio (`/ws/radio-listeners`, `/ws/radio-private`, `/ws/friends`) — см. [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md) | `vite.log` (общий лог `node`-контейнера) |
| `test-stem-tuning-api.mjs` | Windows-хост (Node ≥ 18) | Запустить **один** stem-tuning render через `POST /api/stem-tuning-lab/sessions/{trackId}/render`, опросить `GET /renders/{taskId}/status` и сохранить артефакты | `.tmp/stem-tuning-api/` |
| `batch-stem-tuning-api.mjs` | Windows-хост (Node ≥ 18) | Запустить серию stem-tuning render по spec-JSON (несколько кейсов с разными mode-config) с возможностью baseline-сравнения; внутренне вызывает `test-stem-tuning-api.mjs` через `spawnSync` | `.tmp/stem-tuning-batch/` |
| `vocals-macro-fine-search.mjs` | Windows-хост (Node ≥ 18) | Грид-поиск макропресета `vocals_rewrite` с поддержкой compare-сегментов (`--compare-segments-input`); кандидаты расширяются вокруг победителя | `.tmp/vocals-macro-fine-search/` |
| `vocals-macro-batch-search.mjs` | Windows-хост (Node ≥ 18) | Аналогичный грид-поиск, но «батчевый» вариант (без авто-расширения вокруг победителя), используется как первый прогон перед fine-search | `.tmp/vocals-macro-batch-search/` |
| `compare-stem-render.py` | внутри контейнера `ml-worker` (`docker compose exec ml-worker python /app/scripts/compare-stem-render.py …`) — пути захардкожены под `APP_ROOT = /app` | Сравнить отрендеренный стем из `lab_artifacts/tuning_lab/track_{id}/render_{render_id}/stem_manifest.json` с исходным базовым стемом и опциональным baseline-рендером. Считает proxy-метрики: cosine, RMS/peak, spectral ratios и score deltas; **не** считает SI-SDR без ground truth | JSON в STDOUT; при необходимости перенаправить в файл через shell `>` |
| `probe_golden_reference_pair.py` | внутри контейнера `ml-worker` (`docker compose exec ml-worker python /app/scripts/probe_golden_reference_pair.py …`) | Зафиксировать metadata/provenance для пары `full mix` / `minus`: SHA256, codec, duration, energy-envelope alignment, residual proxy. Подходит для lossy smoke/regression fixtures, не для финального objective quality verdict | JSON в STDOUT или путь из `--output-json` |
| `build_instrumental_from_stem_manifest.py` | внутри контейнера `ml-worker` (`docker compose exec ml-worker python /app/scripts/build_instrumental_from_stem_manifest.py …`) | Собрать derived instrumental из `stem_manifest.json` как сумму всех stem-ов кроме `vocals`/`vocal`. Использует тот же safety rule, что текущий full-export helper: peak-normalize только если сумма вышла за `1.0` | FLAC из `--output-audio` и JSON в STDOUT/`--output-json` |
| `compare_instrumental_to_reference_pair.py` | внутри контейнера `ml-worker` (`docker compose exec ml-worker python /app/scripts/compare_instrumental_to_reference_pair.py …`) | Сравнить будущий `model_instrumental` с reference-minus из probe JSON. Считает alignment, cosine, RMS/duration/spectral drift и опциональные threshold checks. Threshold fail включается только флагом `--fail-on-threshold` | JSON в STDOUT или путь из `--output-json` |
| `run_nine_thou_quality_gate.py` | внутри контейнера `ml-worker`; удобный host-entrypoint: `pwsh scripts/run-nine-thou-quality-gate.ps1` | Единый repeatable Docker-only gate для локальной пары Nine Thou: обновляет probe, пересобирает derived instrumentals из manifest, прогоняет proxy threshold verdicts **без** `--fail-on-threshold` и добавляет pseudo-objective SDR/SI-SDR/SIR/SAR-like метрики из full mix/minus | `/app/lab_artifacts/golden_audio/quality_gate_repeatable/quality_gate_summary.json` и per-candidate `*_gate.json` |
| `run_ground_truth_quality_gate.py` | внутри контейнера `ml-worker`; удобный host-entrypoint: `pwsh scripts/run-ground-truth-quality-gate.ps1` | Objective gate для verified isolated stems. В `synthetic-smoke` создаёт deterministic stem fixture и проверяет, что perfect-кандидат проходит, а degraded-кандидат падает; в `evaluate` принимает `ground_truth_manifest` с `ground_truth_contract.verified=true` и считает SDR/SI-SDR/SIR/SAR-like по stem-ам | `/app/lab_artifacts/ground_truth_quality/synthetic_smoke/ground_truth_quality_summary.json` или `--output-dir` |
| `build_flstudio_ground_truth_bundle.py` | внутри контейнера `ml-worker`; для FL Studio split mixer exports | Импортирует full mix + isolated split mixer tracks в `ground_truth_manifest.json`. `source_mix.flac` — authoritative full mix input; `stems/*.flac` — individual reference targets для сравнения того, что система извлекает из full mix; `reference_sum_mix.flac` — diagnostic sum only, не эталонный микс. `ground_truth_contract.verified=true` означает verified isolated stems. `full_mix_reconstruction_contract.status=accepted` отдельно требуется только для редкого 1:1 claim; флаг `--require-full-mix-reconstruction` возвращает non-zero при drift | `source_mix.flac`, `reference_sum_mix.flac`, `stems/*.flac`, `ground_truth_manifest.json` |
| `run-virtualdj-baseline-gate.ps1` | Windows-хост + Docker `ml-worker` для разбора артефактов/метрик | Собирает host-probe VirtualDJ, опционально проверяет официальный Network Control HTTP plugin (`/query`/`/execute`), монтирует `settings.xml`/`database.xml`/History/Cache и папку VirtualDJ exports в `ml-worker`, строит `virtualdj_baseline_manifest.py`, а при `-RunQualityGate` запускает существующий Nine Thou proxy gate с VirtualDJ `Instrumental` export как `reference-minus`. Не меняет настройки VirtualDJ и не использует `--fail-on-threshold` | `lab_artifacts/virtualdj/host_probe.json`, `/app/lab_artifacts/virtualdj/baseline_manifest.json`, опционально `/app/lab_artifacts/virtualdj/nine_thou_quality_gate/quality_gate_summary.json` |
| `export_instrumental_resurrection_minus.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/export_instrumental_resurrection_minus.py …`) | Прямой Docker-only экспорт Instrumental Resurrection minus через vendored `music_source_separation_training/inference.py` и checkpoint `bs_roformer_instrumental_resurrection_unwa`. Target stem по умолчанию — `other`, потому что YAML задаёт `target_instrument: other`. Пишет durable JSONL progress log и `execution_contract` в manifest | `instrumental_resurrection_minus.flac`, `raw_ref.flac`, `stem_manifest.json`, `instrumental_resurrection_progress.jsonl` |
| `adaptive_vocal_isolation.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/adaptive_vocal_isolation.py …`) | Adaptive Docker-only controller для isolated vocals: запускает vocal profiles, строит anchor-suppressed/body-repair/router candidates, `segment_issue_repair` и `no_vocal_leakage_cleanup`, считает proxy metrics на dropout/tail preservation/backing bleed/unsupported leakage и выбирает лучший candidate. Поддерживает JSONL feedback store. Пишет отдельный selected delivery gate manifest, чтобы selected candidate/selected audio не использовались без final gate status | `adaptive_vocal_manifest.json`, `selected_delivery_gate_manifest.json`, `01_adaptive_selected_vocals.flac`, candidate subdirs |
| `build_instrumental_preference_dataset.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/build_instrumental_preference_dataset.py …`) | Training-only сборщик segment-level предпочтений для instrumental/minus route: принимает source mix, preferred `stem_manifest.json`, optional selected delivery gate, human defect labels (`residual_vocal_bleed`, `instrumental_muffling` и т.п.), строит `segments.jsonl` и feedback JSONL для будущего teacher-calibrated selector. Не является listen/result delivery и всегда пишет `delivery_allowed=false` | `instrumental_preference_dataset_manifest.json`, `segments.jsonl`, `feedback.jsonl`, `instrumental_preference_progress.jsonl` |
| `teacher_calibrated_instrumental_selector.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/teacher_calibrated_instrumental_selector.py …`) | Proxy selector для precomputed instrumental candidates: читает `instrumental_preference_dataset_manifest.json`, candidate `stem_manifest.json`, profile-level feedback priors и segment-level human A/B feedback, считает segment-level audio features, выбирает candidate по tradeoff residual-vocal suppression vs instrumental preservation и опционально рендерит diagnostic FLAC. Rendered FLAC сохраняет subtype базового candidate, когда это поддерживается. Не является accepted delivery; downstream quality gate обязателен | `teacher_calibrated_selector_manifest.json`, `segment_selection_plan.jsonl`, `teacher_calibrated_selector_progress.jsonl`, optional `teacher_calibrated_instrumental.flac` |
| `instrumental_self_judge_gate.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/instrumental_self_judge_gate.py …`) | Автоматический no-reference self-judge для instrumental/minus artifacts: считает segment-level preservation metrics, muffling proxies, optional vocal-stem leakage proxies или selector-provided vocal features, ранжирует blocker windows и пишет `proxy_pass|proxy_blocked`. Это не human A/B и не accepted delivery; при blockers маршрут должен продолжать candidate/repair iteration | `instrumental_self_judge_manifest.json`, `instrumental_self_judge_segments.jsonl`, `instrumental_self_judge_progress.jsonl` |
| `instrumental_diagnostic_comparison_gate.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/instrumental_diagnostic_comparison_gate.py …`) | No-reference diagnostic comparison для двух instrumental/minus candidates. Читает source mix, baseline candidate, tested candidate и optional preference dataset/selector manifest; считает segment-level proxy deltas по residual-vocal suppression vs instrumental preservation labels. Не заменяет reference/human gate и всегда пишет `delivery_allowed=false` | `instrumental_diagnostic_comparison_manifest.json`, `segment_comparison.jsonl`, `instrumental_diagnostic_comparison_progress.jsonl` |
| `build_diagnostic_ab_listening_pack.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/build_diagnostic_ab_listening_pack.py …`) | No-gain диагностический A/B pack для окон из `instrumental_diagnostic_comparison_gate.py`. Пишет single clips без gain, normalization, limiting или fade; single clips сохраняют source subtype, а manifest фиксирует roundtrip-аудит. Это human-review diagnostic only, не accepted delivery/listen result | `diagnostic_ab_listening_manifest.json`, `diagnostic_ab_listening_progress.jsonl`, `clips/*/A_*.flac`, `clips/*/B_*.flac`, `clips/*/AB_*.flac`, `clips/*/BA_*.flac` |
| `build_vocal_guided_minus.py` | внутри контейнера `ml-worker` (`docker compose run --rm --no-deps … python /app/tools/build_vocal_guided_minus.py …`) | Строит final minus из `raw mix - selected vocals`, опционально смешивая результат с уже существующим base minus по vocal activity envelope. Это additive слой поверх выбранного vocal stem, а не новый model inference | `vocal_guided_minus_manifest.json`, `01_selected_vocals.flac`, `03_vocal_guided_minus.flac`, listen pack |
| `define-track-segments.py` | внутри контейнера `ml-worker` (`docker compose exec ml-worker python /app/scripts/define-track-segments.py …`) | Сгенерировать «определения сегментов» (`vocal_drop`, `intro` и т.п.) из реальной волновой формы трека через librosa: окна, RMS, доминирующий тип. Пишет JSON, который потом подсовывается в `--compare-segments-input` для grid-search скриптов выше | JSON в путь, переданный в `--output` |
| `run-baseline-reseed.ps1` | Windows-хост (PowerShell 7+) | **Большой автоматический сценарий**: останавливает `app` + все `ml-worker*`, делает `redis-cli FLUSHDB`, чистит `docker_data\storage\{audio,...}`, прогоняет `migrate:fresh`, поднимает сервисы обратно, регистрирует первого admin (он автоматом получает `role=admin`), импортирует список YouTube URL батчем и ждёт полного analysis + training. Используется для baseline регрессионного теста | Все артефакты — в `lab_artifacts\baseline_reseed_<RunStamp>\` (`run.log`, `status.json`, `pre_reset_*.json`, `post_reset_*.json`, `admin_credentials.json`, ответы API) |
| `recover-youtube-metadata.ps1` | Windows-хост (PowerShell 7+ + Docker + AcoustID API key) | Восстановить настоящие `title`/`artist` для треков `WHERE title LIKE 'YouTube Import %'`. Поднимает helper-контейнер `codex-acoustid-helper` с `chromaprint`+`ffmpeg`, считает fingerprint (`fpcalc`), идёт в [acoustid.org/v2/lookup](https://api.acoustid.org/v2/lookup), сравнивает с YouTube-кандидатами через `yt-dlp ytsearch10` (внутри контейнера `app`), при достаточном покрытии токенов — апдейтит БД. Поддерживает `-DryRun` | Лог в STDOUT; апдейты в `tracks` напрямую (если не `-DryRun`) |
| `radio-browser-matrix.ps1` | Windows-хост (PowerShell + Chrome/Edge установлены) | Headless-проверка публичного радио (`/radio/<slug>`) в Chrome и Edge: запускает каждый браузер в `--headless=new --autoplay-policy=no-user-gesture-required --dump-dom`, парсит из `<body>` атрибуты `data-radio-playback-capability`, `data-radio-playback-mode`, `data-radio-waveform-transport`, `data-radio-waveform-track-id`, `data-radio-bitrate` | JSON-массив (по одной записи на каждый найденный браузер) в STDOUT |

### 13.2 Конвенции

- **`MUSIC_AI_API_TOKEN`** — общая ENV-переменная для всех `*.mjs` скриптов; альтернатива флагу `--token`. Берётся через `process.env.MUSIC_AI_API_TOKEN`. Должна быть Bearer-токеном с правами на запрос (для stem-tuning render — `role:user`/`role:admin`, см. [`API.md` §5](./API.md)).
- **`--base-url`** — все `*.mjs` по умолчанию идут на `http://localhost:8000`. Для стенда за Cloudflare Quick Tunnel — переопределить.
- **`.tmp/...` output-папки** — служебные, не коммитятся (под `.gitignore`).
- **Python-утилиты используют `APP_ROOT = /app`** — это путь внутри `ml-worker` контейнера (см. `docker-compose.yml:60`, bind `./ml_worker_improved:/app`). Запускать их с хоста напрямую нельзя — пути не совпадут; всегда `docker compose exec ml-worker python /app/scripts/<script>.py …`.
- **Windows-only PowerShell-скрипты** (`*.ps1`) опираются на Windows-пути (`D:\AIMusicProject\...`) и `docker compose exec` через CLI. На Linux/Mac не запустятся как есть.

### 13.3 Типичный pipeline для grid-search преста vocals

1. (один раз) `docker compose exec ml-worker python /app/scripts/define-track-segments.py --track-id 111 --output /app/storage/segments-111.json`.
2. `node scripts/vocals-macro-batch-search.mjs --track 111 --token %MUSIC_AI_API_TOKEN% --compare-segments-input docker_data\storage\segments-111.json --compare-segment-tag vocal_drop` — широкий первый проход.
3. `node scripts/vocals-macro-fine-search.mjs --track 111 --token %MUSIC_AI_API_TOKEN% --top-k 5 --compare-segments-input docker_data\storage\segments-111.json --compare-segment-tag vocal_drop` — уточнение вокруг победителей.
4. `node scripts/test-stem-tuning-api.mjs --track 111 --token %MUSIC_AI_API_TOKEN% --mode-config-input <winner.json> --print-request` — финальный «production» прогон.
5. (опционально) `docker compose exec ml-worker sh -lc "python /app/scripts/compare-stem-render.py --track-id 111 --render-id <new_render_id> --target-stem vocals --baseline-render-id <old_render_id> > /app/storage/compare.json"` — сравнение proxy-метрик с baseline-рендером.

### 13.4 Golden reference pair для instrumental regression

Повторяемый локальный gate запускается одной командой с хоста, но всё аудио-декодирование, сборка и метрики выполняются внутри `ml-worker`:

```powershell
pwsh scripts/run-nine-thou-quality-gate.ps1
```

Скрипт пишет `quality_gate_summary.json`, `reference_pair_probe.json` и per-candidate `*_gate.json` в `/app/lab_artifacts/golden_audio/quality_gate_repeatable/`. Он не вызывает `--fail-on-threshold`: threshold verdict хранится в JSON (`verdict.status`), а exit code отражает только runtime-ошибку самого прогона.

Objective-слой в этом gate помечен как `pseudo_objective_quality`: `instrumental_reference` строится из gain-matched `minus`, а `vocal_residual_reference` — как `full mix - gain_matched_minus`. Если full/minus alignment, cosine или gain выглядят подозрительно, summary пишет `pseudo_ground_truth_contract.warnings`; при таких warning `SDR/SI-SDR/SIR/SAR-like` значения остаются только repeatable локальным ranking/regression, а не финальным benchmark без verified isolated stem ground truth.

### 13.5 Verified ground-truth objective gate

Для проверки настоящего objective-metric слоя используется отдельный Docker-only gate. Smoke-прогон сам создаёт lossless synthetic fixture с независимыми isolated stem-ами, затем сравнивает perfect/degraded кандидатов:

```powershell
pwsh scripts/run-ground-truth-quality-gate.ps1
```

Выход: `/app/lab_artifacts/ground_truth_quality/synthetic_smoke/ground_truth_quality_summary.json`, `ground_truth/ground_truth_manifest.json`, `synthetic_perfect_objective_report.json`, `synthetic_degraded_objective_report.json`.

Для реального benchmark вместо synthetic smoke нужен manifest с явным oracle-контрактом:

```json
{
  "schema_version": 1,
  "ground_truth_contract": {
    "verified": true,
    "verification_method": "studio_isolated_stems_or_manual_lossless_stem_bundle",
    "source": "dataset_or_internal_fixture_name"
  },
  "stems": {
    "vocals": {
      "path": "/app/lab_artifacts/.../ground_truth/vocals.flac",
      "sha256": "<sha256>"
    },
    "drums": {
      "path": "/app/lab_artifacts/.../ground_truth/drums.flac",
      "sha256": "<sha256>"
    },
    "bass": {
      "path": "/app/lab_artifacts/.../ground_truth/bass.flac",
      "sha256": "<sha256>"
    },
    "other": {
      "path": "/app/lab_artifacts/.../ground_truth/other.flac",
      "sha256": "<sha256>"
    }
  }
}
```

Для ground-truth manifest `sha256` обязателен для каждого stem-а и проверяется перед декодированием audio. Это trust boundary: файл с `verified=true`, но без совпадающего hash, не допускается до расчёта метрик.

Evaluate-режим:

```bash
docker compose run --rm --no-deps --entrypoint python ml-worker \
  /app/scripts/run_ground_truth_quality_gate.py \
  --mode evaluate \
  --ground-truth-manifest /app/lab_artifacts/<fixture>/ground_truth_manifest.json \
  --candidate-manifest /app/lab_artifacts/<candidate>/candidate_manifest.json \
  --output-dir /app/lab_artifacts/<fixture>/objective_eval
```

Этот gate считает `sdr_db`, `si_sdr_db`, `sir_db`, `sar_db`, `spectral_convergence`, cosine и RMS drift. Если `ground_truth_contract.verified` не равен `true`, script падает fail-fast, чтобы model-generated stems не были случайно приняты за oracle.

Для локальной пары `full mix` / `minus` сначала фиксируется probe-артефакт:

```bash
docker compose run --rm --no-deps \
  -v "D:\AIMusicProject:/host_project:ro" \
  --entrypoint bash ml-worker -lc \
  "python /app/scripts/probe_golden_reference_pair.py \
    --reference-a '/host_project/Nine Thou (Grant Mohrman Superstars Remix) (320).mp3' \
    --reference-b '/host_project/ytmp3free.cc_styles-of-beyond-nine-thou-need-for-speed-most-wanted-2005-youtubemp3free.org.mp3' \
    --output-json /app/lab_artifacts/golden_audio/nine_thou_reference_pair_probe.json"
```

Потом будущий instrumental-кандидат сравнивается с выбранным reference-minus:

```bash
docker compose run --rm --no-deps \
  -v "D:\AIMusicProject:/host_project:ro" \
  --entrypoint bash ml-worker -lc \
  "python /app/scripts/compare_instrumental_to_reference_pair.py \
    --probe-json /app/lab_artifacts/golden_audio/nine_thou_reference_pair_probe.json \
    --candidate-instrumental /app/lab_artifacts/<candidate>/instrumental.flac \
    --reference-key reference_b \
    --output-json /app/lab_artifacts/golden_audio/<candidate>_vs_reference_b.json"
```

Если кандидат был получен через `export_hq_stems_full.py`, отдельного `instrumental.flac` там нет. Сначала собери derived instrumental из manifest:

```bash
docker compose run --rm --no-deps \
  --entrypoint bash ml-worker -lc \
  "python /app/scripts/build_instrumental_from_stem_manifest.py \
    --stem-manifest /app/lab_artifacts/<candidate>/stem_manifest.json \
    --output-audio /app/lab_artifacts/<candidate>/instrumental_from_non_vocals.flac \
    --output-json /app/lab_artifacts/<candidate>/instrumental_from_non_vocals.json"
```

Для этой команды не нужен `/host_project` mount, если `stem_manifest.json` и выходной файл находятся внутри уже смонтированного `lab_artifacts` (`/app/lab_artifacts`). Builder intentionally fail-fast при разных shape/sample-rate stem-ов, чтобы regression-метрики не строились на тихо усечённом audio.

Прямой Instrumental Resurrection candidate запускается тем же Docker-only способом. Для исходника с Windows-пути нужен read-only mount проекта в `/host_project`; результаты пишутся в уже смонтированный `lab_artifacts`:

```bash
docker compose run --rm --no-deps \
  -v "D:\AIMusicProject:/host_project:ro" \
  --entrypoint bash ml-worker -lc \
  "python /app/tools/export_instrumental_resurrection_minus.py \
    '/host_project/Nine Thou (Grant Mohrman Superstars Remix) (320).mp3' \
    /app/lab_artifacts/golden_audio/nine_thou_instrumental_resurrection_candidate \
    --device cuda:0"
```

Эти отчёты не являются SDR/SI-SDR. Они предназначены для smoke/regression tracking по lossy reference-minus: alignment, cosine, RMS drift, duration drift, spectral drift и residual proxy.

### 13.6 VirtualDJ baseline automation

VirtualDJ установлен и работает как Windows GUI-программа, поэтому управление самим приложением остаётся host-side. Аудио-декодирование, разбор export pack и regression-метрики остаются внутри Docker `ml-worker`.

Базовый probe без запуска quality gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-virtualdj-baseline-gate.ps1
```

Что делает wrapper:

1. Пишет `lab_artifacts/virtualdj/host_probe.json` с путём `virtualdj.exe`, версией, running process, TCP-соединениями процесса, `Plugins64` evidence и optional `network_control` probe.
2. Монтирует `%LOCALAPPDATA%\VirtualDJ` в `/virtualdj_data:ro` и папку export pack в `/virtualdj_export:ro`.
3. Запускает `/app/scripts/virtualdj_baseline_manifest.py` внутри `ml-worker`; manifest содержит выбранные настройки stems/CDJ export/remote, записи `database.xml`, History/Cache summary и auto-mapped audio exports.
4. Если указан `-RunQualityGate` и в export pack есть файл с ролью `instrumental_mix` (`Instrumental`, `Karaoke`, `Minus`, `No Vocals` в имени), запускает `/app/scripts/run_nine_thou_quality_gate.py` с этим файлом как `--reference-minus`.

Официальный control surface VirtualDJ для автоматизации — `Network Control` plugin: VirtualDJ docs описывают HTTP endpoints `/query` и `/execute` для VDJScript, но требуют VirtualDJ 2023+ и Pro license, а сам plugin должен быть установлен через `Config -> Extensions -> Effects -> Other` и включён в `Master Effect -> Auto-Start`. Wrapper не обходит лицензию, не устанавливает plugin скрыто и не считает отсутствие endpoint runtime-ошибкой: в `host_probe.json` будет `network_control.status=unavailable`.

Network Control probe без ручного export:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-virtualdj-baseline-gate.ps1 `
  -LaunchVirtualDj `
  -OpenTrack "D:\AIMusicProject\Nine Thou (Grant Mohrman Superstars Remix) (320).mp3" `
  -ProbeNetworkControl `
  -WaitForNetworkControlSeconds 30 `
  -CloseVirtualDjWhenDone
```

Если Network Control доступен, wrapper автоматически вызывает safe VDJScript queries: `get_version`, `get_build`, `get_license`, `get_vdj_folder`. Для проверки загрузки трека и stem readiness добавьте `-LoadOpenTrackViaNetworkControl`; тогда через `/execute` будет вызван `deck 1 load "<OpenTrack>"`, затем через `/query` будут сняты `get_filepath`, `get_title`, `get_artist`, `get_bpm`, `get_key`, `has_stems`, `has_stems "ready"`.

Не подтверждено официальными docs: отдельный headless CDJ/stems export API. Поэтому wrapper не притворяется экспортёром: Docker quality gate запускается только по реально найденным audio files в `-ExportDir`.

Пример для папки, куда руками или через VirtualDJ CDJ/Stems export положены `Acapella` и `Instrumental`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-virtualdj-baseline-gate.ps1 `
  -ExportDir "D:\AIMusicProject\lab_artifacts\virtualdj\exports\nine_thou" `
  -RunQualityGate
```

### 13.7 Adaptive minus repair controller

Production-adaptive rule: все examples с конкретными треками ниже являются regression/listening
fixtures, а не нормативными per-track rules. Production route не должен содержать hardcoded track
names, fixed timestamps, EQ или special-case branches; такие значения допустимы только как
diagnostic focus windows и должны быть отражены в manifest.

`/app/tools/adaptive_minus_repair.py` - Docker-only controller для выбора repair-профиля без
ручного перебора. Он запускает несколько vocal-safe profiles через
`/app/tools/apply_flexible_minus_repair.py`, читает их `stem_manifest.json`, ранжирует кандидатов по
proxy drift/added-layer/focus metrics и пишет `adaptive_manifest.json`.

2026-05-05 update: adaptive minus profiles начинаются с `pure_base`. Этот кандидат вызывает
`apply_flexible_minus_repair.py --pure-base` и публикует `base_minus` без ambience/texture/noise layer.
Остальные profiles (`strict_no_texture`, `soft_delta`, `vocal_safe_balanced`) являются opt-in A/B repair
кандидатами, а не clean-minus default.

Пример:

```powershell
docker compose run --rm --no-deps --entrypoint python ml-worker `
  /app/tools/adaptive_minus_repair.py `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/raw_ref.flac `
  /app/lab_artifacts/golden_audio/<track>_instrumental_resurrection_isolated/instrumental_resurrection_minus.flac `
  /app/lab_artifacts/golden_audio/<track>_adaptive_minus_repair `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_adaptive_minus_repair `
  --feedback-store /app/lab_artifacts/golden_audio/adaptive_minus_feedback.jsonl `
  --focus-window intro:0:30
```

Этот controller адаптирует selection policy через persisted feedback store. Он не train/fine-tune
нейросеть и не превращает lossy minus proxy в verified objective SDR/SI-SDR benchmark.

### 13.8 Adaptive vocal isolation controller

`/app/tools/adaptive_vocal_isolation.py` - Docker-only controller для isolated vocal stem.
Он должен оставаться обобщаемым adaptive route для разных треков: если качество на новом материале
плохое, меняйте candidates/router/scoring/diagnostics/feedback contract, а не добавляйте per-track
ветку.
Он запускает несколько vocal candidates (`hybrid_tail_preserve`, `hybrid_balanced_clean`,
`ft_tail_safe`, `bs_artifact_clean`), затем строит derived-кандидаты
`ft_tail_safe_anchor_clean` / `ft_tail_safe_anchor_strict`: они сохраняют continuity из
`ft_tail_safe`, но используют `bs_artifact_clean` как vocal anchor для подавления
uplifter/noise/synth/robot-like leakage. После этого controller строит
`ft_anchor_clean_vocal_body_repair`: он берёт clean anchor candidate как базу и дозаполняет
просевшее тело вокала consensus-донором только в vocal-core band. Controller считает proxy metrics по dropout/tail
preservation/backing bleed/unsupported anchor leakage и выбирает один
`01_adaptive_selected_vocals.flac`.

2026-05-05 update: controller поддерживает IR-residual и external vocal candidates:

- `--ir-instrumental-minus-path` добавляет `ir_residual_vocals = raw_ref - instrumental_resurrection_minus`.
- `--external-vocal-candidate NAME=PATH` добавляет precomputed MDX/MelBand/model-zoo vocal candidates без нового
  inference внутри controller.
- Если `ir_residual_vocals` доступен, controller строит `ir_residual_anchor_clean`,
  `ir_residual_supported_body_repair` и `ir_residual_supported_router`. Эти derived candidates проходят тот же
  debleed/router/quality proxy/feedback selection path.
- IR residual не является verified isolated vocal и не даёт SDR/SI-SDR ground truth; это proxy evidence для сложных
  drop-секций, где обычные candidates проглатывают окончания слов.

Segment-router candidates строятся после body-repair candidates. Они не запускают
новый model inference, а локально по time-frequency frames выбирают между clean base,
body/drop repair, tail-preserve donor и strict/anchor high-band guard. Это нужно для треков,
где один full-track winner либо слишком агрессивно глушит вокал на drop, либо тащит synth/noise
на спокойных участках. `segment_adaptive_vocal_router` остаётся conservative busy-safe baseline.
`busy_vocal_supported_router` сильнее восстанавливает vocal body на dense drop только при partial
strict/anchor speech evidence и сохраняет unsupported busy/noise damping. Busy/drop activity не
считается доказательством вокала без strict/anchor support, поэтому router damped на загруженных
non-vocal местах. Отключается флагом
`--disable-segment-router`.

Issue-aware segment selection строится после primary candidate scoring и `segment_issue_report`.
Это explicit derived candidate `issue_aware_segment_selection`, а не silent mutation выбранного
файла. Layer берёт окна только из detector output, выбирает для каждого problematic window менее
рискованный existing candidate по adjusted issue score, склеивает с crossfade, пишет
`window_decisions`/dependency fingerprints в manifest, затем пересчитывает proxy metrics/issues и
участвует в обычном ranking. Отключается только флагом
`--disable-issue-aware-segment-selection`.
Reuse для таких runs должен быть provenance-safe: candidate manifest принимается только если source
path/SHA-256, window, mode, tuning, dependency fingerprints и `stems.vocals.path` внутри текущей
candidate directory совпадают с текущим запуском.

Segment issue repair строится после issue-aware selection, если detector всё ещё видит unresolved
labels на текущем best candidate. Это explicit derived candidate `segment_issue_repair`: он не
копирует весь problematic window из другого profile, а запускает conservative STFT-router с donor,
strict и anchor support, затем подмешивает результат только в diagnostic windows. `vocal_dropout`,
`vocal_tail_loss`, `local_vocal_body_dip`, `presence_loss` и `air_detail_loss` включают controlled vocal-core recovery.
`excessive_reverb_tail_smear` помечает noisy/wide tail residue, где окно не выглядит как hard dropout,
но high-band air/reverb smear слышимо загрязняет vocal stem; это protection label для
no-vocal cleanup, поэтому он не запускает hard suppression даже вместе с thin/presence labels. `kick_punch_bleed`,
`high_noise_or_robotic_artifact`, unsupported/wide leakage labels создают `bleed_control`
decisions без donor body/tail fill; если одновременно есть core vocal-loss, route использует
`combined_recovery_and_bleed_control` с более мягким debleed.
`backing_vocal_loss_risk` остаётся diagnostic risk без isolated backing ground truth и сам по себе
не запускает donor repair. Отключается флагом `--disable-segment-issue-repair`.

No-vocal leakage cleanup строится после `segment_issue_repair`, если текущий best candidate содержит
окна, где detector видит vocal-loss/noise, но candidate consensus не подтверждает надёжный vocal evidence.
Это explicit derived candidate `no_vocal_leakage_cleanup`: он подавляет selected vocal stem только в
consensus no-vocal и соседних low-energy bridge окнах, пишет `window_decisions`/gain/crossfade в manifest
и заново участвует в ranking после пересчёта proxy metrics/issues. Layer не использует fixed timestamps,
пропускает `backing_vocal_loss_risk`; `local_vocal_body_dip` сам по себе не считается no-vocal сигналом. Отключается
флагом `--disable-no-vocal-leakage-cleanup`.
Scoring добавляет этому candidate ограниченный no-vocal cleanup bonus, чтобы intentional silence в no-vocal
окнах не проигрывал обычному dropout detector.

`/app/tools/apply_vocal_harshness_guard.py` - Docker-only derived layer для случая,
когда base vocal сохраняет тело голоса лучше альтернатив, но в detector-marked windows остаётся
режущий high-band / robotic noise. `adaptive_vocal_isolation.py` теперь также строит
`vocal_harshness_guard` как ranked candidate после `no_vocal_leakage_cleanup`. Guard берёт решения
только из `segment_issue_report`, работает по labels `high_noise_or_robotic_artifact`,
`high_band_unsupported_leakage`, `unsupported_anchor_leakage`, `excessive_reverb_tail_smear`,
сохраняет base phase и меняет только
high band. Labels потери вокала уменьшают силу guard, а unlabelled windows не трогаются.

`adaptive_vocal_manifest.json.detected_issue_action_contract` фиксирует production rule:
каждый detected issue label выбранного result получает action candidate или explicit rejection reason.
High-band harshness и reverb-tail-smear идут в `vocal_harshness_guard`; vocal recovery/bleed labels идут в
`segment_issue_repair` / `issue_aware_segment_selection`; no-vocal windows идут в
`no_vocal_leakage_cleanup` только при consensus no-vocal evidence. `backing_vocal_loss_risk`
остаётся protective rejection, потому что backing vocals являются vocal content.

```powershell
docker compose run --rm --no-deps -T --entrypoint python ml-worker `
  /app/tools/apply_vocal_harshness_guard.py `
  /app/lab_artifacts/golden_audio/<track>_adaptive_vocal_isolation/adaptive_vocal_manifest.json `
  /app/lab_artifacts/golden_audio/<track>_vocal_harshness_guard `
  --listen-dir /app/lab_artifacts/golden_audio/<track>_listen_vocal_harshness_guard `
  --base-profile issue_aware_segment_selection `
  --anchor-profile ft_tail_safe
```

2026-05-05 consistency update: `quality_consistent_ir_router` добавляет более консервативный
IR-residual-supported route для ровного качества по всему треку. Он сильнее требует strict/anchor
support, не бустит raw-busy activity и мягче подмешивает body repair, чтобы плотные drop-секции не
ломали спокойные/средние секции. Scoring также считает per-window consistency metrics через
`--segment-consistency-sec` (default `5.0`; `0` отключает слой для synthetic tests).

2026-05-05 issue diagnostics update: controller добавляет ranked `segment_issue_report`
для каждого candidate и `selected_segment_issue_report` для selected vocal stem.
Отчёт автоматически помечает windows как `vocal_dropout`, `vocal_tail_loss`,
`thin_or_overcleaned`, `leaky_full_mix_risk`, `low_band_bleed`, `kick_punch_bleed`,
`unsupported_anchor_leakage`, `high_band_unsupported_leakage`, `high_noise_or_robotic_artifact`,
`wide_stereo_non_vocal_leakage`, `backing_vocal_loss_risk`, `local_vocal_body_dip`, `presence_loss` или `air_detail_loss`.
Backing vocals считаются vocal content: `backing_vocal_loss_risk` помечает proxy-риск, где source
имеет широкую side-информацию, а selected vocal stem становится заметно уже. Это не isolated
ground truth по backing vocals, а repeatable сигнал для A/B и feedback priors.
`--segment-issue-report-max-windows` ограничивает ranked windows в manifest,
а `--segment-issue-listen-clip-count` пишет top selected-candidate issue clips в
`listen_dir/segment_issues/` как raw/vocal пары. Это repeatable diagnostic layer,
а не verified objective SDR/SI-SDR без isolated ground truth. Если эти issues подтверждаются
на слух, исправление должно идти через общий `issue_aware_segment_selection`,
`segment_issue_repair`, `no_vocal_leakage_cleanup`, router/scoring/feedback contract, а не через fixed timestamps.

Пример для файла из Downloads:

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

Controller сам готовит `/tmp/msst_zfturbo` symlink/copy runtime для BS-RoFormer из `/app/vendor`,
потому что новый `docker compose run` не сохраняет `/tmp` между запусками.
Если первые candidates уже готовы, добавьте `--reuse-existing-candidates`, чтобы расширить search
space без повторного inference уже существующих candidate directories.
Reuse теперь provenance-safe: controller не переиспользует candidate manifest, если `audio_path`,
source SHA-256, `preview_window_sec`, candidate mode, tuning, локальный `stems.vocals.path` или
dependency fingerprints не совпадает с текущим запуском.
Для derived/body/router candidates fingerprint включает SHA-256 dependency manifests и vocal files, поэтому
downstream candidate не переиспользуется после скрытой перезаписи upstream candidate на том же пути. Такие
отказы фиксируются в `adaptive_vocal_manifest.json.reuse_rejected_profiles`.
После retune не оставляйте старые калибровочные директории вида
`segment_adaptive_vocal_router_pre_support_gate_*` внутри `candidates/`: переносите их в
`archived_candidates/`, иначе broad manifest/report scans могут смешать старые tuning-параметры
с текущим `segment_adaptive_vocal_router`.
Listen pack включает `01_adaptive_selected_vocals.flac` и ranked copies
`candidate_01_<profile>.flac`, `candidate_02_<profile>.flac` для A/B.
Если `--segment-issue-listen-clip-count` больше `0`, listen pack также включает
`segment_issues/segment_issue_clips.json` и пары `issue_XX_*_raw_*.flac` /
`issue_XX_*_selected_vocals_*.flac` для самых проблемных окон selected candidate.

Для быстрой проверки проблемного intro/drop участка сохраняйте отдельный first30 listen pack в
`lab_artifacts/golden_audio/<track>_listen_adaptive_vocal_isolation_first30_<date>`: raw reference,
selected candidate, ближайшие alternatives и `first30_metrics.json`.

Feedback store пример:

```json
{"profile":"hybrid_tail_preserve","rating":"accept","score":0.8}
{"profile":"bs_artifact_clean","rating":"reject","swallowed_endings":true}
{"profile":"hybrid_balanced_clean","rating":"reject","backing_bleed":true}
{"profile":"ft_tail_safe","rating":"reject","backing_vocal_loss":true}
{"profile":"ft_tail_safe","rating":"reject","non_vocal_bleed":true,"uplifter_bleed":true,"synth_bleed":true,"robotic_drop_bleed":true}
{"profile":"ft_tail_safe_anchor_clean","rating":"accept","score":0.8}
```

Это adaptive profile selection, а не обучение весов нейросети. Без verified isolated vocals
результат оценивается proxy metrics и прослушиванием.

Если first30 pack показывает, что adaptive candidates фактически звучат как один и тот же
компромисс, используйте model-zoo rebuild:

1. Запустите реальные external backbones через existing wrappers:
   - `/app/tools/export_mdx23c_vocal_preview.py`;
   - `/app/tools/export_melband_vocal_preview.py`.
2. Соберите repeatable postprocess pack:

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

`build_vocal_model_zoo_candidates.py` не повторяет inference. Он берёт уже готовые first30 outputs,
строит MelBand debleed и multi-model support-gate candidates, пишет listen pack, `candidate_map.json`,
`model_zoo_metrics.json`, `model_zoo_delta_metrics.json` и manifest. Это правильный следующий слой, когда один backbone либо глотает
окончания слов, либо оставляет robot/synth/uplifter leakage.

Если `07_melband_supported_by_bs_mdx_first30.flac` чистый, но на drop слишком глушит вокал,
сравните append-only варианты `08_melband_drop_safe_first30.flac`,
`09_melband_drop_body_first30.flac` и `10_melband_drop_air_guard_first30.flac`. Они сохраняют
MelBand-primary routing, но делают support-gate мягче в active vocal-core зоне.

Wrapper может запустить VirtualDJ и передать ему файл трека:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-virtualdj-baseline-gate.ps1 `
  -LaunchVirtualDj `
  -Visible `
  -OpenTrack "D:\AIMusicProject\Nine Thou (Grant Mohrman Superstars Remix) (320).mp3" `
  -WaitForExportsSeconds 120
```

Важно: этот workflow не превращает VirtualDJ export в verified ground truth. VirtualDJ `Instrumental`/`Acapella` — это baseline/reference для A/B и repeatable proxy regression. Для настоящего objective benchmark по SDR/SI-SDR/SIR/SAR всё ещё нужен `ground_truth_manifest` с `ground_truth_contract.verified=true` и SHA-256 по каждому isolated stem.

## 14. Чек-лист перед релизом

- [ ] `php artisan migrate --pretend` (просмотреть планируемые ALTER)
- [ ] `php artisan migrate`
- [ ] `php artisan config:clear && php artisan route:clear`
- [ ] `npm run build` (сборка прод-фронта в `public/build/`)
- [ ] Прогон тестов:
  ```
  docker compose exec app php artisan test
  docker compose exec ml-worker pytest tests/
  ```
- [ ] Бэкап Postgres + `docker_data/storage/audio/`
- [ ] Лог-ротация проверена (доступное место > 20 ГБ)
- [ ] Активные ML-задачи `queue_tasks WHERE status='processing'` либо завершены, либо принудительно отменены
### Archived diagnostic vocal recovery checkpoint

`11_10_focus_25_30_vocal_air_detail.flac`, `12_10_focus_25_30_vocal_air_plus.flac`, and
`13_11_focus_25_30_vocal_body_plus.flac` are archived diagnostic follow-ups for one historical focused
listening window. They are not a production route, not a reusable timestamp rule, and not permission to add
per-track branches. Production vocal recovery must stay inside the adaptive candidate/router/proxy-metrics
flow described in section 13.8.
