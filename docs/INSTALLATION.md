# Установка и первый запуск

Документ описывает реальный, а не предполагаемый, путь поднятия проекта. Все шаги сверены с `docker-compose.yml`, `docker/app/start.sh`, `docker/ml-worker/Dockerfile` и `.env.example`.

## 1. Системные требования

| Компонент | Минимум | Источник в репо |
|-----------|---------|-----------------|
| Docker Engine с поддержкой Compose v2 | 24+ | `docker-compose.yml` использует синтаксис v2 (нет ключа `version:`) |
| **NVIDIA GPU + Container Toolkit** | CUDA-совместимый | `ml-worker*` имеет `REQUIRE_CUDA=1`, `ALLOW_CPU_STRUCTURE_FALLBACK=0`, `deploy.resources.reservations.devices: [{driver: nvidia, ...}]` (`docker-compose.yml:107-118`) |
| Свободная RAM | 16 ГБ+ (3×ML-worker по 12 ГБ лимита) | `docker-compose.yml:111-114, 182-185, 253-256` |
| Свободное место | 30 ГБ+ под модели и стемы | `docker_data/`, `lab_artifacts/` |
| ОС | Linux / Windows + WSL2 / macOS (без GPU — частичная поддержка) | tmpfs, `LD_PRELOAD` явно ориентированы на Linux/WSL |
| PHP / Node / Python | НЕ нужны на хосте | Всё внутри контейнеров |

> Без GPU поднять стек «как есть» нельзя — Compose откажется выделять `nvidia` устройства. Если GPU физически нет, придётся вручную убрать блоки `deploy.resources.reservations.devices` из всех трёх ML-воркеров и переключить переменные `REQUIRE_CUDA=0`, `ALLOW_CPU_STRUCTURE_FALLBACK=1`, `ML_TORCH_DEVICE=cpu`. Производительность многих анализов резко упадёт; часть моделей (allin1 на больших треках) станет неприменима.

## 2. Подготовка окружения

```powershell
# 1. Склонировать проект (или распаковать)
cd D:\AIMusicProject

# 2. Создать .env из шаблона
Copy-Item .env.example .env

# 3. (Только для нового APP_KEY) сгенерировать ключ — после первого старта app-контейнера:
#    docker compose exec app php artisan key:generate
```

Важные переменные `.env` (источник: `.env.example`, перекрытия в `docker-compose.yml`):

| Переменная | По умолчанию | Где используется |
|------------|--------------|------------------|
| `APP_PORT` | `8000` | `docker-compose.yml:11` — внешний порт `app` |
| `APP_URL` | `http://localhost:8000` | `app:APP_URL`, `node:VITE_API_URL` |
| `APP_KEY` | (пусто) | Laravel шифрование/cookies — нужно сгенерировать! |
| `APP_ENV` | `local` | Передаётся в контейнер `app` |
| `APP_DEBUG` | `true` | То же |
| `DB_DATABASE` | `music_recommendations` | `app`, `postgres`, все `ml-worker*` |
| `DB_USERNAME` | `postgres` | То же |
| `DB_PASSWORD` | `secret` | То же |
| `DB_PORT` | `5432` | Внешний порт `postgres` |
| `REDIS_PASSWORD` | `null` (как строка) | В compose у `app` пробрасывается как `${REDIS_PASSWORD:-null}` |
| `REDIS_PREFIX` | (не в `.env.example`, но `node`-контейнер ставит `ai_music_recommendations_database_`) | Используется и `MLQueueService`, и Python-воркером |
| `OLLAMA_PORT` | `11434` | Внешний порт `ollama` |
| `LLM_PROVIDER` | `ollama` | Выбор backend: `ollama` или `vllm` |
| `VLLM_PORT` | `8001` | Внешний порт опционального vLLM backend |
| `VLLM_MODEL` | (пусто) | Явно выбранная HF text-LLM модель для vLLM OpenAI-compatible server |
| `VITE_PORT` | `5173` | Внешний порт `node` |

> `REDIS_PREFIX` критичен: Laravel и Python должны видеть одинаковый префикс, иначе `BLPOP` в Python не подхватит ничего из Laravel. Проверить, что в Laravel `config/database.php` префикс совпадает (по умолчанию выводится из `REDIS_PREFIX`/`Cache::store()`).

## 3. Сборка и старт

```powershell
# Сборка только app + ml-worker (postgres/redis/ollama/node — pull готовых образов)
docker compose build app ml-worker

# Поднять всё
docker compose up -d

# Логи в реальном времени
docker compose logs -f app
docker compose logs -f ml-worker
docker compose logs -f node
```

Что происходит при первом старте:

1. **`postgres`** инициализирует БД из `docker/postgres/init.sql` (если файл есть), иначе создаёт пустую базу `music_recommendations`.
2. **`app`**:
   - `start.sh` (см. `docker/app/start.sh`) подготавливает `storage/{logs,app/audio,framework/{cache,sessions,views}}` и `bootstrap/cache`.
   - Считает `sha1(composer.lock + composer.json)`, сравнивает с `vendor/.composer-lock-hash`. Если не совпало — запускает `composer install --no-interaction --prefer-dist` (лог: `/var/log/app/composer-install.log`) и обновляет хеш.
   - `chown -R www-data:www-data` + `chmod 775/664` для `storage/logs`, `storage/framework`, `bootstrap/cache`.
   - Принудительно `chmod 0777 /var/www/html/storage/app/audio` (чтобы YouTube-воркеры под другим UID на WSL bind-mount могли писать).
   - Запускает `supervisord` (`/etc/supervisor.d/supervisord.ini`). Под ним — `php-fpm`, `nginx`, **8 процессов** `queue:work --queue=youtube`, 2 — `queue:work --queue=analysis,default`, по 1 — `radio-prewarm`/`radio-hls-prewarm`, и `laravel-scheduler` (см. [`OPERATIONS.md` §2](./OPERATIONS.md#2-что-внутри-контейнера-app)).
3. **`ml-worker*`** запускают `python -m worker.main` (см. `docker/ml-worker/Dockerfile` и `ml_worker_improved/worker/main.py`):
   - Подключаются к Redis и Postgres (с ретраями).
   - Регистрируют обработчики задач.
   - Поднимают disk-telemetry sampler (`disk_telemetry.py`), пишут в `ml_worker[_2|_3]_disk.log`.
4. **`node`**:
   - Считает `sha1(package-lock.json + package.json)`. Если хеш не совпал с `node_modules/.install-hash` — запускает `npm install` и обновляет хеш. Иначе переиспользует кеш.
   - Запускает `npm run dev:docker`, который параллельно поднимает `npm run radio:ws` (WebSocket-сервер public-radio на порту 8080 внутри контейнера) и Vite (`vite --host 0.0.0.0` на 5173).
5. **`ollama`** просто стартует `ollama serve` на 11434.

## 4. Миграции и первый администратор

```powershell
# Применить миграции (38 файлов в database/migrations)
docker compose exec app php artisan migrate

# Создать пользователя через tinker (модель App\Models\User)
docker compose exec app php artisan tinker
```

В tinker:
```php
$u = App\Models\User::create([
    'name' => 'Admin',
    'email' => 'admin@example.local',
    'password' => 'change-me-please', // хешируется автоматически (cast 'hashed')
    'role' => 'admin',
]);
$plainToken = bin2hex(random_bytes(32));
$u->apiTokens()->create([
    'name' => 'bootstrap',
    'token_hash' => hash('sha256', $plainToken),
    'expires_at' => null,
]);
echo "Bearer token: $plainToken\n";
```

Полученный `Bearer`-токен подставляйте во все запросы к `/api/*` (см. [`API.md`](./API.md)).

> Сейчас в репозитории нет seed'а админа и нет artisan-команды для генерации токена; это известный gap. Регистрация через `POST /api/auth/register` доступна — но первый пользователь автоматически админом не становится.

## 5. Настройка Ollama (для AI Music Chat)

```powershell
# Подтянуть нужную модель внутрь тома music-ollama-data
docker compose exec ollama ollama pull llama3
# Проверить, что модель работает
docker compose exec ollama ollama list
```

Конкретное имя модели в коде не зашито — `ChatController` обращается по умолчанию к Ollama HTTP API; если переменная окружения `OLLAMA_MODEL`/аналог не выставлена, модель должна совпадать с дефолтом контроллера (см. реализацию `ChatController::aiChat`). Проверка готовности — `GET /api/chat/ai/health` (требует Bearer).

### 5.1 Опционально: vLLM вместо Ollama

vLLM запускается отдельным profile-gated сервисом и по умолчанию не занимает GPU:

```powershell
docker compose --profile llm-vllm up -d vllm
```

Минимальные env-настройки для приложения:

```dotenv
LLM_PROVIDER=vllm
VLLM_URL=http://vllm:8000/v1
VLLM_MODEL=<hf-text-llm-model>
VLLM_QUANTIZATION=awq_marlin
VLLM_KV_CACHE_DTYPE=auto
VLLM_GPU_MEMORY_UTILIZATION=0.82
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_PIPELINE_PARALLEL_SIZE=1
VLLM_MAX_NUM_SEQS=8
VLLM_MAX_NUM_BATCHED_TOKENS=4096
```

Сервис уже включает базовые оптимизации инференса: AWQ/Marlin-квантизацию для 12GB GPU, PagedAttention через vLLM runtime, prefix caching, chunked prefill, continuous batching knobs (`VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS`) и явный single-GPU parallelism (`TP=1`, `PP=1`). Speculative decoding не включен по умолчанию: для одного 12GB GPU он может конкурировать с основной моделью за VRAM/throughput и требует отдельного benchmark.

vLLM обслуживает OpenAI-compatible `/v1/chat/completions` и `/v1/models`; это backend для LLM/chat/controller inference. Demucs/MDX23C/BS-RoFormer/MelBand/SCNet остаются рабочими audio-separation моделями в `ml-worker` routes и не переносятся в vLLM.

## 6. Сборка фронта на проде

В dev-режиме `node`-контейнер сам поднимает Vite. Для прода:

```powershell
docker compose exec node npm run build
```

Сборка попадёт в `public/build/` (см. `vite.config.ts`, `manifest: true`). Laravel умеет резолвить эти ассеты через `@vite([...])`-директиву в `resources/views/app.blade.php`.

## 7. GPU-override

`docker-compose.gpu.yml` — это **не** «включить GPU» (он уже включён в базе), а дополнительные настройки, которые могут понадобиться, например, точная привязка `CUDA_VISIBLE_DEVICES`, отдельный `Dockerfile.gpu`, либо специальные настройки для Ollama. Применяется так:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 8. Cloudflare-туннель (опционально)

Описан в `cloudflare_tunnel/README.md`:

```powershell
# Установить cloudflared (winget / choco / official installer)
cloudflared tunnel --url http://localhost:8000
```

Cloudflare выдаст временный URL вида `https://*.trycloudflare.com`, направленный на ваш `app`-контейнер.

## 9. Проверка после старта

```powershell
# Health
curl http://localhost:8000/up
curl http://localhost:8000/api/health

# Логин
curl -X POST http://localhost:8000/api/auth/login `
     -H "Content-Type: application/json" `
     -d "{\"email\":\"admin@example.local\",\"password\":\"change-me-please\"}"

# Очередь ML (с админ-токеном)
curl http://localhost:8000/api/queue-status/stream `
     -H "Authorization: Bearer <token>"

# Vite dev (доступен только в development; редактирует SPA hot-reload):
curl http://localhost:5173
```

## 10. Типовые проблемы

| Симптом | Причина | Что делать |
|---------|---------|------------|
| `could not select device driver "nvidia"` | Нет NVIDIA Container Toolkit | Установить toolkit; либо удалить блоки `devices` из `docker-compose.yml` |
| ML-воркер падает с `No DNN in stream executor` | Конфликт cuDNN PyTorch vs TF | Убедиться, что `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcudnn.so.9` присутствует (в базовом compose уже выставлено) |
| Очередь не разгребается | Префиксы Redis у Laravel и Python разные | Сверить `REDIS_PREFIX` (`ai_music_recommendations_database_`) во всех сервисах |
| `400 / 401` на `/api/...` | Нет/неверный Bearer | Получить новый через `/api/auth/login` или `tinker` |
| Очень медленный `composer install` / `npm install` на WSL | Bind-mount `vendor`/`node_modules` | Уже решено named-volumes `app_vendor` и `node_modules` (`docker-compose.yml:13, 343`) |
| Demucs пишет терабайты в `C:` | tmpfs не смонтировался | Проверить, что `tmpfs: /app/storage/tmp` живёт в compose и WSL хватает RAM |

См. также [`OPERATIONS.md`](./OPERATIONS.md) для эксплуатации и мониторинга.
