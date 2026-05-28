# ML Worker (Python)

Документ описывает Python-воркер: устройство процесса, типы задач, GPU-настройки, фоновые операции и зависимости. Все факты сверены с `ml_worker_improved/worker/`, `docker/ml-worker/Dockerfile`, `docker/ml-worker/requirements.txt`, `docker-compose.yml`.

## 1. Точка входа

- **Запуск:** `CMD ["python", "-m", "worker.main"]` (см. `docker/ml-worker/Dockerfile` — последняя строка).
- **Файл-инициализатор:** `ml_worker_improved/worker/main.py`.
- В compose монтируется bind: `./ml_worker_improved:/app` — рабочее дерево соответствует папке `ml_worker_improved` в репозитории.
- Параллельно поднимаются три экземпляра контейнера: `ml-worker`, `ml-worker-2`, `ml-worker-3`. Все читают одну и ту же очередь Redis, разделение нагрузки — на стороне `BLPOP` (атомарный pop с timeout 5 с, см. `queue_worker.py:755`).

## 2. Структура каталога `ml_worker_improved/worker/`

| Файл | Зачем нужен |
|------|-------------|
| `main.py` | Точка входа, регистрация хендлеров, инициализация логирования, disk-telemetry, миграции рейтингов сегментов |
| `queue_worker.py` | Реализация цикла потребителя Redis, таймауты, watchdog, GPU-cleanup, micro-batching |
| `audio_analyzer.py` | Полный анализ трека (BPM, key, mood, vocals, embeddings) |
| `structure_analyzer.py` | Структура трека через `allin1` (intro/verse/chorus/drop/...). См. `structure_analyzer_snapshot.txt` в корне репо |
| `dj_analyzer.py` | DJ-анализ (Camelot, energy, in/out cue) |
| `preference_model.py` | Тренировка/инференс BPR-модели предпочтений |
| `segment_preference_model.py`, `segment_index.py`, `segment_melodic_refinement.py`, `segment_rating_remap.py` | Сегментная подсистема: индексация в FAISS, refinement, ремап рейтингов между legacy и user-scoped таблицами |
| `pattern_analyzer.py` | Сравнение мелодических паттернов (хромограмма, pitch contour) |
| `melody_alignment.py`, `instrument_melody_profile.py`, `track_instrument_profile.py`, `source_pattern_feature_space.py` | Поддерживающие фичи для pattern search и сегментации |
| `recommendation_lab_gate.py`, `recommendation_lab_section_aware.py`, `recommendation_melodic_rerank.py` | Эвристики/реранкинги для рекомендательной лаборатории |
| `stem_separator_defaults.py`, `stem_source_build.py`, `stem_tuning_render.py`, `high_quality_stem_export.py` | Сепарация стемов, рендер пресетов из StemTuningLab |
| `faiss_manager.py` | Создание/загрузка/обновление FAISS-индекса (`/app/faiss_index/`) |
| `device_policy.py` | Резолв `torch.device('cuda'|'cpu')`, диагностика NVCC/cuDNN |
| `disk_telemetry.py` | Семплер использования диска (отдельный лог-файл) |
| `download_models.py` | Утилита для предзагрузки моделей (вызывается на сборке образа) |

## 3. Жизненный цикл задачи

1. Laravel вызывает `App\Services\MLQueueService::push*` → `RPUSH` (или `LPUSH` для `pushFirst`/`pushTrackAnalysis`) в Redis-список `${REDIS_PREFIX}ml_worker:tasks` (имя константы — `QueueWorker.QUEUE_NAME`, см. `queue_worker.py:71-72`). Для приоритетных типов задач (`pushPriority`) дополнительно вызывается переупорядочивание очереди, см. §3.2 ниже.
2. Один из ML-воркеров делает `BLPOP` (`queue_worker.py:755`, timeout = 5 с — всегда, независимо от типа задачи) и десериализует JSON. Сам воркер не различает «обычные» и «приоритетные» задачи: они уже стоят в нужном порядке слева направо благодаря переупорядочиванию в Laravel. Это значит:
   - `pushFirst`/`pushTrackAnalysis` (LPUSH в левый конец) → задача попадёт на обработку **первой**.
   - `push` (RPUSH в правый конец) → задача в хвосте, обработается после всего, что левее.
   - `pushPriority` (RPUSH + переупорядочивание так, что приоритетные task-types сдвигаются в левую часть) → попадёт на обработку **раньше** оставшихся непривилегированных типов, но **после** всего, что было LPUSH-нуто или поднято этим же `pushPriority` ранее.
3. `QueueWorker.process_task`:
   - Обновляет `queue_tasks.status = 'processing'`.
   - Запускает соответствующий handler из реестра (`register_handler` / `register_batch_handler`).
   - При успехе — `markAsCompleted`, при таймауте/ошибке — `markAsFailed` с `error_message`.

### 3.1 Зарегистрированные типы задач (полный список из `main.py:4164-4212`)

| `task_type` | Handler | Что делает | Куда пишет результат |
|-------------|---------|------------|----------------------|
| `track_analysis` | `handle_track_analysis` + `handle_track_analysis_batch` (batch-handler) | BPM, key, mood, vocals, mood/genre/embeddings + структурный анализ (allin1) | Поля `tracks` (`bpm`, `key`, `mood`, `genre`, `analysis_status='completed'`, `embedding_path`, `loudness_*`); записи в `track_segments`, `track_structure_sections` |
| `fragment_analysis` | `handle_fragment_analysis` | Эмбеддинг конкретного фрагмента + индексация в FAISS | `fragments.embedding_path`, `analysis_status`; FAISS-индекс в `/app/faiss_index/` |
| `model_training` | `handle_model_training` (внутри — `_handle_model_training_inner` или `_inner_v2`) | BPR-тренировка персональной preference-модели | `users.preference_model_status`, история в `preference_model_training_runs` |
| `dj_analysis` | `handle_dj_analysis` | Camelot-нотация, energy-кривая, in/out cue | Поля `tracks` (`dj_camelot`, `dj_energy_*`, `dj_cue_*`) |
| `segment_search` | `handle_segment_search` | Поиск похожих сегментов по эмбеддингу запроса | Результат возвращается обратно в Laravel (queue_tasks.result) |
| `embedding_recommendations` | `handle_embedding_recommendations` | Re-ranked рекомендации на основе пользовательских лайков и эмбеддингов | `queue_tasks.result` |
| `stem_source_build` | `handle_stem_source_build` | Сепарация стемов через Demucs / BS-Mamba2 / MDX23C / MelBand-Roformer для последующего Tuning Lab | Файлы в `lab_artifacts/.../sources/`, манифест JSON |
| `querysep_long_tail_build` | `handle_querysep_long_tail_build` | Long-tail stem build через внешний `querysep-worker` и `banquet_byoq` без proxy/fallback | Файлы в shared `lab_artifacts/.../long_tail...`, `querysep_long_tail_manifest.json` |
| `stem_tuning_render` | `handle_stem_tuning_render` (исполняется в **дочернем процессе** для изоляции памяти) | Микширует выбранные стемы по пресету Tuning Lab | Файлы в `lab_artifacts/.../renders/`, статус и прогресс — в Redis `${REDIS_PREFIX}stem_tuning_render_status:{taskId}` (см. `queue_worker.py:259`) |
| `train_segment_model` | `handle_train_segment_model` | Тренировка пользовательской сегментной preference-модели | `users.segment_preference_model_*` поля |
| `analyze_segments` | `handle_analyze_segments` | Сегментация трека (на основе structure-выводов) + эмбеддинги по каждому сегменту | `track_segments`, статус — Redis `${REDIS_PREFIX}segment_analysis_status:{trackId}` (`queue_worker.py:230`) |
| `backfill_pattern_example_embeddings` | `handle_backfill_pattern_example_embeddings` | Дозаполнение `embedding_path` у уже существующих `pattern_examples` (миграция данных) | `pattern_examples.embedding_path` |

> Stem-задачи дополнительно используют control-ключ `${REDIS_PREFIX}stem_tuning_render_control:{taskId}` (`queue_worker.py:297, 312`) — через него Laravel шлёт сигналы отмены (см. `MLQueueService::requestStemTuningRenderCancellation`).

> Сегментный анализ держит heartbeat-локи `${REDIS_PREFIX}ml_worker:segment_analysis_lock(_heartbeat)` и `${REDIS_PREFIX}ml_worker:structure_gpu_lock` (`queue_worker.py:381-383`) — это защищает GPU от параллельного запуска тяжёлых allin1/segment-моделей на нескольких воркерах.

### 3.2 Приоритезация со стороны Laravel

Воркер сам из очереди выгребает по `BLPOP` строго слева направо — никакой логики приоритетов в Python нет. Приоритезация полностью **на стороне Laravel** (`app/Services/MLQueueService.php`), которая решает, что окажется ближе к левому концу:

| Метод Laravel | Что делает с очередью | Какие task-types из этого выигрывают |
|---------------|----------------------|--------------------------------------|
| `push($payload)` | RPUSH в хвост, без переупорядочивания | `dj_analysis`, `analyze_segments` (фоновые), `backfill_pattern_example_embeddings` |
| `pushFirst($payload)` | LPUSH прямо в голову | `track_analysis` (через `pushTrackAnalysis`) |
| `pushPriority($payload)` | RPUSH + `prioritizeQueuedTaskTypes(PRIORITY_TASK_TYPES)`: одна транзакция MULTI/EXEC, которая переставляет приоритетные task-types ближе к голове, сохраняя FIFO внутри обеих партиций | `fragment_analysis`, `model_training`, `embedding_recommendations`, `similarity_search`, `stem_source_build`, `querysep_long_tail_build`, `stem_tuning_render` (плюс сам `track_analysis`, если ему так пришли) |
| `pushPrioritySegmentAnalysis($trackId)` | RPUSH `analyze_segments` с `priority=manual` + `prioritizeQueuedManualSegmentAnalysisTasks()` | `analyze_segments` (только записи с `priority=manual`, то есть ручной запрос пользователя из UI) |

`PRIORITY_TASK_TYPES` определена константой и зафиксирована в `MLQueueService.php:26-34`:

```
['track_analysis', 'fragment_analysis', 'model_training',
 'embedding_recommendations', 'similarity_search',
 'stem_source_build', 'querysep_long_tail_build', 'stem_tuning_render']
```

Что это **не**:

- Не отдельная очередь Redis. Очередь — один список `${REDIS_PREFIX}ml_worker:tasks`.
- Не sorted set / не priority queue в строгом смысле. Это **переупорядочивание FIFO** при каждом `pushPriority`.
- Не гарантирует head-of-line при наличии конкурирующего `pushFirst` — `pushFirst` всегда ближе к голове, чем последующий `pushPriority`.

Возврат `pushPriority` — `['queue_length' => N, 'promoted' => K]`, где `N` — длина очереди после операции, `K` — сколько элементов оказалось в приоритетной партиции. Эти числа доступны вызывающему коду (`StemTuningLabController` логирует их при постановке `stem_tuning_render`).

Для воркера эта приоритезация прозрачна — он всегда делает `BLPOP` и видит только то, что Laravel заранее переставил ближе к левому концу. Прямые `LPOP` без блокировки используются только в служебной логике re-queue таймаутнутых сегментных задач (`queue_worker.py:582,588`), не в основном цикле.

## 4. Защитные механизмы

(подтверждение: `queue_worker.py`)

- **Soft timeout** через `task_timeout` context manager — поднимает `TaskCancelledError`, чтобы хендлер вышел чисто.
- **Hard timeout watchdog** (`_start_hard_timeout_watchdog`) — отдельный поток. Если задача висит дольше `MAX_SEGMENT_ANALYSIS_DURATION` (по умолчанию 900 с в compose), вызывает `os._exit(124)` — Docker перезапустит контейнер с `restart: unless-stopped`.
- **Re-queue застрявших сегментных задач** (`_requeue_timed_out_segment_task`) — возвращает payload в очередь.
- **Освобождение Redis-локов** (`_cleanup_process_owned_analysis_locks`) — снимает блокировки, привязанные к pid процесса.
- **GPU-cleanup**:
  - `_cleanup_gpu_memory()` — `torch.cuda.empty_cache()` после задачи.
  - `_deep_cleanup_gpu_memory()` — также `gc.collect()` + сброс CUDA-кеша TF.
  - **Idle GPU cleanup** в `run()` — если воркер простаивает, периодически чистит видеопамять.
- **Circuit breaker** — серия подряд идущих ошибок Redis/DB временно тормозит цикл с увеличивающимся sleep, чтобы не залить логи.
- **Micro-batching для `track_analysis`**:
  - `_collect_track_analysis_batch` собирает до `TRACK_ANALYSIS_BATCH_SIZE` задач.
  - Ждёт максимум `TRACK_ANALYSIS_BATCH_WAIT_MS`.
  - В compose сейчас выставлено `BATCH_SIZE=1`, `WAIT_MS=3000` (`docker-compose.yml:81-82`) — то есть фактически батча нет, конфигурация сохраняет возможность включить.

## 5. Переменные окружения

| Переменная | Compose-default | Зачем |
|------------|-----------------|-------|
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` | `redis`, `6379`, `(пусто)` | Подключение к Redis |
| `REDIS_PREFIX` | `ai_music_recommendations_database_` | **Должно совпадать** с Laravel. В `docker-compose.yml` для `ml-worker*` явно **не выставляется** — берётся дефолт `os.getenv('REDIS_PREFIX', 'ai_music_recommendations_database_')` из `queue_worker.py:71`, который намеренно совпадает с Laravel-префиксом по умолчанию (`config/database.php → redis.options.prefix`). Если переопределяете в Laravel — продублируйте и в окружении ML-сервисов. |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | `postgres`, `5432`, `music_recommendations`, ... | Подключение к Postgres через `psycopg2` |
| `STORAGE_PATH` | `/app/storage` | Корень для всех путей файлов треков и логов |
| `FAISS_INDEX_PATH` | `/app/faiss_index` | Где живёт FAISS-индекс |
| `MODELS_PATH` | `/app/models` | Веса моделей |
| `ML_WORKER_LOG_NAME` | `ml_worker.log` (для `_2`/`_3` — `ml_worker_2.log`/`ml_worker_3.log`) | Имя лог-файла |
| `ML_WORKER_ERROR_LOG_NAME` | `ml_worker_errors.log` | Только ERROR+ |
| `ML_WORKER_DISK_LOG_NAME` | `ml_worker_disk.log` | Disk-telemetry |
| `MAX_SEGMENT_ANALYSIS_DURATION` | `900` (сек) | Hard-timeout сегментного анализа |
| `ALLOW_CPU_STRUCTURE_FALLBACK` | `0` | `1` — разрешить fall-back `structure_analyzer` на CPU; иначе сразу fail |
| `ENABLE_STRUCTURE_GUIDED_SEGMENTATION` | `1` | Использовать allin1-структуру для подсказки сегментов |
| `TRACK_ANALYSIS_BATCH_SIZE` | `1` | Micro-batching size |
| `TRACK_ANALYSIS_BATCH_WAIT_MS` | `3000` | Macro-pause перед обработкой батча |
| `ALLIN1_TMP_ROOT` | `/app/storage/tmp/allin1` (для `_2`/`_3` — `allin1-2`/`allin1-3`) | tmpfs для allin1 |
| `XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME` | `/app/storage/cache/...` | Переиспользуемый кеш |
| `HF_HUB_OFFLINE` | `1` | Запрет интернета для HuggingFace в рантайме |
| `REQUIRE_CUDA` | `1` | Падать, если нет GPU |
| `ML_TORCH_DEVICE` | `cuda` | Целевое устройство |
| `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES` | `all`, `compute,utility` | Стандартные NVIDIA Container Toolkit env |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS` | `1` | Линейная алгебра — однопоточная (чтобы не топить GPU) |
| `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` | `1` | TensorFlow тоже однопоточный |
| `LD_PRELOAD` | `/usr/lib/x86_64-linux-gnu/libcudnn.so.9` | **Обязательно** — иначе TF (Discogs-EffNet) падает с «No DNN in stream executor» из-за конфликта с PyTorch-cuDNN |
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Не отжирать всю VRAM сразу |
| `DISK_TELEMETRY_ENABLED` | `1` | Включить семплер диска |
| `DISK_TELEMETRY_SAMPLE_SECONDS` | `5` | Период семплирования |
| `PYTHONUNBUFFERED` | `1` | Чтобы `docker logs` показывал stdout сразу |

## 6. Зависимости (`docker/ml-worker/requirements.txt`)

Полный файл — в `docker/ml-worker/requirements.txt`. Ключевые пины и причины:

| Пакет | Версия | Почему именно она |
|-------|--------|-------------------|
| `torch` | `2.5.1` | Совместимость с CUDA 12.6 + cuDNN, выбранным базовым образом |
| `torchaudio` | `<2.9` | Старший `torchaudio` тянет `torchcodec`, конфликтующий с Demucs |
| `tensorflow` | `2.21.0` | Для Discogs-EffNet (Essentia модель) |
| `essentia` | (последняя стабильная) | BPM/key/mood/loudness/embeddings |
| `madmom` | git-snapshot | На PyPI давно сломан, ставится напрямую из git |
| `natten` | пиннинг | Для allin1; см. ниже про патч |
| `allin1` | (закреплено) | Структурный анализ. `docker/ml-worker/Dockerfile` запускает `patch_natten_compat.py`, чтобы поправить совместимость с `natten>=0.15` |
| `basic-pitch` | `0.4.0` | Ставится с `--no-deps` (его tf-pin конфликтует), потом доустанавливаются runtime-зависимости |
| `ml_collections`, `omegaconf`, `beartype`, `rotary_embedding_torch`, `loralib` | см. `docker/ml-worker/requirements.txt` | Runtime-зависимости vendored `music_source_separation_training` для BS-RoFormer inference helpers, включая Instrumental Resurrection |
| `faiss-cpu` | (последняя) | Поиск похожих эмбеддингов; CPU-вариант намеренно (CUDA-faiss не даёт выигрыша на текущих объёмах) |
| `librosa`, `soundfile`, `audioread`, `scipy`, `numpy`, `scikit-learn` | стандартные | DSP-набор |
| `psycopg2-binary` | | Подключение к Postgres |
| `redis` | | Клиент Redis |
| `yt-dlp` | | YouTube-импорт (есть и в `app`, и в воркере) |
| `pytest` | `>=8.3` | Test runner для `ml_worker_improved/tests`, где часть файлов импортирует `pytest` напрямую |
| `structlog`, `orjson`, `python-dotenv`, `tqdm`, `pyyaml`, `requests`, `pathlib2`, `typing-extensions` | | Утилиты |

## 7. Логирование

`main.py:43-87` поднимает три RotatingFileHandler-а:

| Логгер | Файл | Размер ротации | Backup |
|--------|------|----------------|--------|
| root (INFO+) | `${STORAGE_PATH}/logs/ml_worker.log` | 10 МБ | 5 |
| root (ERROR+) | `${STORAGE_PATH}/logs/ml_worker_errors.log` | 5 МБ | 3 |
| `disk_telemetry` | `${STORAGE_PATH}/logs/ml_worker_disk.log` | 10 МБ | 5 |

Имена файлов перекрываются env (`ML_WORKER_LOG_NAME`/`ERROR_LOG_NAME`/`DISK_LOG_NAME`) — для `ml-worker-2/3` уже выставлены уникальные имена, чтобы не пересекались на одном bind-mount.

## 8. Связанные с кодом, но не очевидные детали

- **Sidecar-файлы** (`embedding_path`, `chroma_path`, `pitch_contour_path`) пишутся на стороне Laravel при `Track::created`/`updated` через `App\Support\TrackSidecarWriter`, а ML-воркер только перезаписывает их содержимое. Удаление трека → каскадное удаление файлов.
- **Lab-сессии Stem Tuning** регистрируются в `config/stem_tuning_lab.php`, и `StemTuningRenderService` (Laravel) проверяет наличие конкретного `track_id` в этом конфиге перед постановкой задачи.
- **Sequential rating remap** (`main.py:_apply_segment_rating_remap`) выполняется перед началом обработки — чтобы legacy-`pattern_examples.rating` оставался согласован с новым `user_pattern_example_ratings`.
- **`build_rating_safe_segment_remap` + `evaluate_rating_safe_remap_acceptance`** (модуль `segment_rating_remap`) — гарантия, что любой пересегмент не «потеряет» уже выставленные оценки.

## 9. Где искать модельные веса

Внутри контейнера:

- `/app/models/` — bind на `docker_data/models/`. Пользовательские модели и checkpoint'ы.
- `/app/storage/cache/huggingface/hub/` — модели с HuggingFace, скачанные на сборке (`HF_HUB_OFFLINE=1` в рантайме).
- `/app/vendor/` — bind на `ml_worker_improved/vendor/` в репозитории. Содержит:
  - `bs_mamba2_vocals/`
  - `mdx23c_vocals_official/`
  - `mel_band_roformer_vocal_model/`
  - и другие источники сепараторов, см. реальный листинг `ml_worker_improved/vendor/`.

## 10. Тесты

- Юнит-тесты Python: `ml_worker_improved/tests/test_*.py`. Запуск:

```powershell
docker compose exec ml-worker pytest tests/
```

## 11. Querysep long-tail task

`querysep_long_tail_build` зарегистрирован в основном `ml_worker_improved` как production task contract для long-tail stem-ов (`strings_orchestra`, `brass_woodwinds`, `synth_pad`, `fx_risers_impacts` и смежные классы).

Контракт намеренно fail-fast: worker принимает только `backend=banquet_byoq`, `dry_run=false`, `metadata.routing_mode=banquet_byoq` и локально видимый output через shared-volume `path_mappings`. `clap_router`, proxy из `other` и невидимый private output не считаются успешным результатом.

Подробный payload, Docker-only validation и shared-path contract описаны в [`QUERYSEP_LONG_TAIL_BUILD.md`](./QUERYSEP_LONG_TAIL_BUILD.md).

- Интеграция с Laravel-сторнкой: `tests/Feature/` (PHPUnit) — гонять через `docker compose exec app php artisan test`.

> Тестовое окружение Laravel в `phpunit.xml` использует SQLite in-memory; чисто Python-тесты Docker-зависимы (нужен GPU для большинства).
