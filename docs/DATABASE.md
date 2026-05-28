# Схема базы данных

PostgreSQL 16 (`postgres:16-alpine` в `docker-compose.yml:269`). База — `${DB_DATABASE:-music_recommendations}`. Все таблицы создаются миграциями из `database/migrations/` (**37 `.php`-файлов** + `.gitkeep`; ранее упоминание «38 миграций» считало `.gitkeep` за миграцию — это неверно).

> Тестовый прогон phpunit (`phpunit.xml`) использует **in-memory SQLite** (`DB_CONNECTION=sqlite`, `DB_DATABASE=:memory:`). Это значит, что часть PostgreSQL-специфичных конструкций (`jsonb`, GIN-индексы, `ALTER TABLE ... CHECK`) при тестировании может работать иначе. Учитывайте при написании тестов.

## 1. Полный список миграций (хронологически)

```
2024_01_01_000001_create_tracks_table.php
2024_01_01_000002_create_fragments_table.php
2024_01_01_000003_create_preference_model_status_table.php
2024_01_01_000004_create_queue_tasks_table.php
2024_01_15_000001_add_comment_and_rated_at_to_fragments_table.php
2026_01_28_132542_create_failed_jobs_table.php
2026_01_28_132543_create_jobs_table.php
2026_01_28_142952_add_unique_index_to_youtube_url.php
2026_01_29_000001_create_playlists_table.php
2026_01_29_000002_create_playlist_tracks_table.php
2026_01_29_000003_create_track_ratings_table.php
2026_01_29_000004_create_rating_history_table.php
2026_01_29_000005_create_user_track_preferences_table.php
2026_01_29_000006_create_listening_history_table.php
2026_01_29_000007_create_track_skips_table.php
2026_01_29_000008_create_radio_feedback_table.php
2026_01_29_000009_create_track_dj_analysis_table.php
2026_01_29_000010_create_user_settings_table.php
2026_01_29_100000_add_youtube_channel_to_tracks_table.php
2026_01_29_113311_add_jsonb_tags_to_tracks_table.php
2026_01_29_200001_create_patterns_table.php
2026_01_29_200002_create_pattern_examples_table.php
2026_01_29_200003_create_pattern_matches_table.php
2026_01_29_200004_add_chroma_path_to_tracks_table.php
2026_01_30_163153_add_preference_score_to_tracks_table.php
2026_03_14_000001_update_pattern_examples_for_feedback_and_embeddings.php
2026_03_14_000002_create_preference_model_training_runs_table.php
2026_03_22_000011_add_loudness_fields_to_tracks_table.php
2026_03_22_000012_create_users_and_api_tokens_tables.php
2026_03_22_000013_add_user_scope_to_personalization_tables.php
2026_03_22_000014_add_user_scoped_preference_models.php
2026_03_22_000015_add_user_scoped_feedback_signals.php
2026_03_22_000016_add_user_id_to_radio_feedback_table.php
2026_03_24_000017_create_friendships_and_direct_messages_tables.php
2026_03_24_000018_add_user_id_to_track_skips_table.php
2026_03_26_000001_add_artist_credits_to_tracks_table.php
2026_03_26_000002_backfill_artist_slugs_and_add_index.php
```

## 2. Ключевые таблицы

### 2.1 `tracks` (`2024_01_01_000001`, последующие альтер-миграции)

Базовый набор колонок (источник: `2024_01_01_000001_create_tracks_table.php`):

| Колонка | Тип | Назначение |
|---------|-----|------------|
| `id` | bigint PK | |
| `title` | string(255) | |
| `artist` | string(255) nullable | (далее заменено `artist_credits` jsonb-структурой) |
| `album` | string(255) nullable | |
| `year` | integer nullable | |
| `duration` | float | в секундах |
| `file_path` | string(500) | путь к локальному файлу относительно `STORAGE_PATH` |
| `source` | string(50) | `'upload' | 'youtube'` |
| `youtube_url` | string(500) nullable | (с уникальным индексом, добавлен `2026_01_28_142952`) |
| `genre` | string(100) nullable | результат ML |
| `bpm` | float nullable | |
| `key` | string(10) nullable | например `C#` |
| `mode` | string(10) nullable | `major | minor` |
| `mood` | string(50) nullable | |
| `has_vocals` | boolean nullable | |
| `is_acoustic` | boolean nullable | |
| `analysis_status` | string(20) default `'pending'` | `pending | processing | completed | failed` |
| `is_liked` | boolean default false | глобальный лайк трека (старая модель, до user-scoped) |
| `manually_edited` | jsonb default `{}` | какие поля правил пользователь вручную (запрещает перезапись из ML) |
| `embedding_path` | string(500) nullable | путь к sidecar с эмбеддингом |
| `created_at`, `updated_at` | timestamp | |

Индексы: по `genre`, `mood`, `bpm`, `is_liked`, `analysis_status`.

Дополнительно (через альтер-миграции):

- `chroma_path` (string nullable, `2026_01_29_200004`) — путь к sidecar с хромограммой.
- `pitch_contour_path` — добавлен в pattern-search ветке.
- `loudness_*` (`integrated`, `range`, `true_peak`, ...) — `2026_03_22_000011`.
- `preference_score` — `2026_01_30_163153`.
- `youtube_channel_id` / `youtube_channel_title` — `2026_01_29_100000`.
- `tags` jsonb — `2026_01_29_113311`.
- `artist_credits` jsonb + `artist_slugs` — `2026_03_26_000001/000002` (см. `App\Services\ArtistCreditParser`).

### 2.2 `fragments` (`2024_01_01_000002`)

| Колонка | Тип | Назначение |
|---------|-----|------------|
| `id` | bigint PK | |
| `track_id` | FK → tracks.id, `cascade` | |
| `start_time`, `end_time` | float | CHECK `end_time > start_time` (`ALTER TABLE ... ADD CONSTRAINT valid_time_range`) |
| `is_liked` | boolean default false | |
| `embedding_path` | string(500) nullable | |
| `analysis_status` | string(20) default `'pending'` | |
| `created_at` | timestamp | (только creation, без `updated_at` — модель `Fragment` явно `UPDATED_AT = null`) |

Дополнительно:
- `comment` (string nullable), `rated_at` (timestamp nullable) — `2024_01_15_000001`.
- `user_id` (FK users) — `2026_03_22_000013`.

### 2.3 `queue_tasks` (`2024_01_01_000004`)

| Колонка | Тип | Назначение |
|---------|-----|------------|
| `id` | bigint PK | |
| `task_type` | string(50) | `analysis | fragment_analysis | training | recommendation | stem_source_build | querysep_long_tail_build | stem_tuning_render` |
| `payload` | jsonb | произвольные данные задачи |
| `status` | string(20) default `'pending'` | + `processing | completed | failed | cancelled` (см. `App\Models\QueueTask::STATUS_*`) |
| `error_message` | text nullable | |
| `created_at`, `completed_at` | timestamp | (без `updated_at`) |

Индексы: `task_type`, `status`.

### 2.4 `users` и `api_tokens` (`2026_03_22_000012`)

`users`:
- `id`, `name`, `email` UNIQUE, `password` (cast `'hashed'`), `role` (default `'user'`, индекс), `remember_token`, `timestamps`.

`api_tokens`:
- `id`, `user_id` FK cascade, `name` (string 100 nullable), `token_hash` (CHAR-64 UNIQUE — SHA256), `last_used_at` nullable + индекс, `expires_at` nullable, `created_at`.

Логика — в `app/Http/Middleware/AuthenticateApiToken.php`. Сам plaintext-токен **никогда** не хранится.

### 2.5 Стандартные Laravel-таблицы

- `failed_jobs` (`2026_01_28_132542`) — стандартная таблица Laravel для упавших job'ов очереди.
- `jobs` (`2026_01_28_132543`) — fall-back, если очередь принудительно переключат на `database`.

`QUEUE_CONNECTION=redis` (`docker-compose.yml:34`), поэтому в нормальном режиме `jobs` пустая. Однако **внутри `app`-контейнера через `supervisord` крутится 12 процессов `php artisan queue:work`** (8 для `youtube`, 2 для `analysis,default`, по 1 для `radio-prewarm` и `radio-hls-prewarm`, см. `docker/app/supervisord.conf`). Они живут в Redis по стандартному laravel-queue протоколу и реально участвуют в работе — например, через них летят job'ы импорта YouTube. Параллельно с ними существует **отдельная** ML-очередь (Redis-список `ml_worker:tasks`), которой управляет `App\Services\MLQueueService` и потребляют Python-воркеры. Не путайте эти два потока.

## 3. Полный table-by-table reference (по всем 37 миграциям)

Источник для каждой строки — соответствующий файл в `database/migrations/`. Если поле добавлено более поздней миграцией, это указано в колонке «Источник». Все FK по умолчанию `cascade on delete`, исключения отмечены явно.

### 3.1 `tracks` — каталог треков

Базовая миграция: `2024_01_01_000001_create_tracks_table.php`.

| Колонка | Тип | Источник | Назначение |
|---------|-----|----------|------------|
| `id` | bigint PK | `_000001` | |
| `title` | string(255) | `_000001` | |
| `artist` | string(255) nullable | `_000001` | legacy plain-text; теперь дополнен `artist_credits`/`artist_slugs` |
| `artist_credits` | jsonb nullable | `2026_03_26_000001` | список исполнителей `[{name, role, slug}, ...]` (см. `ArtistCreditParser`) |
| `artist_slugs` | jsonb nullable | `2026_03_26_000001` | плоский массив slug'ов; GIN-индекс `idx_tracks_artist_slugs_gin` создаёт `2026_03_26_000002` |
| `album` | string(255) nullable | `_000001` | |
| `year` | integer nullable | `_000001` | |
| `duration` | float | `_000001` | в секундах |
| `file_path` | string(500) | `_000001` | путь относительно `storage/app/audio/...` |
| `source` | string(50) | `_000001` | `'upload' \| 'youtube'` |
| `youtube_url` | string(500) nullable | `_000001` | |
| `youtube_video_id` | string(20) nullable UNIQUE | `2026_01_28_142952` | защита от двойного импорта |
| `youtube_channel` | string nullable | `2026_01_29_100000` | имя канала, отдельно от `artist` |
| `genre` | string(100) nullable | `_000001` | основной жанр (top-1 от ML) |
| `genre_tags` | jsonb nullable | `2026_01_29_113311` | top-N тегов с вероятностями `[{tag, prob}]` |
| `bpm` | float nullable | `_000001` | |
| `key` | string(10) nullable | `_000001` | например `C#` |
| `mode` | string(10) nullable | `_000001` | `major \| minor` |
| `mood` | string(50) nullable | `_000001` | основной mood |
| `mood_tags` | jsonb nullable | `2026_01_29_113311` | top-N с вероятностями |
| `has_vocals` | boolean nullable | `_000001` | |
| `is_acoustic` | boolean nullable | `_000001` | |
| `loudness_integrated_lufs` | float nullable | `2026_03_22_000011` | LUFS Integrated |
| `loudness_true_peak_db` | float nullable | `2026_03_22_000011` | True-peak dBTP |
| `loudness_gain_db` | float nullable | `2026_03_22_000011` | вычисленный gain для нормализации |
| `loudness_target_lufs` | float nullable | `2026_03_22_000011` | цель нормализации |
| `analysis_status` | string(20) default `'pending'` | `_000001` | `pending \| processing \| completed \| failed` |
| `is_liked` | boolean default false | `_000001` | глобальный лайк (legacy; user-scoped лайки лежат в `user_track_likes`) |
| `manually_edited` | jsonb default `{}` | `_000001` | какие поля юзер правил руками — ML их не перезаписывает |
| `embedding_path` | string(500) nullable | `_000001` | sidecar с эмбеддингом |
| `chroma_path` | string nullable | `2026_01_29_200004` | sidecar с хромограммой |
| `pitch_contour_path` | string nullable | `2026_01_29_200004` | sidecar с pitch-контуром |
| `preference_score` | float nullable + index | `2026_01_30_163153` | глобальная (legacy) preference-score; user-scoped — в `user_track_scores` |
| `score_calculated_at` | timestamp nullable | `2026_01_30_163153` | |
| `created_at`, `updated_at` | timestamp | `_000001` | оба `useCurrent()` |

Индексы базовой миграции: `idx_tracks_genre`, `idx_tracks_mood`, `idx_tracks_bpm`, `idx_tracks_liked`, `idx_tracks_analysis_status`. GIN — `idx_tracks_artist_slugs_gin` (`pgsql`-only, `2026_03_26_000002`).

### 3.2 `fragments` — пользовательские фрагменты

Базовая миграция: `2024_01_01_000002_create_fragments_table.php`.

| Колонка | Тип | Источник | Назначение |
|---------|-----|----------|------------|
| `id` | bigint PK | `_000002` | |
| `track_id` | FK → tracks.id, cascade | `_000002` | |
| `user_id` | FK → users.id, cascade nullable | `2026_03_22_000015` | владелец фрагмента (null для legacy-данных) |
| `start_time`, `end_time` | float | `_000002` | CHECK `end_time > start_time` (constraint `valid_time_range`) |
| `is_liked` | boolean nullable | `_000002` → `2024_01_15_000001` | `null` = не оценён, `true/false` = лайк/дизлайк |
| `comment` | text nullable | `2024_01_15_000001` | до 500 символов на уровне валидатора (поле без жёсткого limit) |
| `rated_at` | timestamp nullable | `2024_01_15_000001` | когда оценили |
| `embedding_path` | string(500) nullable | `_000002` | sidecar |
| `analysis_status` | string(20) default `'pending'` | `_000002` | |
| `created_at` | timestamp | `_000002` | без `updated_at` (модель `Fragment` явно `UPDATED_AT = null`) |

Индексы: `idx_fragments_track_id`, `idx_fragments_liked`, `idx_fragments_user_track` (`2026_03_22_000015`).

### 3.3 `preference_model_status`

Базовая миграция: `2024_01_01_000003`. User-scope добавлен `2026_03_22_000014`.

| Колонка | Тип | Источник |
|---------|-----|----------|
| `id` | bigint PK | `_000003` |
| `user_id` | FK → users.id, cascade nullable | `2026_03_22_000014`, UNIQUE по `user_id` (`preference_model_status_user_unique`) |
| `status` | string(20) default `'not_trained'` | `not_trained \| training \| trained` |
| `model_path` | string(500) nullable | путь к pickle/torch-файлу модели |
| `trained_at` | timestamp nullable | |
| `likes_count` | integer default 0 | сколько лайков было на момент тренировки |
| `created_at`, `updated_at` | timestamp | оба `useCurrent()` |

### 3.4 `queue_tasks` — учёт ML-задач

Базовая миграция: `2024_01_01_000004`.

| Колонка | Тип | Назначение |
|---------|-----|------------|
| `id` | bigint PK | |
| `task_type` | string(50) | `analysis \| fragment_analysis \| training \| recommendation \| stem_source_build \| querysep_long_tail_build \| stem_tuning_render \| segment_analysis \| ...` (полный список — в `App\Models\QueueTask::TYPE_*`) |
| `payload` | jsonb | произвольные данные задачи |
| `status` | string(20) default `'pending'` | `pending \| processing \| completed \| failed \| cancelled` (`STATUS_*` в модели) |
| `error_message` | text nullable | |
| `created_at`, `completed_at` | timestamp | без `updated_at` |

Индексы: `idx_queue_tasks_type`, `idx_queue_tasks_status`.

### 3.5 `failed_jobs` (`2026_01_28_132542`)

Стандартная Laravel-таблица: `id`, `uuid` UNIQUE, `connection`, `queue`, `payload` (longtext), `exception` (longtext), `failed_at`. Используется `php artisan queue:work --tries=3` (см. `docker/app/supervisord.conf`).

### 3.6 `jobs` (`2026_01_28_132543`)

Fallback для `QUEUE_CONNECTION=database`. Колонки: `id` (bigInc), `queue` (index), `payload`, `attempts` (uint8), `reserved_at` nullable, `available_at`, `created_at`. В рантайме обычно пуста (`QUEUE_CONNECTION=redis`), но миграция создаёт таблицу всегда.

### 3.7 `playlists` (`2026_01_29_000001`)

| Колонка | Тип | Источник |
|---------|-----|----------|
| `id` | bigint PK | `_000001` |
| `user_id` | FK → users.id nullable, `nullOnDelete` | `2026_03_22_000013`, индекс `idx_playlists_user` |
| `name` | string(255) | `_000001` |
| `description` | text nullable | `_000001` |
| `is_ai_generated` | boolean default false | `_000001`, индекс `idx_playlists_ai_generated` |
| `track_count` | integer default 0 | `_000001`, кеш-счётчик |
| `total_duration` | integer default 0 | `_000001`, секунды |
| `created_at`, `updated_at` | timestamp | оба `useCurrent()`; индекс `idx_playlists_created_at` |

### 3.8 `playlist_tracks` (`2026_01_29_000002`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `playlist_id` | FK cascade |
| `track_id` | FK cascade |
| `position` | integer (для drag-n-drop reorder) |
| `added_at` | timestamp `useCurrent()` |

UNIQUE `uq_playlist_tracks(playlist_id, track_id)`. Индексы: `idx_playlist_tracks_playlist`, `_track`, `_position`.

### 3.9 `track_ratings` (`2026_01_29_000003`, user-scope `_000013`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK → users.id nullable, `nullOnDelete` (добавлен `_000013`) |
| `track_id` | FK cascade. UNIQUE по `(user_id, track_id)` (`track_ratings_user_track_unique`); до `_000013` была уникальность только по `track_id` |
| `rating` | smallint nullable. CHECK `rating IS NULL OR (rating BETWEEN 1 AND 5)` (constraint `chk_rating_range`) |
| `created_at`, `updated_at` | timestamp |

Индексы: `idx_track_ratings_rating`, `idx_track_ratings_user`.

### 3.10 `rating_history` (`2026_01_29_000004`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK nullable, `nullOnDelete` (добавлен `_000013`) |
| `track_id` | FK → tracks.id nullable, `nullOnDelete` (логи переживают удаление трека) |
| `old_rating`, `new_rating` | smallint nullable |
| `changed_at` | timestamp `useCurrent()` |

Индексы: `idx_rating_history_track`, `_changed_at`, `_user`.

### 3.11 `user_track_preferences` (`2026_01_29_000005`, user-scope `_000013`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK nullable, `nullOnDelete` (`_000013`) |
| `track_id` | FK cascade. UNIQUE по `(user_id, track_id)` (`user_track_preferences_user_track_unique`); до `_000013` — уникальность только по `track_id` |
| `volume` | float default 1.0 |
| `last_position` | float default 0 (секунды) |
| `updated_at` | timestamp |

Индексы: `idx_user_track_preferences_track`, `_user`.

### 3.12 `listening_history` (`2026_01_29_000006`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `track_id` | FK cascade |
| `user_id` | FK cascade nullable (`2026_03_22_000015`) |
| `started_at` | timestamp `useCurrent()` |
| `duration_played` | float nullable (секунды) |
| `completed` | boolean default false |
| `source` | string(50) nullable: `queue \| playlist \| radio \| ai_dj` |

Индексы: `idx_listening_history_track`, `_started_at`, `_source`, `idx_listening_history_user_started`.

### 3.13 `track_skips` (`2026_01_29_000007`, user-scope `_000018`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `track_id` | FK cascade |
| `user_id` | FK nullable, `nullOnDelete` (`_000018`) |
| `position_at_skip` | float nullable (секунды) |
| `source` | string(50) nullable |
| `skipped_at` | timestamp `useCurrent()` |

Индексы: `idx_track_skips_track`, `_skipped_at`, `idx_track_skips_user_track`.

### 3.14 `radio_feedback` (`2026_01_29_000008`, user-scope `_000016`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK nullable, `nullOnDelete` (`_000016`) |
| `track_id` | FK cascade |
| `feedback_type` | string(10). CHECK `feedback_type IN ('more','less')` (`chk_feedback_type`) |
| `created_at` | timestamp |

Индексы: `idx_radio_feedback_track`, `_type`, `_created_at`, `(user_id, track_id)` (`_000016`).

### 3.15 `track_dj_analysis` (`2026_01_29_000009`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `track_id` | FK cascade UNIQUE |
| `camelot_key` | string(3) (например `8A`, `11B`) |
| `bpm` | float |
| `energy_level` | smallint. CHECK `energy_level IS NULL OR (1..10)` (`chk_energy_level_range`) |
| `intro_point`, `outro_point` | float (секунды) |
| `drop_points`, `breakdown_points` | jsonb default `[]` (массивы секунд) |
| `analyzed_at` | timestamp `useCurrent()` |

Индексы: `idx_track_dj_analysis_key`, `_bpm`, `_energy`. Источник анализа — `ml_worker_improved/worker/dj_analyzer.py`.

### 3.16 `user_settings` (`2026_01_29_000010`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `key` | string(100) UNIQUE |
| `value` | jsonb |
| `updated_at` | timestamp |

Индекс `idx_user_settings_key`. **Важно:** в самой миграции `user_id` нет — таблица хранит **глобальные** настройки в виде ключ-значение. Per-user настройки следует выражать в `key` (например `theme:user:1`) — конкретные ключи определяются в коде сервисов.

### 3.17 `patterns` (`2026_01_29_200001`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `name` | string |
| `description` | text nullable |
| `representation_type` | string default `'chroma'` (`pitch_contour \| chroma \| midi`) |
| `transpose_invariant`, `tempo_invariant` | boolean default true |
| `match_threshold` | float default 0.7 |
| `importance` | integer default 1 (валидация 0..3) |
| `prototype_path` | string nullable (sidecar с усреднённым прототипом) |
| `timestamps()` | стандарт |

### 3.18 `pattern_examples` (`2026_01_29_200002`, расширения `_000001/2026_03_14`)

| Колонка | Тип | Источник |
|---------|-----|----------|
| `id` | bigint PK | `_200002` |
| `pattern_id` | FK cascade | `_200002` |
| `track_id` | FK cascade | `_200002` |
| `start_time`, `end_time` | float | `_200002` |
| `is_positive` | boolean nullable (исходно NOT NULL default true; `2026_03_14_000001` → DROP NOT NULL → `null` = «не оценён»; worker INSERT'ит NULL для авто-сегментов) | `_200002` → `2026_03_14_000001` |
| `notes` | text nullable | `_200002` |
| `representation_path` | string nullable | `_200002` |
| `rating` | float nullable (0..5 stars) | `2026_03_14_000001` |
| `embedding_path` | string(500) nullable | `2026_03_14_000001` |
| `timestamps()` | | `_200002` |

Индексы: `(pattern_id, is_positive)`, `idx_pattern_examples_track_id` (`pgsql` `CREATE INDEX IF NOT EXISTS`).

### 3.19 `pattern_matches` (`2026_01_29_200003`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `pattern_id` | FK cascade |
| `track_id` | FK cascade |
| `start_time`, `end_time` | float |
| `score` | float (DTW-расстояние или similarity) |
| `confirmed` | boolean nullable (`null` — не ревьювили, `true/false` — пользовательский вердикт) |
| `timestamps()` | |

Индексы: `(pattern_id, score)`, `(track_id, pattern_id)`.

### 3.20 `preference_model_training_runs` (`2026_03_14_000002`, user-scope `_000014`)

История каждого запуска обучения preference-model. Все integer-метрики стартуют с 0; `meta` — произвольный JSON для быстрых отладочных дампов.

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `queue_task_id` | unsignedBigInt nullable (без FK — задача может быть удалена) |
| `user_id` | FK → users.id nullable, cascade (`_000014`) |
| `model_path` | string(500) nullable |
| `started_at`, `finished_at` | timestamp nullable |
| `final_loss` | float nullable |
| `positives_total`, `_tracks`, `_fragments`, `_segments` | integer default 0 |
| `explicit_negatives_total`, `_fragments`, `_segments` | integer default 0 |
| `all_tracks_pool`, `tracks_scored`, `tracks_with_segments`, `segment_improved` | integer default 0 |
| `segments_rated_total`, `segments_rated_with_embedding` | integer default 0 |
| `meta` | jsonb nullable |
| `timestamps()` | стандарт |

Индексы: `idx_pmt_runs_queue_task_id`, `idx_pmt_runs_created_at`, `idx_pmt_runs_user_id`.

### 3.21 `users` (`2026_03_22_000012`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `name` | string |
| `email` | string UNIQUE |
| `password` | string (cast `'hashed'` в `App\Models\User`) |
| `role` | string(32) default `'user'` (значимое — `'admin'`; используется middleware `role:admin`) |
| `remember_token` | стандартный |
| `timestamps()` | |

Индекс `idx_users_role`.

### 3.22 `api_tokens` (`2026_03_22_000012`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK cascade |
| `name` | string(100) nullable (метка токена для UI) |
| `token_hash` | string(64) UNIQUE — **SHA-256-хеш** plaintext-токена |
| `last_used_at` | timestamp nullable (антидребезг 300 с — Redis-ключ `api_token:last_used_touch:{tokenId}`) |
| `expires_at` | timestamp nullable |
| `created_at` | timestamp `useCurrent()` |

Индексы: `idx_api_tokens_user`, `_last_used_at`. Plaintext-токен **никогда** не хранится. Логика выпуска/проверки — `app/Http/Middleware/AuthenticateApiToken.php`.

### 3.23 `user_track_scores` (`2026_03_22_000014`)

User-scoped результат preference-модели (заменяет глобальный `tracks.preference_score`).

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK cascade |
| `track_id` | FK cascade |
| `preference_score` | double |
| `score_calculated_at` | timestamp nullable |
| `timestamps()` | |

UNIQUE `(user_id, track_id)`. Индекс `(user_id, preference_score)` — ровно под топ-N запросы.

### 3.24 `user_track_likes` (`2026_03_22_000015`)

User-scoped лайки (вытесняют `tracks.is_liked`).

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK cascade |
| `track_id` | FK cascade |
| `timestamps()` | |

UNIQUE `(user_id, track_id)` (`uq_user_track_likes_user_track`). Индекс `idx_user_track_likes_track_id`.

### 3.25 `user_pattern_example_ratings` (`2026_03_22_000015`)

User-scoped звёздные оценки конкретных сегментов (`pattern_examples`).

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `user_id` | FK cascade |
| `pattern_example_id` | FK cascade |
| `rating` | float |
| `is_positive` | boolean (контроллер выставляет `rating >= 3`) |
| `timestamps()` | |

UNIQUE `(user_id, pattern_example_id)`. Индекс `(user_id, rating)`. Воркер делает `INSERT ... ON CONFLICT (user_id, pattern_example_id)` (см. `ml_worker_improved/worker/main.py`).

### 3.26 `friendships` (`2026_03_24_000017`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `requester_id` | FK → users.id cascade |
| `addressee_id` | FK → users.id cascade |
| `status` | string(24) default `'pending'` (значимые — `'pending' \| 'accepted' \| 'declined' \| 'cancelled'`) |
| `responded_at`, `accepted_at` | timestamp nullable |
| `timestamps()` | |

UNIQUE `(requester_id, addressee_id)`. Индексы по `(requester_id, status)` и `(addressee_id, status)`.

### 3.27 `direct_messages` (`2026_03_24_000017`)

| Колонка | Тип |
|---------|-----|
| `id` | bigint PK |
| `sender_id` | FK → users.id cascade |
| `recipient_id` | FK → users.id cascade |
| `type` | string(32) default `'message'` (`message \| segment_share \| ...`) |
| `body` | text nullable |
| `payload` | jsonb nullable (для `segment_share` хранит структуру сегмента) |
| `read_at` | timestamp nullable |
| `timestamps()` | |

Индексы: `(recipient_id, read_at)` (быстрый «непрочитанные»), `(sender_id, recipient_id, created_at)`, `(recipient_id, sender_id, created_at)` — обе стороны диалога идут по индексу.

### 3.28 Backfill `_000002` (`2026_03_26_000002`)

Это **data-only** миграция: пробегает все `tracks` чанками по 200 (`chunkById`), вызывает `ArtistCreditParser::resolveFromAttributes(...)` и обновляет `title`, `artist`, `artist_credits`, `artist_slugs`. Затем (только на `pgsql`) создаёт `CREATE INDEX IF NOT EXISTS idx_tracks_artist_slugs_gin ON tracks USING gin (artist_slugs)`. На SQLite (тестовый прогон) GIN-индекс не создаётся.

> **Отдельная заметка про test-прогон.** `phpunit.xml` использует in-memory SQLite. Поэтому: (1) constraint'ы `CHECK` из миграций пропускаются молча, (2) `jsonb` мапится на `TEXT`, (3) `CREATE INDEX ... USING gin` внутри `if (DB::getDriverName() === 'pgsql')` исключается. Если делаете тест на jsonb-фильтр или GIN — придётся либо переключить тест на pgsql, либо явно проверять драйвер.

## 4. Конвенции

- **Миграции с датами 2024_01_01_*** — это «псевдо-исторический» базовый блок; даты не отражают реальное время коммитов, нужны только для упорядочивания.
- **`jsonb` повсеместно** для гибких структур (`manually_edited`, `tags`, `payload`, `artist_credits`, превью `loudness_*` — где применимо).
- **Soft-delete не используется** ни в одной из прочитанных миграций. Удаление — каскадное (`onDelete('cascade')`/`cascadeOnDelete`).
- **Артистные имена** хранятся одновременно как `artist` (legacy string) и `artist_credits`/`artist_slugs` (jsonb + плоский поисковый список). Парсер — `App\Services\ArtistCreditParser` (используется аксессорами `Track::artist`, `artistCredits`, `artistSlugs`).

## 5. Сайдкары на диске

Кроме реляционных полей, ML-результаты живут на диске:

| Поле в `tracks` | Где файл | Что внутри |
|------------------|----------|------------|
| `embedding_path` | `storage/app/<path>` | Numpy-эмбеддинг трека (генерируется `audio_analyzer.py`) |
| `chroma_path` | `storage/app/<path>` | Хромограмма (для pattern search) |
| `pitch_contour_path` | `storage/app/<path>` | Pitch-контур |

Sidecar пишутся/удаляются через `App\Support\TrackSidecarWriter`, который вызывается из `App\Models\Track::booted()` при создании/обновлении/удалении `file_path`.

## 6. Полезные SQL-проверки

```sql
-- Сколько треков в каком статусе
SELECT analysis_status, count(*) FROM tracks GROUP BY 1;

-- Активные ML-задачи
SELECT task_type, status, count(*) FROM queue_tasks GROUP BY 1, 2 ORDER BY 1, 2;

-- Замусоренные failed-задачи
SELECT id, task_type, error_message, created_at
FROM queue_tasks
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 50;

-- Покрытие сегментов (приблизительно — точный отчёт даёт `php artisan segments:coverage --json`)
SELECT track_id, count(*) AS segments,
       count(*) FILTER (WHERE embedding_path IS NOT NULL) AS with_embedding
FROM pattern_examples
GROUP BY track_id
ORDER BY segments DESC;
```

## 7. Что не задокументировано в миграциях

- Точный набор колонок миграций после `_000003`-`_000018` не разворачивался в этом документе. Перед изменением схемы прочитайте конкретный файл миграции; имена и индексы держатся в стиле `idx_<table>_<columns>`.
- Если нужен полный дамп схемы, выполните:

```powershell
docker compose exec postgres pg_dump -s -U postgres music_recommendations > schema.sql
```
