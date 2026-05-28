# Справочник API

Полный реестр HTTP-эндпоинтов основан непосредственно на `routes/api.php` и `routes/web.php`. Глубже в формате запроса/ответа — лучше смотреть конкретный контроллер в `app/Http/Controllers/` (имя берите из колонки «Controller»).

## Соглашения

- **База:** `http://localhost:${APP_PORT:-8000}` (`docker-compose.yml:11`).
- **Авторизация:** Bearer-токен в заголовке `Authorization: Bearer <token>` (либо `?token=...` query-параметром). Реализация — `app/Http/Middleware/AuthenticateApiToken.php` (см. [`ARCHITECTURE.md` §4](./ARCHITECTURE.md#4-аутентификация)).
- **Контент-тип:** JSON по умолчанию. Стримовые эндпоинты возвращают `audio/*`, HLS — `application/vnd.apple.mpegurl`, SSE — `text/event-stream`.
- **Группа `auth.token`:** требует валидный токен.
- **Группа `auth.token` + `role:admin`:** требует токен, чей `users.role = 'admin'`.

## 1. Публичные эндпоинты (только `api` middleware)

| Метод | URL | Controller@action | Назначение |
|-------|-----|--------------------|------------|
| GET | `/api/health` | inline closure | Простой пинг (`{status, timestamp}`) |
| GET | `/up` | (Laravel built-in health) | Health-чек по умолчанию |
| POST | `/api/auth/register` | `AuthController@register` | Создание пользователя |
| POST | `/api/auth/login` | `AuthController@login` | Логин, выдача Bearer-токена |
| GET | `/api/tracks/{id}/stream` | `TrackController@stream` | Прямой стрим аудиофайла (с Range) |
| GET | `/api/tracks/{id}/playback-options` | `TrackController@playbackOptions` | Доступные режимы воспроизведения (HLS/стемы) |
| GET | `/api/tracks/{id}/playback/modes/{modeKey}/stream` | `TrackController@playbackModeStream` | Стрим конкретного режима |
| GET | `/api/tracks/{id}/playback/stems/{stemKey}/stream` | `TrackController@playbackStemStream` | Стрим отдельного стема |
| GET | `/api/tracks/{id}/waveform` | `TrackController@waveform` | JSON-пиков волны |
| GET | `/api/tracks/{id}/hls/master.m3u8` | `TrackController@hlsMaster` | HLS master-плейлист |
| GET | `/api/tracks/{id}/hls/{profile}/index.m3u8` | `TrackController@hlsVariant` | HLS variant |
| GET | `/api/tracks/{id}/hls/{profile}/{segment}` | `TrackController@hlsSegment` | HLS-сегмент |
| PUT | `/api/radio/personal-stations/{slug}/genres` | `RadioAccessController@updatePersonalStationGenres` | Жанры персональной станции |
| GET | `/api/public/radio/{slug}` | `PublicRadioController@show` | Снимок состояния публичного радио |
| GET | `/api/public/radio/{slug}/history` | `PublicRadioController@history` | История треков радио |
| POST | `/api/public/radio/{slug}/diagnostics` | `PublicRadioController@diagnostics` | Получает диагностику от плеера |
| GET | `/api/radio/private-overlay/{track}` | `RadioAccessController@privateOverlay` | Оверлей для приватного слоя |

## 2. Авторизованные (`auth.token`)

### 2.1 Auth (продолжение)

| Метод | URL | Controller |
|-------|-----|------------|
| GET | `/api/auth/me` | `AuthController@me` |
| POST | `/api/auth/logout` | `AuthController@logout` |

### 2.2 Tracks

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/tracks/queue-status` | `TrackController@queueStatus` |
| DELETE | `/api/tracks/batch` | `TrackController@destroyBatch` |
| POST | `/api/tracks/{track}/retry-analysis` | `TrackController@retryAnalysis` |
| GET | `/api/tracks` | `TrackController@index` |
| POST | `/api/tracks` | `TrackController@store` |
| PUT/PATCH | `/api/tracks/{id}` | `TrackController@update` |
| DELETE | `/api/tracks/{id}` | `TrackController@destroy` |
| GET | `/api/tracks/{id}` | `TrackController@show` |
| POST | `/api/tracks/youtube` | `TrackController@importYoutube` |
| POST | `/api/tracks/youtube/batch` | `TrackController@importYoutubeBatch` |

> Порядок маршрутов в `routes/api.php` намеренно ставит `tracks/queue-status`, `tracks/batch` и `tracks/{track}/retry-analysis` **до** `apiResource('tracks', ...)`, чтобы `{id}` не «съел» статические сегменты. Не меняйте порядок без понимания этого нюанса (`routes/api.php:80-84`).

### 2.3 Stem Tuning Lab

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/tracks/{track}/tuning-lab` | `StemTuningLabController@show` |
| POST | `/api/tracks/{track}/tuning-lab/build-stems` | `StemTuningLabController@buildStems` |
| GET | `/api/tracks/{track}/tuning-lab/build-stem-tasks/{taskId}` | `StemTuningLabController@buildStatus` |
| POST | `/api/tracks/{track}/tuning-lab/render` | `StemTuningLabController@render` |
| GET | `/api/tracks/{track}/tuning-lab/render-tasks/{taskId}` | `StemTuningLabController@renderStatus` |
| POST | `/api/tracks/{track}/tuning-lab/render-tasks/{taskId}/cancel` | `StemTuningLabController@cancelRender` |
| GET | `/api/tracks/{track}/tuning-lab/sources/{sourceKey}/stream` | `StemTuningLabController@sourceStream` |
| GET | `/api/tracks/{track}/tuning-lab/sources/{sourceKey}/waveform` | `StemTuningLabController@sourceWaveform` |

Лаборатория доступна только для треков, перечисленных в `config/stem_tuning_lab.php` (на сейчас — `track 15`). Для остальных `show` вернёт 404 с подсказкой `buildable: bool`.

> **Очередь:** оба `POST .../render` и `POST .../build-stems` используют `MLQueueService::pushPriority()` — задача типа `stem_tuning_render` / `stem_source_build` ставится в хвост ML-очереди, после чего вся очередь переупорядочивается так, что приоритетные task-types (см. полный список в [`ARCHITECTURE.md` §5 — Priority queueing](./ARCHITECTURE.md#priority-queueing-priority_task_types--pushpriority)) оказываются впереди фонового сегментного backlog. Из-за этого `task_id`, который вы получили в ответе, обычно поднимается на обработку быстрее, чем накопленные ранее `analyze_segments`-задачи.

### 2.4 Artists

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/artists/{slug}` | `ArtistController@show` |
| PUT | `/api/artists/{slug}` | `ArtistController@update` |

### 2.5 Fragments (вложены в треки, shallow)

```
GET    /api/tracks/{track}/fragments
POST   /api/tracks/{track}/fragments
GET    /api/fragments/{fragment}
PUT    /api/fragments/{fragment}
DELETE /api/fragments/{fragment}
```
Controller: `FragmentController` (`apiResource('tracks.fragments', ..., shallow: true)`).

### 2.6 Likes

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/tracks/{track}/like` | `LikeController@likeTrack` |
| DELETE | `/api/tracks/{track}/like` | `LikeController@unlikeTrack` |
| POST | `/api/fragments/{fragment}/like` | `LikeController@likeFragment` |
| DELETE | `/api/fragments/{fragment}/like` | `LikeController@unlikeFragment` |
| GET | `/api/likes` | `LikeController@index` |

### 2.7 Recommendations

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/recommendations` | `RecommendationController@getRecommendations` |
| GET | `/api/tracks/{track}/similar` | `RecommendationController@findSimilarTracks` |
| GET | `/api/fragments/{fragment}/similar` | `RecommendationController@findSimilarFragments` |

### 2.8 Preference Model

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/preference-model/train` | `PreferenceModelController@train` |
| GET | `/api/preference-model/status` | `PreferenceModelController@status` |

### 2.9 Export

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/export/csv` | `ExportController@exportCsv` |

### 2.10 Playlists

```
GET    /api/playlists
POST   /api/playlists
GET    /api/playlists/{playlist}
PUT    /api/playlists/{playlist}
DELETE /api/playlists/{playlist}
POST   /api/playlists/{playlist}/tracks                 PlaylistController@addTrack
DELETE /api/playlists/{playlist}/tracks/{track}         PlaylistController@removeTrack
PUT    /api/playlists/{playlist}/tracks/reorder         PlaylistController@reorderTracks
POST   /api/playlists/generate                          PlaylistController@generate
```

### 2.11 Track Ratings (5★)

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/tracks/{track}/rating` | `TrackRatingController@store` |
| GET | `/api/tracks/{track}/rating` | `TrackRatingController@show` |
| DELETE | `/api/tracks/{track}/rating` | `TrackRatingController@destroy` |
| GET | `/api/tracks/{track}/rating/history` | `TrackRatingController@history` |

### 2.12 Fragment Rating & Comments

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/tracks/{track}/fragments/rate` | `FragmentController@rate` |
| PUT | `/api/fragments/{fragment}/rating` | `FragmentController@updateRating` |
| DELETE | `/api/fragments/{fragment}/rating` | `FragmentController@destroyRating` |
| POST | `/api/fragments/{fragment}/comment` | `FragmentController@addComment` |
| PUT | `/api/fragments/{fragment}/comment` | `FragmentController@updateComment` |
| DELETE | `/api/fragments/{fragment}/comment` | `FragmentController@destroyComment` |

### 2.13 User Track Preferences

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/tracks/{track}/preferences` | `UserTrackPreferenceController@show` |
| PUT | `/api/tracks/{track}/preferences` | `UserTrackPreferenceController@update` |
| DELETE | `/api/tracks/{track}/preferences` | `UserTrackPreferenceController@destroy` |

### 2.14 Listening History & Skip Tracking

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/tracks/{track}/play` | `ListeningHistoryController@recordPlay` |
| POST | `/api/tracks/{track}/skip` | `ListeningHistoryController@recordSkip` |
| GET | `/api/tracks/{track}/skip-stats` | `ListeningHistoryController@getSkipStats` |
| GET | `/api/listening-history` | `ListeningHistoryController@getRecentHistory` |
| PUT | `/api/listening-history/{history}/complete` | `ListeningHistoryController@recordComplete` |

### 2.15 Smart Radio + персональные станции

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/radio/enable` | `SmartRadioController@enable` |
| POST | `/api/radio/disable` | `SmartRadioController@disable` |
| GET | `/api/radio/status` | `SmartRadioController@status` |
| GET | `/api/radio/recommendations` | `SmartRadioController@recommendations` |
| POST | `/api/radio/feedback` | `SmartRadioController@feedback` |
| POST | `/api/radio/access-token` | `RadioAccessController@issueAccessToken` |
| GET | `/api/account/personal-radio` | `RadioAccessController@personalSettings` |
| PUT | `/api/account/personal-radio` | `RadioAccessController@updatePersonalSettings` |

### 2.16 Friends & Direct Chat

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/friends/state` | `FriendController@state` |
| GET | `/api/friends/search` | `FriendController@search` |
| POST | `/api/friends/requests` | `FriendController@sendRequest` |
| POST | `/api/friends/requests/{friendship}/respond` | `FriendController@respond` |
| GET | `/api/friends/conversations/{user}` | `FriendController@conversation` |
| POST | `/api/friends/messages` | `FriendController@sendMessage` |
| POST | `/api/friends/conversations/{user}/read` | `FriendController@markConversationRead` |

### 2.17 AI DJ

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/dj/compatible/{trackId}` | `AIDJController@compatible` |

### 2.18 AI Music Chat (Ollama)

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| POST | `/api/chat/search` | `ChatController@search` |
| GET | `/api/chat/track/{id}` | `ChatController@getTrack` |
| GET | `/api/chat/recommend` | `ChatController@recommend` |
| POST | `/api/chat/ai` | `ChatController@aiChat` |
| GET | `/api/chat/ai/health` | `ChatController@aiHealth` |
| POST | `/api/debug/frontend-logs` | `DebugController@frontendLogs` |

### 2.19 Pattern Search (мелодические паттерны)

```
GET    /api/patterns                            PatternController@index
POST   /api/patterns                            PatternController@store
GET    /api/patterns/{pattern}                  PatternController@show
PUT    /api/patterns/{pattern}                  PatternController@update
DELETE /api/patterns/{pattern}                  PatternController@destroy
POST   /api/patterns/{pattern}/examples         PatternController@addExample
DELETE /api/patterns/{pattern}/examples/{example} PatternController@removeExample
POST   /api/patterns/{pattern}/search           PatternController@search
GET    /api/patterns/{pattern}/matches          PatternController@matches
PUT    /api/patterns/{pattern}/matches/{match}/confirm PatternController@confirmMatch
POST   /api/tracks/{track}/extract-chroma       PatternController@extractChroma
POST   /api/tracks/{track}/extract-pitch        PatternController@extractPitch
```

### 2.20 Segment Labeling (для ML-обучения)

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/tracks/{track}/segments` | `PatternController@getTrackSegments` |
| POST | `/api/tracks/{track}/analyze-segments` | `PatternController@analyzeSegments` |
| GET | `/api/tracks/{track}/analyze-segments/status` | `PatternController@getAnalysisStatus` |
| POST | `/api/segments/{segment}/rate` | `PatternController@rateSegment` |

> **Очередь:** `POST /analyze-segments` всегда ставит задачу типа `analyze_segments` с `priority=manual` и сразу промоутит её над любым `analyze_segments` без `priority=manual` через `MLQueueService::prioritizeQueuedManualSegmentAnalysisTasks()`. Это означает: ручной запрос пользователя из UI пройдёт раньше фонового backlog'а, который накапливает планировщик через `segments:enqueue-missing`. Полная схема — [`ARCHITECTURE.md` §5 — Priority queueing](./ARCHITECTURE.md#priority-queueing-priority_task_types--pushpriority).

## 3. Админские (`auth.token` + `role:admin`)

### 3.1 Очередь / диагностика

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/queue-status/stream` | `QueueStatusController@stream` (Server-Sent Events) |
| GET | `/api/queue-status/failed` | `QueueStatusController@failedTasks` |
| POST | `/api/queue-status/retry/{taskId}` | `QueueStatusController@retryTask` |
| POST | `/api/queue-status/retry-all` | `QueueStatusController@retryAllFailed` |
| DELETE | `/api/queue-status/task/{taskId}` | `QueueStatusController@deleteTask` |
| DELETE | `/api/queue-status/clear-failed` | `QueueStatusController@clearAllFailed` |

### 3.2 Admin / Debug

| Метод | URL | Controller@action |
|-------|-----|--------------------|
| GET | `/api/admin/overview` | `AdminController@overview` |
| POST | `/api/admin/training-queue/repair` | `AdminController@repairTrainingQueue` |
| POST | `/api/admin/users/{user}/preference-model/retrain` | `AdminController@retrainUserModel` |
| POST | `/api/admin/training-tasks/{taskId}/retry` | `AdminController@retryTrainingTask` |
| POST | `/api/admin/training-tasks/{taskId}/cancel` | `AdminController@cancelTrainingTask` |
| GET | `/api/debug/tracks` | `DebugController@trackStats` |
| GET | `/api/debug/system` | `DebugController@systemStats` |
| GET | `/api/debug/tracks/{trackId}` | `DebugController@trackDetail` |

## 4. Web-маршруты (не /api)

Из `routes/web.php`:

| Метод | URL | Поведение |
|-------|-----|-----------|
| GET | `/radio/{slug}` | Рендерит Blade-шаблон `public-radio` (фуллскрин public-radio плеер) |
| GET | `/embed/radio/{slug}` | То же, но `embed: true` (без app-chrome, для iframe) |
| GET | `/{any?}` | Catch-all → `app.blade.php` (контейнер для Vue-роутера) |

## 5. WebSocket (вне Laravel)

WebSocket-сервер запускается в контейнере `node` (`scripts/public-radio-ws-server.mjs`, `radio:ws` в `package.json`). Слушает порт `8080` внутри Docker-сети, не экспонируется наружу напрямую — фронт должен ходить через прокси или через тот же Vite-контейнер. Origin для общения с Laravel — `RADIO_APP_ORIGIN=http://app` (`docker-compose.yml:351`).

Полный реестр сообщений (3 эндпоинта `/ws/radio-listeners`, `/ws/radio-private`, `/ws/friends`, формы `snapshot`/`heartbeat`/`update`/`chat:*`/`presence:*`, Redis-ключи `radio:listeners:*`, `radio:chat:*` и список Laravel REST-эндпоинтов, которые WS-сервер дёргает на каждый коннект и каждые ~5 секунд) — см. [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md).

## 6. Подробные DTO тяжёлых контроллеров

Этот раздел описывает точные формы JSON-ответов для пяти самых нетривиальных контроллеров. Все ключи указаны так, как их пишет код (не «ожидание», а реальный `response()->json([...])`). Объект `track` — это `Track::toArray()` (полный набор колонок таблицы `tracks` плюс accessors модели), `fragment` — `Fragment::toArray()` и т. д. Если форма различается между success/error, оба варианта приведены.

> **Конвенция здесь и ниже:** `<…>` — комментарий о значении или возможных ветках; `<task>` — полное представление `App\Models\QueueTask` (или его подмножество, специально перечисленное), `<track>` — `App\Models\Track`-сериализация. HTTP-код всегда `200` если не написано иное.

### 6.1 `StemTuningLabController`

Источник: `app/Http/Controllers/StemTuningLabController.php`.

**`GET /api/tracks/{track}/tuning-lab` — `show`**

```json
{
  "track": { "<полный Track::toArray()>": "..." },
  "lab": { "<сессия из StemTuningLabRegistry::sessionForTrack()>": "..." }
}
```

Ошибка 404 — когда стемов ещё нет:

```json
{
  "error": "Not found",
  "message": "No stem sources were found for this track yet. Build stems first, then open the tuning lab.",
  "buildable": true
}
```

**`POST /api/tracks/{track}/tuning-lab/render` (HTTP 202)** и **`/build-stems`** возвращают одинаковую форму:

```json
{
  "task_id": 1234,
  "status": "pending",
  "task_type": "stem_tuning_render",       // или "stem_source_build"
  "message": "Stem tuning render queued"   // или "Stem source build queued"
}
```

> **Под капотом:** `StemTuningRenderService::queue()` / `StemSourceBuildService::queue()` дёргают `MLQueueService::pushStemTuningRender()` / `pushStemSourceBuild()`, которые внутри используют `pushPriority` (см. `app/Services/MLQueueService.php:182-272` и [`ML_WORKER.md` §3.2](./ML_WORKER.md#32-приоритезация-со-стороны-laravel)). Возвращаемые `queue_length`/`promoted` логируются на стороне Laravel и в API не возвращаются.

Ошибки этих эндпоинтов — HTTP 422:

```json
{ "error": "Render unavailable", "message": "<RuntimeException::getMessage()>" }
```

**`GET /api/tracks/{track}/tuning-lab/render-tasks/{taskId}` — `renderStatus`**

Состояние читается из таблицы `queue_tasks` и из Redis-ключа `stem_tuning_render_status:{taskId}`. Логи нормализуются в строгий формат.

```json
{
  "task_id": 1234,
  "status": "processing",                    // pending | processing | completed | failed | cancelled
  "progress": 42,                            // 0..100; для completed/cancelled принудительно 100, если воркер ничего не записал
  "message": "Mixing stems...",              // дефолты: completed → "Container render completed", cancelled → "Container render cancelled", иначе "Waiting for worker status"
  "error_message": null,                     // или строка из QueueTask::error_message
  "completed_at": "2026-04-19T12:34:56+00:00", // или null
  "logs": [
    {
      "timestamp": "2026-04-19T12:34:51+00:00",
      "level": "info",
      "progress": 30,
      "message": "Loading stem sources",
      "stage": "load"                        // или null
    }
  ],
  "payload": { "track_id": 15, "render_id": "render_abcdef" }
}
```

Ошибка 404: `{ "error": "Not found", "message": "Stem tuning render task not found" }`.

**`POST /api/tracks/{track}/tuning-lab/render-tasks/{taskId}/cancel` — `cancelRender`**

Отдаёт ту же форму, что `renderStatus`, плюс **дополнительное поле** `queue_cleanup` (произвольная структура из `StemTuningRenderService::cancel()`):

```json
{
  "task_id": 1234,
  "status": "cancelled",
  "progress": 100,
  "message": "Container render cancelled",
  "error_message": null,
  "completed_at": "...",
  "logs": [ /* … */ ],
  "queue_cleanup": { "<service-specific>": "..." },
  "payload": { "track_id": 15, "render_id": "render_abcdef" }
}
```

**`GET /api/tracks/{track}/tuning-lab/build-stem-tasks/{taskId}` — `buildStatus`**

Минимальная форма (logs/progress не возвращаются; есть только статус):

```json
{
  "task_id": 1234,
  "status": "completed",
  "error_message": null,
  "completed_at": "2026-04-19T12:34:56+00:00",
  "payload": { "track_id": 15, "build_id": "build_abcdef" }
}
```

**`GET /api/tracks/{track}/tuning-lab/sources/{sourceKey}/waveform` — `sourceWaveform`**

Тело — то, что вернул `StemTuningWaveformService::getWaveformPayload()` (как правило, набор массивов peaks). Заголовок `Cache-Control: public, max-age=86400`. Ошибки:

- 404 `{ "error": "Not found", "message": "Tuning source not found" }`
- 500 `{ "error": "Waveform unavailable", "message": "<RuntimeException::getMessage()>" }`

**`GET /api/tracks/{track}/tuning-lab/sources/{sourceKey}/stream` — `sourceStream`** возвращает **бинарный аудио-поток**: либо через `X-Accel-Redirect` (если файл лежит под `storage/app` или `lab_artifacts`), либо как обычный stream с поддержкой `Range` (HTTP 206 + `Content-Range`).

### 6.2 `ChatController`

Источник: `app/Http/Controllers/ChatController.php` + `app/Services/ChatSearchService.php` + `app/Services/LLMChatService.php`.

**`POST /api/chat/search`**

Делегируется в `ChatSearchService::searchSimilarSegments`, который пушит задачу в Redis-список `ml_worker:tasks` (`type: segment_search`) и ждёт результат в `chat_search_result:{taskId}` до 60 с.

> **Важно:** хотя `segment_search` входит в `MLQueueService::PRIORITY_TASK_TYPES`, `ChatSearchService` ставит её **обычным `RPUSH`** (`app/Services/ChatSearchService.php:84`), а не через `MLQueueService::pushPriority`. Это значит, что без параллельной постановки через `pushPriority` чатовый поиск по сегментам будет ждать вместе со всем сегментным backlog'ом. Аналогично `recommend` ниже ставит `embedding_recommendations` через `MLQueueService::push` (`ChatController.php:185`) — без авто-промоутинга.

Успешный ответ (форма результата приходит из ML worker'а через Redis):

```json
{
  "query": { "track_id": 123, "start_time": 29.0, "end_time": 35.0 },
  "results": [ /* массив сегментов из ML worker'а */ ],
  "total": 7
}
```

Если воркер не ответил за 60 с — тот же контракт, но `results: []`, `total: 0`, `error: "Search timeout - ML worker may be busy"`.

Ошибки контроллера (HTTP 422 / 404 / 500) имеют единый шаблон:

```json
{ "error": "Validation failed", "message": "track_id is required and must be a number", "code": "INVALID_TRACK_ID" }
```

`code` — один из: `INVALID_TRACK_ID`, `INVALID_START_TIME`, `INVALID_END_TIME`, `NEGATIVE_TIME`, `INVALID_RANGE`, `TRACK_NOT_FOUND` (404), `VALIDATION_ERROR`, `SEARCH_ERROR` (500).

**`GET /api/chat/track/{id}` — `getTrack`**

```json
{ "id": 123, "title": "...", "artist": "...", "duration": 213.4, "file_path": "audio/uploads/123.mp3" }
```

Или 404 `{ "error": "Not found", "message": "Track with ID 123 not found" }`.

**`GET /api/chat/recommend` — `recommend`**

Сначала пушит ML-задачу `embedding_recommendations` с `task_id = uniqid('chat_recommend_', true)` и ждёт результат в `chat_recommend_result:{taskId}` до 30 с. Если ответ есть — возвращает **в точности** то, что лежало в Redis (форма формируется ML-воркером).

При таймауте — fallback `getSimpleRecommendations(int $limit)`. Три ветки:

1. **У пользователя нет лайков и нет рейтингов 4★+, но есть `user_track_scores`:**

```json
{
  "recommendations": [
    {
      "track":      { /* Track */ },
      "score":      87.3,
      "reasons":    ["Похож по звуку на ваши лайки и сегменты"],
      "explanation":"Похож по звуку на ваши лайки и сегменты"
    }
  ],
  "total": 1,
  "preference_summary": {
    "top_genres": [], "top_moods": [], "avg_bpm": null, "tracks_analyzed": 124
  },
  "has_preferences": true,
  "message": null
}
```

2. **Совсем нет данных:**

```json
{
  "recommendations": [],
  "message": "Недостаточно данных о ваших предпочтениях. Поставьте лайки или оценки трекам.",
  "has_preferences": false
}
```

3. **Есть лайки/рейтинги:** обычная форма (как №1), но `preference_summary.top_genres`/`top_moods`/`avg_bpm` заполнены, `tracks_analyzed` = размер preference-профиля. Если кандидаты исчерпаны — `recommendations: []`, `message: "Вы уже прослушали все доступные треки!"`, `has_preferences: true`.

**`POST /api/chat/ai` — `aiChat`**

Делегируется в `LLMChatService::chat()`. Контроллер просто прокидывает результат. Возможные `type`-значения (определяются по action из ответа Ollama):

| `type` | Когда | Содержимое `tracks` |
|--------|-------|---------------------|
| `chat` | свободный диалог / неудачный JSON-парсинг | `[]` |
| `search` | LLM выбрал `SEARCH` | результат поиска |
| `recommendations` | `RECOMMEND` | подобранные треки |
| `similar` | `SIMILAR` | top-K по эмбеддингу |
| `stats` | `STATS` | `[]` |
| `info` | `INFO` | `[track]` или `[]` |
| `like` / `dislike` | `LIKE` / `DISLIKE` | `[track]` или `[]` |
| `rate` | `RATE` | `[track]` или `[]` |
| `learn` | `LEARN` | расширенный список |
| `rate_fragment` | `RATE_FRAGMENT` | `[track]` или `[]` |
| `fragments` | `FRAGMENTS` | `[track]` или `[]` |
| `import` | `IMPORT` (YouTube) | `[]` |

Минимальный shape:

```json
{ "type": "chat", "message": "<строка для пользователя>", "tracks": [] }
```

Ошибка контроллера (HTTP 500): `{ "error": "AI chat failed", "message": "<exception>" }`. Валидация: `{ "error": "Message is required" }` (HTTP 422).

**`GET /api/chat/ai/health` — `aiHealth`**

Делегируется в `LLMChatService::checkHealth()`. HTTP 200 при `status === "ok"`, иначе HTTP 503.

```json
// 200 OK
{
  "status": "ok",
  "ollama_url": "http://ollama:11434",
  "model": "llama3:8b",
  "model_loaded": true,
  "available_models": ["llama3:8b", "qwen2:7b"]
}

// 503
{ "status": "error", "message": "Ollama недоступен" }
```

### 6.3 `PatternController`

Источник: `app/Http/Controllers/PatternController.php` + `app/Services/TrackFeedbackSnapshotService.php`.

**Pattern CRUD (`index`, `store`, `show`, `update`, `destroy`)**

- `index` → массив `Pattern`-моделей с `withCount(['examples', 'matches'])` (поля `examples_count`, `matches_count`).
- `store` (HTTP 201) → одна `Pattern`-модель.
- `show` → одна `Pattern`-модель с `examples.track` и `matches.track` через eager-load.
- `update` → одна `Pattern`-модель.
- `destroy` → `{ "message": "Pattern deleted" }`.

**`POST /api/patterns/{pattern}/examples` — `addExample` (HTTP 201)** возвращает `PatternExample` с подгруженным `track`.

**`DELETE /api/patterns/{pattern}/examples/{example}` — `removeExample`**: `{ "message": "Example removed" }` или 400 `{ "error": "Example does not belong to this pattern" }`.

**`POST /api/patterns/{pattern}/search`, `POST /api/tracks/{track}/extract-chroma`, `POST /api/tracks/{track}/extract-pitch`** — все три кладут задачу в `MLQueueService::pushTask(...)` и возвращают:

```json
{ "message": "Search started",            "task_id": "<строка task_id из MLQueueService>" }
{ "message": "Chroma extraction started", "task_id": "..." }
{ "message": "Pitch extraction started",  "task_id": "..." }
```

`search` без позитивных примеров → 400 `{ "error": "Pattern needs at least one positive example" }`. `extractChroma`/`extractPitch` для не-проанализированного трека → 400 `{ "error": "Track must be analyzed first" }`.

**`GET /api/patterns/{pattern}/matches`** → массив `PatternMatch`-моделей с подгруженным `track`, отсортирован по `score DESC`.

**`PUT /api/patterns/{pattern}/matches/{match}/confirm`** → обновлённый `PatternMatch`. Если `confirmed=true` и такого example ещё нет — он создаётся (`is_positive=true`). 400 `{ "error": "Match does not belong to this pattern" }`.

**`GET /api/tracks/{track}/segments` — `getTrackSegments`**

Делегируется в `TrackFeedbackSnapshotService::buildForUser()`. Полная форма:

```json
{
  "track_id": 15,
  "fragments": [
    {
      "id": 87, "track_id": 15,
      "start_time": 12.5, "end_time": 18.0,
      "is_liked": true,                  // bool | null (null = не оценён)
      "name": null,                      // зарезервировано, всегда null
      "comment": "Любимый дроп"          // или null
    }
  ],
  "segments": [
    {
      "id": 4001,
      "start": 0.0, "end": 8.5,
      "type": "intro",                   // ключ из SegmentDisplayLabel (см. App\Support\SegmentDisplayLabel)
      "display_label": "Intro",          // человекочитаемая метка
      "importance": 1,                   // 0..3
      "is_liked": true,                  // bool | null (rating >= 3 → true)
      "rating": 4.0,                     // 0..5 | null
      "pattern_id": 7,
      "pattern_name": "Intro detector"
    }
  ],
  "track_rating": 4,                     // целое 1..5 (приоритет — explicit), либо null
  "explicit_track_rating": 4,            // явный TrackRating пользователя (1..5 | null)
  "computed_track_rating": 3.7,          // взвешенное по сегментам (0..5 | null)
  "preference_score": 0.81,              // ML-скор пользователя (0..1 | null)
  "sentiment": "positive"                // "positive" | "neutral" | "negative" | null
}
```

Без авторизации — 401 `{ "error": "Unauthorized", "message": "Authentication is required" }`.

**`POST /api/tracks/{track}/analyze-segments` — `analyzeSegments`**

Логика дедупликации тяжёлая: Cache-lock на ключе вокруг постановки + проверка статуса в Redis (`segment_analysis_status:{trackId}`) + `MLQueueService::promoteAndDeduplicateQueuedSegmentAnalysisTask` для ручного промоутинга и склейки дублей. Если задача новая — вызывается `pushPrioritySegmentAnalysis`, которая после `RPUSH` сразу делает `prioritizeQueuedManualSegmentAnalysisTasks`. Ответы:

```json
// Новая задача поставлена в очередь
{ "message": "Segment analysis started",      "track_id": 15, "task_id": "ml_task_…", "deduplicated": false }

// Уже есть активная (HTTP 202)
{ "message": "Segment analysis already queued.","track_id": 15, "task_id": "ml_task_…", "status": "processing", "deduplicated": true }

// Лок не взяли и активной нет (HTTP 409)
{ "error": "Segment analysis enqueue is busy", "message": "Could not reserve the segment analysis slot. Try again." }

// Трек не анализирован (HTTP 400)
{ "error": "Track must be analyzed first" }
```

Допустимые `status`: `queued`, `processing`.

**`GET /api/tracks/{track}/analyze-segments/status` — `getAnalysisStatus`**

```json
// Если в Redis ничего нет
{ "status": "idle", "track_id": 15 }

// Иначе — то, что лежало в Redis (структура управляется MLQueueService)
{ "status": "processing", "task_id": "...", "track_id": 15, "progress": 30, "message": "..." }
```

**`POST /api/segments/{segment}/rate` — `rateSegment`**

```json
{
  "message": "Segment rated",
  "segment": {
    "id": 4001, "pattern_example_id": 4001, "track_id": 15,
    "start_time": 0.0, "end_time": 8.5,
    "rating": 4.0,            // 0..5
    "is_positive": true       // rating >= 3
  },
  "track_rating": 4,          // эффективный (после синхронизации) 1..5 | null
  "computed_track_rating": 3.6, // взвешенное (0..5 | null)
  "sentiment": "positive",    // см. TrackRatingPropagationService::sentimentForRating
  "feedback": null
}
```

404 — `{ "error": "Segment not found" }`.

### 6.4 `RecommendationController`

Источник: `app/Http/Controllers/RecommendationController.php` + `app/Services/UserPreferenceScoreService.php` (через DI).

**`GET /api/recommendations` — `getRecommendations`**

Если у пользователя ещё не обучена модель (`PreferenceModelStatus` без `model_path`/`trained_at`) — HTTP 400:

```json
{
  "error": "Model not trained",
  "message": "Модель ещё не обучена. Пожалуйста, обучите модель или используйте поиск похожих треков.",
  "suggest_training": true
}
```

Если модель есть — фильтруем кандидатов (исключая уже оценённые / с оценёнными сегментами треки), создаём в БД запись `queue_tasks` (тип `recommendation`), берём оценки из `UserPreferenceScoreService` и собираем рекомендации с диверсификацией по жанру:

```json
{
  "recommendations": [
    {
      "track":  { /* Track::toArray() */ },
      "score":  0.8721,            // 4 знака после запятой
      "reason": "preference_model" // константа
    }
  ],
  "ratings":  { "15": 4, "23": 3 }, // map<track_id, rating> для текущего юзера
  "segments": {
    "15": [
      {
        "id": 4001,
        "start": 0.0, "end": 8.5,
        "type": "intro",
        "display_label": "Intro",
        "importance": 1,
        "rating": 4.0
      }
    ]
  },
  "total": 20,
  "model_status": "trained"
}
```

Если кандидатов нет — `{ "recommendations": [], "message": "No tracks available for recommendations" }`.

Параметры query: `genre`, `mood`, `bpm_min`+`bpm_max`, `limit` (clamp 0..100, default 20). Если `genre` указан — диверсификация **отключается** (кейс «один жанр»).

**`GET /api/tracks/{trackId}/similar` — `findSimilarTracks`**

```json
{
  "similar_tracks": [
    {
      "track_id":  42,
      "track":     { /* Track::toArray() */ },
      "similarity": 0.72         // 2 знака; clamp 0.10..0.95
    }
  ],
  "query_track": { /* Track::toArray() */ },
  "task_id":     null            // зарезервировано на случай асинхронного режима
}
```

Параметр query: `k` (default 10, max 50), плюс те же `genre`/`mood`/`bpm_min`/`bpm_max`. Текущий алгоритм — синхронный весовой scoring (genre 30 % / bpm 25 % / key 20 % / mood 25 % + ±5 % шума), без обращения к ML worker'у. Ошибки: 404 (`Track not found`), 400 (`Track has not been analyzed yet`).

**`GET /api/fragments/{fragmentId}/similar` — `findSimilarFragments`**

```json
{
  "similar_fragments": [
    {
      "fragment_id": 87,
      "fragment":    { /* Fragment::toArray() с подгруженным track */ },
      "similarity":  0.68        // 2 знака; clamp 0.0..0.99
    }
  ],
  "query_fragment": { /* Fragment::toArray() с track */ }
}
```

Ошибки: 404, 400 (как у треков).

### 6.5 `SmartRadioController`

Источник: `app/Http/Controllers/SmartRadioController.php` + `app/Services/SmartRadioService.php`.

**`POST /api/radio/enable`** — body `{ "seed_track_id": 15 }`:

```json
{
  "message":       "Smart Radio enabled with seed track: <track.title>",
  "enabled":       true,
  "seed_track_id": 15
}
```

Ошибки: 400 (валидация), 404 (`Seed track not found`), 500 (`Failed to enable Smart Radio: <e>`).

**`POST /api/radio/disable`**:

```json
{ "message": "Smart Radio disabled", "enabled": false }
```

**`GET /api/radio/status`**:

```json
{
  "enabled":       true,
  "seed_track_id": 15,
  "enabled_at":    "2026-04-19T12:00:00+00:00",
  "disabled_at":   null
}
```

(Если радио ни разу не включалось — все четыре поля могут быть `false` / `null`.)

**`GET /api/radio/recommendations?count=10`** (count clamp 1..50, default 10) — вызывает `SmartRadioService::getPreferenceBasedRecommendations()`:

```json
{
  "count": 5,
  "tracks": [
    {
      "id": 42, "title": "...", "artist": "...", "album": "...",
      "duration": 213.4, "genre": "Drum & Bass", "mood": "energetic",
      "bpm": 174.0, "key": "8A",
      "analysis_status": "completed",
      "recommendation_score": 0.83   // float | null
    }
  ]
}
```

Поле `recommendation_score` — внедряется сервисом прямо в модель `Track` как «виртуальный» атрибут (см. `SmartRadioService::getPreferenceBasedRecommendations`). Ошибки: 400 (валидация count), 500 (`Failed to get recommendations: <e>`).

**`POST /api/radio/feedback`** — body `{ "track_id": 15, "feedback_type": "more"|"less" }`. HTTP 201:

```json
{
  "message":           "Feedback recorded successfully",
  "track_id":          15,
  "feedback_type":     "more",
  "weight_adjustment": 0.2,       // SmartRadioService::MORE_WEIGHT_ADJUSTMENT / LESS_WEIGHT_ADJUSTMENT
  "new_weight":        0.6        // накопленный вес для (user, track)
}
```

Параллельно создаётся запись в `radio_feedback`. Ошибки: 400 (валидация), 404 (`Track not found`), 500 (`Failed to record feedback: <e>`).

## 7. Примеры

```bash
# Логин
curl -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"admin@example.local","password":"change-me-please"}'

# Список треков
curl http://localhost:8000/api/tracks \
     -H "Authorization: Bearer $TOKEN"

# Импорт с YouTube
curl -X POST http://localhost:8000/api/tracks/youtube \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"url":"https://youtu.be/dQw4w9WgXcQ"}'

# Поставить лайк фрагменту
curl -X POST http://localhost:8000/api/fragments/123/like \
     -H "Authorization: Bearer $TOKEN"

# SSE — поток статусов очереди (admin)
curl -N http://localhost:8000/api/queue-status/stream \
     -H "Authorization: Bearer $ADMIN_TOKEN"
```

## 8. Не задокументированное в коде, но видимое

- **Лимит размера загружаемого файла:** в `docker/app/Dockerfile` PHP-FPM настроен на `upload_max_filesize=100M`, `post_max_size=100M`. README исторически упоминает «up to 100MB» — это и есть источник.
- **Поддерживаемые аудио-форматы:** определяется `TrackController::store` + `getID3` (на бэкенде) и `audioread`/`librosa` (в воркере). Минимум — MP3, WAV; реальный список нужно сверять с валидаторами `TrackController`.
