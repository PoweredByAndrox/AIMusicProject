# Public Radio — WebSocket-протокол

Источник истины для этого документа — `scripts/public-radio-ws-server.mjs`. WS-сервер запускается командой `node scripts/public-radio-ws-server.mjs` (npm-скрипт `radio:ws` в `package.json`) внутри сервиса `node` из `docker-compose.yml`. Все детали ниже сверены с кодом, а не предполагаются.

Сопутствующие файлы:

- `resources/js/public-radio.ts` — клиентские типы (`StationDescriptor`, `StationTrack`, `StationSnapshot` и т. д.).
- `app/Http/Controllers/PublicRadioController.php` + `app/Services/PublicRadioStationService.php` — Laravel-эндпоинты, которые WS-сервер дёргает по HTTP.
- `app/Http/Controllers/RadioAccessController.php` — выдача `access_token` для приватных/персональных станций.
- `routes/web.php` — Blade-маршруты `/radio/{slug}` и `/embed/radio/{slug}`, которые загружают полностраничный плеер.

> **Важно.** WS-сервер **не** использует Laravel Broadcasting / Pusher / Echo. Вся «реактивность» делается через циклический `fetch` к Laravel REST API + push в подключённые сокеты. Laravel ничего самостоятельно в WS не пишет.

## 1. Окружение и запуск

| Переменная | Значение по умолчанию | Источник | Назначение |
|------------|----------------------|----------|------------|
| `RADIO_WS_PORT` | `8080` | `scripts/public-radio-ws-server.mjs:7` | Порт внутри контейнера `node`. |
| `RADIO_APP_ORIGIN` | `http://app` | `:8` (значение из `docker-compose.yml:351`) | Origin Laravel-приложения для серверных fetch-вызовов. |
| `REDIS_HOST` | `redis` | `:9` | Хост Redis. |
| `REDIS_PORT` | `6379` | `:10` | Порт Redis. |
| `REDIS_PASSWORD` | `''` | `:11` | Опциональный пароль. |
| `REDIS_PREFIX` | `ai_music_recommendations_database_` | `:12` | Префикс ключей; **должен совпадать** с Laravel-префиксом (см. `config/database.php:redis.options.prefix`). |

Внутренние таймауты и лимиты (захардкожены):

- `fallbackSnapshotIntervalMs = 60_000` — fallback-период перерисовки snapshot, если до конца текущего трека дольше минуты.
- `heartbeatIntervalMs = 10_000` — частота `ping` всем клиентам.
- `staleClientTimeoutMs = 25_000` — после такого простоя клиент считается мёртвым и `terminate`-ится.
- `chatHistoryLimit = 50` — сколько сообщений хранится в Redis-списке чата.
- `chatMessageMaxLength = 400` — макс. длина одного `chat_message` после нормализации.
- `chatHistoryTtlSeconds = 604_800` — TTL ключей чата (7 суток).
- `hlsStorageRoot = /var/www/html/storage/app/public_radio_hls`.
- `waveformStorageRoot = /var/www/html/storage/app/public_radio_waveforms`.

## 2. Эндпоинты (HTTP upgrade)

Один HTTP-сервер слушает порт `RADIO_WS_PORT` и маршрутизирует upgrade-запросы по `pathname`:

| Path | Кто это | Состояние клиента (Map) |
|------|---------|-------------------------|
| `/ws/radio-listeners` | Публичная станция (live-плеер на `/radio/{slug}`). Любой посетитель, без обязательной аутентификации. | `publicClients` |
| `/ws/radio-private` | Приватный «overlay» для конкретного трека (приглашённые слушатели). | `privateClients` |
| `/ws/friends` | Лента друзей и личных сообщений. Только аутентифицированные пользователи. | `friendClients` |

Любой другой `pathname` → `socket.destroy()` (HTTP 426 на не-WS запросе: `'Upgrade Required'`).

Query-параметры начального запроса (одинаковые для всех эндпоинтов):

| Параметр | Назначение |
|----------|------------|
| `station` | (public/private) slug станции; нормализуется через `normalizeStationSlug` (lowercase + trim). |
| `clientId` | Стабильный ID клиента; если пусто, генерируется `randomUUID()`. **Важно:** при коллизии `(station, clientId)` старое соединение принудительно закрывается с кодом `4001 'superseded'`. |
| `accessToken` | Bearer-токен пользователя. Для персональных станций (`personal-{userId}`) — обязателен. Для friends — обязателен. |

## 3. Внутреннее состояние

| Map / Set | Содержимое |
|-----------|------------|
| `publicClients: Map<WebSocket, State>` | Состояние публичного клиента: `{id, stationSlug, clientId, listening, accessToken, viewer, lastSeenAt, isAlive}`. |
| `privateClients: Map<WebSocket, State>` | Состояние приватного клиента: `{id, stationSlug, clientId, token, lastSeenAt, isAlive}`. |
| `friendClients: Map<WebSocket, State>` | Состояние friends-клиента: `{id, accessToken, user, lastSeenAt, isAlive}`. |
| `stationClients: Map<slug, Set<WebSocket>>` | Кто на каждой станции слушает (для broadcast'а snapshot/listener_count/chat). |
| `stationLoops: Map<slug, Timeout>` | Таймер периодического опроса Laravel для каждой активной станции. |
| `stationSnapshots: Map<slug, string>` | Сериализованный «сравнимый» snapshot (без `generated_at`/`listener_count`/`viewer`). Если новый snapshot равен старому — клиентам ничего не шлётся (если только `force=true`). |
| `pendingStationSnapshots`, `pendingHlsWarmups`, `pendingWaveformPayloads`, `pendingViewerStates`, `pendingPrivateOverlays` | Защита от «громового стада»: одновременные одинаковые fetch-запросы дедуплицируются. |
| `prewarmedHlsTracks: Set<trackId>` | Какие треки уже прогреты в HLS на диск. |
| `viewerStateCache: Map<accessToken, viewer\|null>` | Кеш `viewer` на уровне токена. |
| `publicClientSockets`, `privateClientSockets`, `friendUserSockets` | Обратные индексы для быстрого закрытия дубликатов и таргетированного broadcast'а. |

## 4. HTTP-зависимости (Laravel API)

WS-сервер ходит **только** на следующие эндпоинты `RADIO_APP_ORIGIN`:

| Метод + URL | Когда | Аутентификация | Где в коде |
|-------------|-------|----------------|------------|
| `GET /api/public/radio/{stationSlug}?access_token=…` | Каждая итерация цикла станции, при подписке, при смене жанров. | `access_token` query (для персональных станций обязателен; для публичных можно опустить). | `fetchStationSnapshot` |
| `GET /api/tracks/{trackId}/hls/master.m3u8` | Прогрев HLS для текущего/следующего трека. | — | `prewarmTrackHls` |
| `GET /api/tracks/{trackId}/waveform` | Если на диске нет `current.json` waveform'а. | — | `fetchWaveformPayload` |
| `GET /api/radio/private-overlay/{trackId}?access_token=…` | По `request_overlay`. | `access_token` query. | `fetchPrivateOverlay` |
| `PUT /api/radio/personal-stations/{stationSlug}/genres?access_token=…` | По `update_personal_station_genres`. | `access_token` query. | `updatePersonalStationGenres` |
| `GET /api/auth/me` | Аутентификация friends-сокета. | `Authorization: Bearer …`. | `fetchAuthenticatedUser` |
| `GET /api/friends/state` | bootstrap / refresh. | Bearer. | `fetchFriendState` |
| `GET /api/friends/search?query=…` | По `search_users`. | Bearer. | `searchFriendUsers` |
| `POST /api/friends/requests` | По `send_friend_request`. | Bearer + body `{user_id}`. | `sendFriendRequest` |
| `POST /api/friends/requests/{friendshipId}/respond` | По `respond_friend_request`. | Bearer + body `{action}`. | `respondToFriendRequest` |
| `GET /api/friends/conversations/{userId}` | По `open_conversation`. | Bearer. | `fetchFriendConversation` |
| `POST /api/friends/messages` | По `send_direct_message`. | Bearer + body `{recipient_id, text, share}`. | `sendFriendDirectMessage` |
| `POST /api/friends/conversations/{userId}/read` | По `open_conversation` / `mark_conversation_read`. | Bearer. | `markFriendConversationRead` |

Все Laravel-эндпоинты, кроме первых трёх, требуют корректный API-токен и пользователя. WS-сервер кладёт его в `Authorization: Bearer …` (для friends) или в `?access_token=…` (для публичного/приватного радио — это контракт `RadioAccessController`, см. `routes/api.php`).

> WS-сервер не пишет ничего напрямую в БД. Все мутации (заявки в друзья, сообщения, смена жанров) идут через Laravel REST API.

## 5. Redis-ключи

WS-сервер использует Redis только для двух вещей:

| Ключ | Тип | Назначение |
|------|-----|-----------|
| `${REDIS_PREFIX}public_radio_listener_count:{stationSlug}` | string (число) | Текущее число слушателей; TTL 120 с. Удаляется, если на станции 0 живых клиентов. |
| `${REDIS_PREFIX}public_radio_chat:{stationSlug}` | list (LPUSH) | История чата; TRIM до `chatHistoryLimit`, EXPIRE `chatHistoryTtlSeconds`. |

Префикс по умолчанию (`ai_music_recommendations_database_`) совпадает с Laravel-префиксом, так что Laravel может читать те же значения через стандартный `Redis::connection()`.

## 6. Жизненный цикл соединения

1. **Heartbeat.** Каждые 10 с (`heartbeatTimer`) сервер шлёт `ping`. Клиент должен отвечать `pong`. Иначе после 25 с молчания соединение `terminate()`-ится.
2. **Дублирующие соединения.** Если клиент с тем же `(stationSlug, clientId)` уже есть — старый сокет закрывается с кодом `4001 'superseded'`. Это защищает от «двух вкладок» на одной станции.
3. **Закрытие.** `socket.on('close')` вызывает `removePublicClient` / `removePrivateClient` / `removeFriendClient`. После удаления публичного клиента — `broadcastCount(stationSlug)` пересчитывает и рассылает новое значение.
4. **Грейсфул shutdown.** На `SIGINT`/`SIGTERM` сервер останавливает все таймеры, `terminate`-ит все сокеты, `redis.quit()` и `process.exit(0)`.

## 7. Протокол `/ws/radio-listeners`

### 7.1 Сообщения клиент → сервер

Все сообщения — JSON с полем `type`. Допустимый набор берётся **только** из `if`-веток в обработчике (`scripts/public-radio-ws-server.mjs:1365…`); остальные `type`-значения игнорируются молча.

| `type` | Поля payload | Что делает сервер |
|--------|--------------|--------------------|
| `presence` | `stationSlug`, `clientId?`, `accessToken?`, `listening?: bool` | Обновляет состояние клиента (heartbeat-маркер `listening=true/false`). Запускает (если нужно) loop станции и шлёт текущий snapshot/`viewer_state`. |
| `subscribe` | то же | То же, что `presence`, но семантически — «подключаюсь к станции». |
| `request_snapshot` | то же | То же, плюс `force=true` — шлёт snapshot, даже если он не менялся. |
| `request_overlay` | `stationSlug`, `trackId`, `accessToken?` | Дёргает `/api/radio/private-overlay/{trackId}` и отвечает `private_overlay` или `private_overlay_unauthorized`. |
| `request_chat_history` | `stationSlug` | Если `viewer` есть — шлёт `chat_history`. Иначе — `chat_unauthorized`. |
| `chat_message` | `stationSlug`, `text` | Нормализует текст (`trim` + одиночные пробелы + `slice(0, 400)`). LPUSH в Redis `public_radio_chat:{slug}`, TRIM до 50, EXPIRE 7 дней. Затем `broadcastChatEntry` всем клиентам станции, у которых есть `viewer` (анонимы чат не получают). |
| `chat_share_segment` | `stationSlug`, `segment: { trackId, segmentId, start, end, label, trackTitle, artist? }` | То же, но с типом `segment_share`. Валидация: все обязательные поля, `end > start`. |
| `update_personal_station_genres` | `stationSlug`, `preferredGenres: string[]` | `PUT /api/radio/personal-stations/{slug}/genres`. Затем `pushStationSnapshot` всем + `broadcastCount`. |
| `request_waveform` | `stationSlug`, `trackId` | `fetchWaveformPayload` → шлёт `waveform_payload`. |
| `prewarm_track` | `trackId` | `prewarmTrackHls(trackId)`. Ничего не отвечает в случае успеха. |

`updatePublicClientState` дополнительно валидирует, что для персональных станций (`/^personal-\d+$/`) **есть** `accessToken`, иначе бросает `Personal station access token is required`. На любой брошенный экспепшн отвечает `error`-сообщением.

### 7.2 Сообщения сервер → клиент

| `type` | Когда | Поля |
|--------|-------|------|
| `error` | На любую брошенную ошибку обработчика. | `{type:'error', stationSlug, message}` |
| `listener_count` | После каждого `broadcastCount` (подключение/отключение, переключение `listening`, смена станции). | `{type:'listener_count', stationSlug, count}` |
| `station_snapshot` | После `broadcastSnapshot`/`pushStationSnapshot`. | `{type:'station_snapshot', stationSlug, snapshot}` — см. раздел 8. |
| `viewer_state` | После `presence`/`subscribe`/`request_snapshot`, если есть `accessToken`. | `{type:'viewer_state', stationSlug, viewer}` (`viewer` — `null` если токен невалиден). |
| `chat_unauthorized` | На `request_chat_history` / `chat_message` / `chat_share_segment` без `viewer`. | `{type:'chat_unauthorized', stationSlug}` |
| `chat_history` | На `request_chat_history`. | `{type:'chat_history', stationSlug, entries: ChatEntry[]}` |
| `chat_message` | На каждое новое сообщение/share — всем клиентам станции, у кого есть `viewer`. | `{type:'chat_message', stationSlug, entry: ChatEntry}` |
| `private_overlay_unauthorized` | На `request_overlay` без токена / при 401-403. | `{type:'private_overlay_unauthorized', stationSlug, trackId}` |
| `private_overlay` | На `request_overlay`. | `{type:'private_overlay', stationSlug, trackId, overlay}` (тело — то, что вернул Laravel). |
| `waveform_payload` | На `request_waveform`. | `{type:'waveform_payload', stationSlug, trackId, waveform}` |

### 7.3 Структура `snapshot`

Snapshot собирается Laravel'ом (`PublicRadioStationService::snapshot`), потом WS-сервер выполняет три модификации:

1. **Удаляет `viewer`** (отдельно отправляется через `viewer_state`).
2. **Подмешивает HLS / waveform URL** через `rewriteSnapshotTrack`:
   - `now_playing.hls_master_url = /storage/public_radio_hls/{trackId}/current/master.m3u8` — **только если** на диске уже есть `master.m3u8` (`hasCurrentHlsPackage`); иначе оставляет исходный URL и асинхронно прогревает.
   - `now_playing.waveform_url = /storage/public_radio_waveforms/{trackId}/current.json`.
   - То же делает для `up_next[]` и `history[]`, проверяя готовность каждого трека отдельно.
3. **Добавляет `now_playing_waveform`** — JSON waveform'а для текущего трека (если ничего нет — `null`).
4. **Перезаписывает `listener_count`** актуальным значением `stationClientCount(slug)`.

Минимальный «контракт» формы snapshot (по `resources/js/public-radio.ts` и `PublicRadioStationService`):

```ts
type StationTrack = {
  track_id: number
  title: string
  artist: string | null
  duration: number              // секунды
  played_at?: string | null     // ISO
  remaining_seconds?: number    // только у now_playing
  hls_master_url?: string       // переписан WS-сервером, если файл готов
  waveform_url?: string         // переписан WS-сервером
  // … прочие поля от Laravel (genre, mood, bpm, …)
}

type StationSnapshot = {
  station: { slug: string, name: string, description?: string, … }
  generated_at: string                      // ISO
  now_playing: StationTrack | null
  now_playing_waveform: object | null       // добавлено WS-сервером
  up_next: StationTrack[]
  history: StationTrack[]
  listener_count: number                    // переписан WS-сервером
}
```

Поле `viewer` приходит от Laravel, но WS-сервер удаляет его до broadcast'а.

### 7.4 Ритм обновления

`scheduleStationLoop` рассчитывает следующую задержку через `nextSnapshotDelay(snapshot)`:

- если `now_playing.remaining_seconds <= 0` или нет — `60_000` мс.
- если `> 60` сек — `60_000` мс.
- иначе — `max(1250, remaining_seconds * 1000 + 500)` мс (через 500 мс после конца трека).

Так Laravel API дёргается редко на «спокойных» интервалах, но плотно перед сменой трека.

`broadcastSnapshot` сравнивает сериализацию (без `generated_at`/`listener_count`/`viewer`) с предыдущей и шлёт только если она изменилась (`force=true` обходит сравнение).

## 8. Протокол `/ws/radio-private`

Используется для приватного контента, который зависит от токена (например — расширенные данные конкретного трека). Состояние — отдельная Map `privateClients`.

### 8.1 Клиент → сервер

| `type` | Поля | Действие |
|--------|------|----------|
| `subscribe_private` | `stationSlug`, `clientId?`, `accessToken?` | Просто обновляет состояние клиента (`updatePrivateClientState`). |
| `request_overlay` | `stationSlug`, `trackId`, `accessToken?` | Обновляет состояние, дёргает `/api/radio/private-overlay/{trackId}` и шлёт `private_overlay` / `private_overlay_unauthorized`. |

Любой другой `type` — игнорируется (нет broadcast snapshot'а и нет чата).

### 8.2 Сервер → клиент

`error`, `private_overlay`, `private_overlay_unauthorized` — поля как у публичного эндпоинта.

## 9. Протокол `/ws/friends`

Аутентификация **обязательна с первого сообщения**. `ensureFriendClientAuth` берёт `accessToken`, дёргает `/api/auth/me` и сохраняет `viewer` в state. Если токен невалиден — отвечает `friends_unauthorized`.

После любых мутаций (новый запрос в друзья, ответ, отправка сообщения, прочтение) сервер делает **`broadcastFriendsBootstrapToUsers([…uniqueUserIds])`**: для каждого затронутого пользователя дёргает `/api/friends/state` и пересылает новое состояние **на все** его открытые friends-сокеты. Это и есть «реальное время» — через push, инициированный в момент мутации.

### 9.1 Клиент → сервер

| `type` | Поля | Действие |
|--------|------|----------|
| `bootstrap` | `accessToken?` | Шлёт `friends_bootstrap` (полное состояние). |
| `search_users` | `query?: string` | Дёргает `/api/friends/search` → `friends_search_results`. |
| `send_friend_request` | `userId` | `POST /api/friends/requests` → broadcast bootstrap для requester+addressee. |
| `respond_friend_request` | `friendshipId`, `action` (`accept` / `decline` / `cancel`) | `POST /api/friends/requests/{id}/respond` → broadcast bootstrap для обеих сторон. |
| `open_conversation` | `userId` | Шлёт `conversation_history`, затем POST `/conversations/{userId}/read` и broadcast bootstrap **только для viewer'а**. |
| `mark_conversation_read` | `userId` | POST `/conversations/{userId}/read` → broadcast bootstrap для viewer'а. |
| `send_direct_message` | `userId` (получатель), `text?`, `share?` | POST `/api/friends/messages`. На успех → `broadcastDirectMessage(messagePayload)` + bootstrap для sender+recipient. |
| `refresh_state` | — | Алиас `bootstrap`. |

Любая ошибка → `{type:'friends_error', message}`. Любой 401/403 → `{type:'friends_unauthorized'}`.

### 9.2 Сервер → клиент

| `type` | Поля |
|--------|------|
| `friends_unauthorized` | `{type:'friends_unauthorized'}` |
| `friends_bootstrap` | `{type:'friends_bootstrap', state: <ответ /api/friends/state>}` |
| `friends_search_results` | `{type:'friends_search_results', query, users}` |
| `conversation_history` | `{type:'conversation_history', userId, conversationWith, messages: DirectMessage[]}` |
| `direct_message` | `{type:'direct_message', message: DirectMessage}` (приходит обеим сторонам диалога). |
| `friends_error` | `{type:'friends_error', message}` |

`DirectMessage` — то, что Laravel возвращает в поле `direct_message` ответа `POST /api/friends/messages` (структура определяется `App\Http\Controllers\FriendController::sendMessage`). Минимум — `{id, sender:{id,name}, recipient:{id,name}, type, body, payload, created_at, read_at}`.

## 10. Чат: формат записи (`ChatEntry`)

```ts
type ChatEntry =
  | {
      id: string                  // UUID
      type: 'message'
      stationSlug: string
      user: { id: number, name: string }
      text: string                // нормализованный, до 400 символов
      createdAt: string           // ISO
    }
  | {
      id: string
      type: 'segment_share'
      stationSlug: string
      user: { id: number, name: string }
      segment: {
        trackId: number
        trackTitle: string        // до 160 символов
        artist: string            // до 160 символов
        segmentId: number
        label: string             // до 80 символов
        start: number             // секунды
        end: number               // > start
      }
      createdAt: string
    }
```

`history` для `chat_history` развёрнут в **chronological** порядке (LPUSH хранит новые сверху, но WS-сервер делает `.reverse()` перед отправкой).

## 11. Интеграция с Laravel

WS-сервер **не использует** Laravel Events / Broadcasting / Pusher. Связь односторонняя через REST + push в момент мутации. Это значит:

- Если кто-то **из самого Laravel** (например, фоновая job'а) меняет состояние (отправляет сообщение, добавляет трек в очередь радио, обновляет жанры) — **WS-сервер об этом не узнает** до следующего polling-цикла (`fetchStationSnapshot`) или до того, как кто-нибудь из подключённых клиентов пришлёт сообщение, дёргающее `broadcastFriendsBootstrap`.
- Это сознательный выбор архитектуры: WS-сервер — тонкий «фасад» над REST, оркестрация лежит в Laravel'е.
- HLS-прогрев делается через `GET /api/tracks/{id}/hls/master.m3u8`, который в Laravel'е (если файлы ещё не нарезаны) ставит `radio-hls-prewarm` job в очередь (см. `docker/app/supervisord.conf` — отдельный воркер на эту очередь).

## 12. Диагностика

| Симптом | Куда смотреть |
|---------|---------------|
| Слушатели «не видят» друг друга | Redis `${prefix}public_radio_listener_count:{slug}` (TTL 120 с). Если ключа нет — на станции никого живого нет. |
| Чат пуст | Redis `${prefix}public_radio_chat:{slug}` (LRANGE 0 49). Проверить TTL (`TTL key`). |
| `private_overlay_unauthorized` для всех | Token не передаётся / устарел. См. `RadioAccessController::issueAccessToken` (выдаёт через `/api/radio/access-token`). |
| Snapshot не обновляется | Проверить `stationLoops` (логировать `nextSnapshotDelay`). Если станция «осталась без слушателей» → `stopStationLoopIfUnused` чистит таймер. |
| Дубль вкладок | При том же `(slug, clientId)` старый сокет закрывается кодом 4001. Клиенту стоит на это реагировать «тихим» reconnect'ом с новым `clientId`. |
| Все клиенты теряются разом | `redis.connect()` в начале файла — без retry. Падение Redis → процесс не стартует. Используйте `restart: unless-stopped` в `docker-compose.yml`. |

Логирование внутри сервера минимальное (только `console.log` про порт). Для продакшена — оборачивать в systemd/supervisor/pm2 и собирать stdout (см. сервис `node` в `docker-compose.yml`).
