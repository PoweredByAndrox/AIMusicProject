# Frontend (Vue 3 + Vite + TypeScript)

Документ описывает фронтенд: точки входа, маршруты, сборку, dev-режим, WebSocket для public-radio. Источники: `package.json`, `vite.config.ts`, `vitest.config.ts`, `tsconfig.json`, `resources/js/router/index.ts`, `routes/web.php`, `docker-compose.yml`.

## 1. Точки входа

В `vite.config.ts` объявлены три entry:

- `resources/css/app.css` — стили (Tailwind через `tailwind.config.js`).
- `resources/js/app.ts` — основной SPA (Vue Router, Pinia, Toastify, Wavesurfer, HLS).
- `resources/js/public-radio.ts` — отдельная маленькая страница для `/radio/{slug}` (фуллскрин-плеер, не использует роутер всего SPA).

Vite собирает в `public/build/` с manifest-файлом, чтобы Laravel мог разрешить хешированные имена через `@vite([...])`-директиву в Blade.

## 2. Зависимости (`package.json`)

Runtime:

| Пакет | Версия | Зачем |
|-------|--------|-------|
| `vue` | `^3.4.0` | Composition API |
| `vue-router` | `^4.2.5` | SPA-роутинг |
| `pinia` | `^2.1.7` + `pinia-plugin-persistedstate` `^3.2.3` | Стейт + persisted в `localStorage` |
| `axios` | `^1.6.0` | HTTP-клиент к `/api/...` |
| `wavesurfer.js` | `^7.6.0` | Визуализация и проигрывание waveform |
| `hls.js` | `^1.5.17` | HLS воспроизведение в Chromium-браузерах |
| `vue3-toastify` | `^0.2.8` | Тосты |
| `ws` | `^8.18.3` | WebSocket-сервер для public-radio (Node.js, не браузер) |
| `redis` | `^4.7.1` | Клиент Redis для public-radio WS-сервера |

Dev:

- `vite ^5.0.0`, `@vitejs/plugin-vue ^5.0.0`, `laravel-vite-plugin ^1.0.0`
- `typescript ^5.3.0`, `vue-tsc ^3.2.6`, `@vue/language-core ^3.2.6`
- `tailwindcss ^3.4.0`, `postcss ^8.4.32`, `autoprefixer ^10.4.16`
- `vitest ^4.0.18`, `@vue/test-utils ^2.4.6`, `happy-dom ^20.4.0`, `fast-check ^4.5.3`
- `@types/node ^20.10.0`

## 3. Скрипты `npm`

```json
"dev": "vite",
"build": "vue-tsc && vite build",
"build:watch": "vite build --watch",
"dev:docker": "sh -c 'npm run radio:ws & npm run dev -- --host 0.0.0.0'",
"radio:ws": "node scripts/public-radio-ws-server.mjs",
"preview": "vite preview",
"type-check": "vue-tsc --noEmit",
"test": "vitest run",
"test:watch": "vitest"
```

В `docker-compose.yml` сервис `node` исполняет именно `npm run dev:docker`, который параллельно поднимает WebSocket-сервер public-radio и Vite dev-server (с `--host 0.0.0.0`, чтобы был доступен снаружи Docker через `${VITE_PORT:-5173}`).

## 4. SPA-роутинг (`resources/js/router/index.ts`)

| Path | Component (lazy) | Meta |
|------|------------------|------|
| `/radio/:slug` | `PublicRadioView.vue` | `hideAppChrome` |
| `/embed/radio/:slug` | `PublicRadioView.vue` | `hideAppChrome`, `embed: true` |
| `/login` | `LoginView.vue` | `guestOnly` |
| `/register` | `RegisterView.vue` | `guestOnly` |
| `/` | `TrackLibrary.vue` | `requiresAuth` |
| `/account` | `AccountView.vue` | `requiresAuth` |
| `/upload` | `TrackUpload.vue` | `requiresAuth` |
| `/track/:id` | `TrackDetail.vue` | `requiresAuth` |
| `/track/:id/tuning-lab` | `StemTuningLabView.vue` | `requiresAuth` |
| `/track/:id/segments` | `SegmentLabeler.vue` | `requiresAuth` |
| `/artist/:slug` | `ArtistDetailView.vue` | `requiresAuth` |
| `/recommendations` | `RecommendationsView.vue` | `requiresAuth` |
| `/playlists` | `PlaylistList.vue` | `requiresAuth` |
| `/playlists/:id` | `PlaylistDetail.vue` | `requiresAuth` |
| `/patterns` | `PatternList.vue` | `requiresAuth` |
| `/patterns/:id` | `PatternDetail.vue` | `requiresAuth` |
| `/chat` | `MusicChatView.vue` | `requiresAuth` |
| `/friends` | `FriendsView.vue` | `requiresAuth` |
| `/debug` | `DebugView.vue` | `requiresAuth`, `requiresAdmin` |
| `/admin` | `AdminView.vue` | `requiresAuth`, `requiresAdmin` |

Гард `router.beforeEach`:

1. Читает `localStorage[AUTH_TOKEN_STORAGE_KEY]` и `[AUTH_USER_STORAGE_KEY]`.
2. Если пользователя/админ-проверки не хватает — синхронизирует сессию с сервером через `useAuthStore().initialize()` (де-дуплицирует параллельные вызовы через `authRefreshPromise`).
3. Запрещает `requiresAuth` без токена → редирект на `/login` с сохранением `redirect`.
4. Запрещает `guestOnly` при наличии токена → редирект на `/account`.
5. Запрещает `requiresAdmin` если `user.role !== 'admin'` → `/account`.

## 5. Web-маршруты Laravel, на которые опирается фронт

`routes/web.php`:

```php
Route::get('/radio/{slug}', fn ($slug) => view('public-radio', [...]));
Route::get('/embed/radio/{slug}', fn ($slug) => view('public-radio', [..., 'embed' => true]));
Route::get('/{any?}', fn () => view('app'))->where('any', '.*');
```

Catch-all `/{any?}` — это backup для случая, когда пользователь зашёл сразу на `/playlists/42` без перехода через корень: Laravel вернёт `app.blade.php`, а Vue-Router уже на клиенте отрисует нужный компонент.

## 6. Public Radio (отдельная подсистема)

- Точка входа Vite: `resources/js/public-radio.ts` (отдельный bundle).
- Страница: `resources/views/public-radio.blade.php` (рендерится Web-маршрутом).
- WebSocket-сервер: `scripts/public-radio-ws-server.mjs`. Запускается вторым процессом в контейнере `node`. Использует пакет `ws` для сокетов и `redis` для счётчика слушателей и истории чата (НЕ pub/sub — общение с Laravel идёт через REST polling, см. ниже).
- Порты:
  - Внутри контейнера: `RADIO_WS_PORT=8080`.
  - Origin до Laravel: `RADIO_APP_ORIGIN=http://app` (имя сервиса в Docker-сети).
  - Снаружи: WebSocket порт **не экспонируется напрямую** из `docker-compose.yml` — фронт должен ходить через прокси либо через тот же `node`-контейнер. Для прода это нужно сконфигурировать (например, проброс через Nginx).
- Полный протокол сообщений (3 эндпоинта `/ws/radio-listeners`, `/ws/radio-private`, `/ws/friends`, формы payload, Redis-ключи, Laravel HTTP-зависимости) — см. отдельный документ [`PUBLIC_RADIO.md`](./PUBLIC_RADIO.md).

## 7. Сборка и тесты

```powershell
# Прод-сборка
docker compose exec node npm run build

# Type-check без сборки
docker compose exec node npm run type-check

# Юнит-тесты Vitest (happy-dom)
docker compose exec node npm test
docker compose exec node npm run test:watch
```

Vitest конфиг (`vitest.config.ts`) ищет `resources/js/**/*.test.ts`, окружение — `happy-dom`. Property-based тесты — через `fast-check`.

## 8. TypeScript-конфиг

`tsconfig.json` — стандартный для Vue 3 + Vite:

- `target: ESNext`, `module: ESNext`, `moduleResolution: bundler`.
- `strict: true`.
- JSX-настройки для Vue.
- Path-alias `@/*` → `resources/js/*`.

Импорт компонентов: `import LoginView from '@/views/LoginView.vue'` и т. д.

## 9. Структура `resources/`

(подтверждено через рекурсивный листинг репозитория)

```
resources/
├── css/
│   └── app.css                      # Tailwind input
├── js/
│   ├── app.ts                       # SPA entry
│   ├── public-radio.ts              # PublicRadio entry
│   ├── router/index.ts              # Vue Router + auth-гард
│   ├── stores/                      # Pinia stores (authStore, ...)
│   ├── views/                       # Страницы (LoginView, TrackLibrary, ...)
│   ├── components/                  # Переиспользуемые компоненты
│   ├── constants/auth.ts            # AUTH_TOKEN_STORAGE_KEY, AUTH_USER_STORAGE_KEY
│   └── ...
└── views/
    ├── app.blade.php                # SPA shell (загружает Vite manifest)
    └── public-radio.blade.php       # Public radio shell
```

(точное дерево внутри `resources/js/` лучше посмотреть прямо в репозитории — оно достаточно большое)

## 10. Связь с Laravel

- Все API-вызовы — через Axios на тот же origin (`/api/...`), Bearer-токен из `localStorage`.
- Vite в dev-режиме слушает `5173`, а Laravel-Vite-plugin прокидывает `@vite([...])` в `app.blade.php`, так что в браузере вы видите hot-reload.
- В прод-сборке (`npm run build`) ассеты попадают в `public/build/`, Laravel читает manifest и отдаёт хешированные пути.

## 11. Известные особенности

- Public radio открывается без app-chrome (`hideAppChrome: true`). Layout-компонент в SPA должен это уважать.
- Каталог `/upload` поддерживает MP3/WAV до 100 МБ (лимит — `upload_max_filesize` PHP, см. [`OPERATIONS.md` §10](./OPERATIONS.md#10-мониторинг-здоровья) и `docker/app/Dockerfile`).
- `POST /api/auth/login` возвращает в т. ч. `user` — фронт сохраняет его в `localStorage[AUTH_USER_STORAGE_KEY]`. Гард читает оттуда `user.role` для `requiresAdmin`.

## 12. Не путать: blueprint v2 — отдельный фронт

В корне репозитория лежит каталог `stemlab_multicontainer_blueprint_v2/apps/frontend/` — это **отдельный React/Vite UI** для исследовательского стенда сепарации стемов, со своим `package.json`, своим dev-сервером (порт `5173` внутри своего docker-compose, конфликтует с основным Vite!) и своей документацией. Он **не часть** этого Vue-фронта. Подробнее — [`STEMLAB_BLUEPRINT_V2.md`](./STEMLAB_BLUEPRINT_V2.md).
