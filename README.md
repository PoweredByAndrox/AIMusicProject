# AI Music Project

Personal music platform: upload your library (or import from YouTube), get GPU-accelerated ML analysis (BPM / key / mood / structure / stems / embeddings), browse waveforms, label fragments, train a personal preference model, generate playlists, run a smart radio, chat with an LLM about your music, and produce stem-tuned remixes.

## What's actually inside

| Layer | Stack | Notes |
|-------|-------|-------|
| Backend | Laravel 11, PHP 8.2, Nginx, Supervisor | Custom Bearer-token auth (no Sanctum), Redis-backed ML queue |
| Frontend | Vue 3 + TypeScript, Vite, Tailwind, Pinia, Vue Router, Wavesurfer.js, HLS.js, Toastify | Two Vite entry points (SPA + public radio) |
| ML worker | Python 3.11, PyTorch 2.5.1 + CUDA 12.6, TensorFlow 2.21, Essentia, FAISS, allin1, Demucs, NATTEN, basic-pitch, librosa, yt-dlp | Three parallel containers, GPU **required** by default |
| Database | PostgreSQL 16-alpine | 37 migrations |
| Queue / cache | Redis 7-alpine | App-managed key `ai_music_recommendations_database_ml_worker:tasks` |
| LLM | Ollama (port 11434) | For AI Music Chat (`/api/chat/ai`) |
| Public radio | Standalone WebSocket server (`scripts/public-radio-ws-server.mjs`, port 8080 inside Docker network) | Lives in the `node` container |

> **GPU is required for the default `docker-compose.yml`.** Each `ml-worker*` reserves NVIDIA devices and sets `REQUIRE_CUDA=1`, `ALLOW_CPU_STRUCTURE_FALLBACK=0`. The optional `docker-compose.gpu.yml` is an *extra override* on top of that, not the GPU on/off switch.

## Quick start

```bash
git clone <repo-url>
cd AIMusicProject
cp .env.example .env

# Build images that have a Dockerfile (postgres/redis/ollama/node use prebuilt images)
docker compose build app ml-worker

# Boot everything
docker compose up -d

# Generate APP_KEY (writes into .env)
docker compose exec app php artisan key:generate

# Apply migrations
docker compose exec app php artisan migrate
```

Default endpoints (override via `.env`):

- Web app: <http://localhost:8000> (`APP_PORT`)
- Vite dev: <http://localhost:5173> (`VITE_PORT`)
- Postgres: `localhost:5432` (`DB_PORT`)
- Ollama: <http://localhost:11434> (`OLLAMA_PORT`)
- Health: `GET /up` and `GET /api/health`

### Create the first admin

There is no seed for the first user. After `migrate`, do this once via Tinker:

```bash
docker compose exec app php artisan tinker
```

```php
$u = App\Models\User::create([
    'name' => 'Admin',
    'email' => 'admin@example.local',
    'password' => 'change-me-please', // hashed automatically
    'role' => 'admin',
]);
$plain = bin2hex(random_bytes(32));
$u->apiTokens()->create([
    'name' => 'bootstrap',
    'token_hash' => hash('sha256', $plain),
]);
echo "Bearer token: $plain\n";
```

Use that token as `Authorization: Bearer <token>` for every authenticated `/api/*` call.

### Pull an Ollama model (for AI Music Chat)

```bash
docker compose exec ollama ollama pull llama3
```

## Documentation

| Doc | What's in it |
|-----|--------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Container topology, queue protocol, layer-by-layer breakdown, scheduled jobs |
| [`docs/INSTALLATION.md`](docs/INSTALLATION.md) | Detailed installation, env vars, GPU requirements, common pitfalls |
| [`docs/API.md`](docs/API.md) | Full REST endpoint reference (public, authenticated, admin, web routes) |
| [`docs/DATABASE.md`](docs/DATABASE.md) | Tables, key columns, sidecar files, migrations history |
| [`docs/ML_WORKER.md`](docs/ML_WORKER.md) | Python worker internals, task types, env vars, dependencies, watchdogs |
| [`docs/FRONTEND.md`](docs/FRONTEND.md) | Vue routes, Vite entry points, public-radio WebSocket, build/test |
| [`docs/PUBLIC_RADIO.md`](docs/PUBLIC_RADIO.md) | Full WebSocket protocol for `public-radio-ws-server.mjs` (clients, messages, Redis keys, Laravel API calls) |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Day-to-day ops: logs, scheduler, queue, backup, incident runbooks, utility scripts registry |
| [`docs/STEMLAB_BLUEPRINT_V2.md`](docs/STEMLAB_BLUEPRINT_V2.md) | Pointer doc for the parallel research stack `stemlab_multicontainer_blueprint_v2/` (its own Docker compose, FastAPI gateway, React UI, separator workers). The blueprint's own `docs/` is the canonical reference. |
| [`AGENTS.md`](AGENTS.md) | Mandatory rules and response contract for AI agents working on this repo |
| [`TASKS.md`](TASKS.md) | Open backlog (Preference Model, Segment Analysis) |

## Tests

```bash
# PHP (Laravel) — uses in-memory SQLite (see phpunit.xml)
docker compose exec app php artisan test

# Python ML worker
docker compose exec ml-worker pytest tests/

# Frontend (Vitest, happy-dom)
docker compose exec node npm test
```

## Optional: temporary public URL via Cloudflare Quick Tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

See [`cloudflare_tunnel/README.md`](cloudflare_tunnel/README.md).

## License

`composer.json` declares **MIT**. No `LICENSE` file is committed at the repo root.

## Project status

- The 37 migrations under `database/migrations/` cover users, tracks, fragments, queue tasks, playlists, ratings, listening history, smart radio, AI DJ, patterns, friendships, direct messages, stem-tuning lab artifacts.
- The `ml_worker_improved/` tree is the live ML worker. Two adjacent directories — `stemlab_multicontainer_blueprint_v1/` and `stemlab_multicontainer_blueprint_v2/` — are *separate experimental blueprints* not wired into the runtime stack. For v2 see [`docs/STEMLAB_BLUEPRINT_V2.md`](docs/STEMLAB_BLUEPRINT_V2.md) (high-level pointer) and the blueprint's own `stemlab_multicontainer_blueprint_v2/docs/` (full reference).
- See [`TASKS.md`](TASKS.md) for the current backlog and [`AGENTS.md`](AGENTS.md) for AI-assistant working agreements (forced source-of-truth execution).
