# Как фиксить проект

## Если не стартует compose

Проверь:
- `.env`
- порты 8000/5173
- docker desktop / docker engine
- volume mounts `./data` и `./configs`

## Если API жив, а UI пустой

Смотри:
- `frontend/src/api.ts`
- `VITE_API_URL`
- `GET /api/v1/catalog/presets`
- `GET /api/v1/catalog/services`

## Если job создаётся, но worker не отвечает

Смотри:
- `docker compose logs -f api`
- `docker compose logs -f demucs-worker`
- `docker compose logs -f bsroformer-worker`
- `docker compose logs -f audio-separator-worker`

## Если всё работает только в dry-run

Это нормально для первого запуска.

Нужно:
1. скачать веса моделей;
2. прописать реальные пути;
3. переключить `DRY_RUN=false`;
4. раскомментировать/дописать реальные subprocess вызовы.

## Если хочешь менять stem architecture

Точка входа:
- `configs/presets/*.yaml`
- `configs/model_registry.yaml`

Не надо сразу переписывать gateway.

## Если хочешь добавить новый stem

Пример:
- `choir`
- `backing_vocals`
- `synth_bass`
- `strings_short`
- `strings_long`

Что сделать:
1. добавить target в preset,
2. добавить service + task,
3. добавить renderer/card в UI,
4. добавить acceptance rule в orchestrator.

## Если query-based модель ещё не готова

Оставь `querysep-worker` как stub и делай так:
- выделяй `harmonic_residual`
- сохраняй его как отдельный bus
- тестируй на нём разные research модели отдельно

То есть проект всё равно полезен ещё до полной интеграции.
