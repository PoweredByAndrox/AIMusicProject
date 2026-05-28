# StemLab Multi-Container Blueprint

Каркас проекта для локальной системы тестирования и оркестрации music source separation.

## Что это такое

Это **монорепо-скелет** для локальной лаборатории, где:
- разные семейства моделей запускаются в **разных Docker-контейнерах**;
- у всех сервисов есть **единый API и единый UI**;
- можно собирать **иерархический pipeline**: coarse split -> targeted stem routing -> restoration -> AB-тесты;
- можно назначать **свою архитектуру для каждого стэма**;
- можно держать рядом как production-ready сервисы, так и research-ветки.

## Что уже заложено

- `apps/api` — FastAPI gateway + pipeline orchestrator
- `apps/frontend` — React/Vite UI для тестов
- `services/demucs-worker` — обёртка над Demucs
- `services/bsroformer-worker` — обёртка над BS-RoFormer-Infer
- `services/audio-separator-worker` — обёртка над Audio Separator / ensemble
- `services/querysep-worker` — интерфейс под query-based extraction (Banquet / QSCNet / TUSS)
- `services/restoration-worker` — интерфейс под targeted restoration / cleanup
- `configs/presets` — пресеты пайплайнов
- `docs` — объяснение архитектуры, плана и фиксов

## Что важно понять сразу

Этот архив — **не упакованные веса моделей** и не готовый “one-click magic separator”.
Это **проектная база**, которую можно:
1. поднять в Docker,
2. быстро начать тесты,
3. дальше расширять под свои модели и свои чекпоинты.

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build
```

После старта:
- API: `http://localhost:8000`
- UI: `http://localhost:5173`
- health: `http://localhost:8000/health`

## Главная идея проекта

Не пытаться выжать из одной fixed 6-stem модели всё сразу.

Вместо этого:
1. сделать **coarse 6-stem разложение**;
2. для каждого coarse stem или residual bus выбрать **свою лучшую архитектуру**;
3. long-tail классы (`strings`, `synth_lead`, `synth_pad`, `brass_woodwinds`, `organ_reeds`, `fx`) извлекать **не fixed decoder-ом**, а query/prompt-based веткой;
4. после separation делать **restoration**, а не считать separation финалом.

## Куда смотреть первым делом

- `docs/00_START_HERE_RU.md`
- `docs/01_ARCHITECTURE_RU.md`
- `docs/02_MODEL_STRATEGY_RU.md`
- `configs/presets/fine_grained_14stem.yaml`
- `docker-compose.yml`

## Режимы готовности

### Реально можно запускать почти сразу
- gateway
- UI
- Demucs worker
- BS-RoFormer worker
- Audio Separator worker

### Требуют доинтеграции / дообучения / своих чекпоинтов
- querysep worker
- restoration worker для stem-specific experts
- truly fine-grained classes beyond 6 stems

## Лицензии и модели

Проверь лицензии датасетов, чекпоинтов и репозиториев перед коммерческим использованием.
