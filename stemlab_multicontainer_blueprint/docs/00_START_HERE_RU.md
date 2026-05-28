# С чего начать

## 1. Что у тебя уже есть сейчас

Сейчас в текущем лабе уже есть отдельная вокальная ветка (`Vocals Rewrite`) поверх hybrid 6-stem пайплайна, а значит сама идея “свой pipeline под свой stem” у тебя уже частично существует. Новый проект просто распространяет эту идею на все целевые стэмы, а не только на вокал.

## 2. Что надо сделать в первую очередь

1. Поднять `docker compose up --build`
2. Проверить:
   - `GET /health`
   - `GET /api/v1/catalog/presets`
   - `GET /api/v1/catalog/services`
3. Открыть UI на `localhost:5173`
4. Создать dry-run job через форму
5. Посмотреть в job result, как gateway собрал цепочку по стэмам

## 3. Что editable первым делом

- `configs/presets/fine_grained_14stem.yaml`
- `configs/model_registry.yaml`
- `services/querysep-worker/app/main.py`
- `services/restoration-worker/app/main.py`

## 4. Что менять НЕ сразу

Не надо первым делом переписывать:
- UI
- Docker Compose
- gateway orchestration

Сначала добей:
- модельный реестр,
- stem mapping,
- реальные команды запуска нужных моделей.

## 5. Быстрый принцип

- 6-stem — это **не финал**, а **coarse routing layer**
- `other` — это **пул для дальнейшего деления**
- `querysep-worker` — это вход в long-tail инструменты
- `restoration-worker` — это вход в cleanup / dereverb / resurrection
