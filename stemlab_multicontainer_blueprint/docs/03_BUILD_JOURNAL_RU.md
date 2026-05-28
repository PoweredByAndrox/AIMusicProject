# Как я собирал этот проект

## Шаг 1. Не расширял текущий вокальный костыль, а обобщил его

У тебя уже есть логика “вокал обрабатывается не так, как весь остальной микс”.
Значит правильный следующий шаг — не ещё больше специальных ручек только для вокала, а общая архитектура:

- один gateway,
- несколько контейнеров,
- свой stem route для каждого класса.

## Шаг 2. Разделил систему на coarse и detailed stages

Если сразу пытаться вынуть:
- orchestra,
- synth,
- pads,
- brass,
- reeds,
- organ,
- FX,

из одной fixed 6-stem модели, то `other` всё равно будет смешанным контейнером.

Поэтому проект специально построен так:
1. coarse split,
2. residual buses,
3. query-based detail split.

## Шаг 3. Отдельно выделил production и research зоны

### Production-ready
- Demucs
- BS-RoFormer-Infer
- Audio Separator

### Research-integration zone
- querysep worker
- restoration worker
- узкие классы beyond 6 stems

Это сделано специально, чтобы проект можно было чинить без боли:
- production сервисы можно поднимать сразу,
- research ветки можно подключать позже.

## Шаг 4. Вынес stem strategy в YAML, а не в код

Чтобы потом не фиксить Python при каждой перестановке моделей:
- список stems,
- routing,
- fallback,
- cleanup,
- stem families

лежат в `configs/`.

## Шаг 5. Сделал отдельные docs под фиксы

В архиве есть:
- стартовый документ,
- архитектурный,
- модельный,
- roadmap,
- fix guide,
- references.

Это нужно, чтобы через месяц не пытаться вспоминать “зачем тут вообще querysep-worker”.

## Что я бы сделал следующим руками

1. Подключил реальные команды Demucs/BS-RoFormer в workers.
2. Протянул сохранение аудио-артефактов на каждый stage.
3. Добавил waveform preview и section-level A/B.
4. Подключил реальный query-based extractor.
5. Сделал section router:
   - куплет,
   - припев,
   - брейк,
   - dense mix,
   - sparse mix.
