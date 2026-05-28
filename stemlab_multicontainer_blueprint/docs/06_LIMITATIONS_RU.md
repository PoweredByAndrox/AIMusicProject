# Ограничения

## 1. Fine-grained stems beyond 6 не появятся “из воздуха”

Даже с хорошим 6-stem backbone:
- `strings`,
- `synth_lead`,
- `synth_pad`,
- `organ`,
- `reeds`,
- `brass`,
- `fx`

обычно требуют либо:
- query-based extraction,
- либо собственное обучение,
- либо хотя бы staged restoration.

## 2. Один универсальный backbone не лучший для всех stem'ов

То, что хорошо для:
- vocals

не обязательно лучше для:
- drums
- bass
- piano
- strings

## 3. Query-based ветка в этом архиве — интерфейс, не финальный production engine

Она специально оставлена как plug-in zone.

## 4. Restoration нужен отдельно

Если источник уже:
- mastered,
- compressed,
- reverbed,
- coded into lossy format,

то separation alone часто недостаточно.

## 5. `other` никогда не стоит считать финальным смысловым stem'ом

Его надо считать:
- residual pool,
- harmonic bus,
- donor pool,
- candidate pool for further split.
