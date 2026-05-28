# Как новый проект соотносится с твоим текущим Tuning Lab

## Сейчас у тебя

- hybrid 6-stem base
- отдельная вокальная ветка (`Vocals Rewrite`)
- per-stem post-DSP
- focus на локальном тестировании отдельных участков

## Новый проект добавляет

### Было
Только вокал имел отдельную “специальную” архитектурную ветку.

### Станет
Каждый stem / stem family сможет иметь:
- свой backbone,
- свой cleanup,
- свой fallback,
- свой detailed extractor.

## Сопоставление

### `Vocals Rewrite`
-> `bsroformer-worker` + `restoration-worker`

### `Hybrid Build`
-> coarse split stage

### `Stem Rewrite`
-> targeted refine stage

### `Post-DSP`
-> post-processing / acceptance layer

### `Other`
-> residual/harmonic bus для long-tail extraction
