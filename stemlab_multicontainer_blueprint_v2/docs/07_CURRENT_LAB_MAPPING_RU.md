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

После P0–P3 sweep (R1–R14) `Other` собирается так:
- `restoration-worker.other_residual_rebuild` (`mode: direct`, R7)
  берёт ML-`other` напрямую,
- (опционально) `querysep-worker` (Banquet, R11) тащит из него
  `strings/synth/brass/organ/fx`,
- `restoration-worker.stem_ensemble` (R9) объединяет нескольких
  кандидатов в финальный `other`,
- `restoration-worker.wiener_postfilter` (R8) дочищает musical noise.
