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

## 3. Query-based ветка — Banquet wired up, но всё ещё research-grade

С R11 в `querysep-worker` подключён реальный Banquet (`query-bandit`)
и используется в `lyrics_sync_from_track_tandem_banquet_other.yaml`,
но это всё ещё research-зона:
- веса Banquet надо положить в `models/` руками, в комплекте их нет;
- prompt/query поведение зависит от качества чекпоинта и от того, на
  каких таксономиях он обучен;
- для других query-моделей (QSCNet / TUSS) `querysep-worker` остаётся
  plug-in zone.

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

С R7 у `restoration-worker.other_residual_rebuild` появился режим
`mode: direct`, который берёт `other` напрямую из ML-модели и чистит
вокальный bleed — это уменьшает накопление ошибок относительно
старого `mode: residual` (вычитание), но семантически `other` всё
равно остаётся «всё, что не попало в named stems».

## 6. Lyrics “1-в-1” нельзя гарантировать чисто автоматом

Даже хороший локальный ASR/alignment pipeline не гарантирует идеальный текст для любого вокала.

Особенно мешают:
- сильный реверб,
- хор/даблы,
- экстремальная обработка,
- неразборчивое пение,
- смешанные языки,
- плохой source.

Поэтому в проекте и есть два режима:
- automatic transcription;
- reference-text alignment.

Если тебе нужен максимально точный текст, почти всегда лучше сначала дать reference lyrics.
