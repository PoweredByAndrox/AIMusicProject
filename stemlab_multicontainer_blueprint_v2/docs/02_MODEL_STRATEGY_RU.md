# Стратегия по моделям и стэмам

## Цель

Не просто 6 stem'ов, а **иерархическая многостемная система**, где для каждого семейства источников своя логика.

## Уровень 1: coarse stems

Базовый backbone:
- `vocals`
- `drums`
- `bass`
- `guitar`
- `piano`
- `other`

### Рекомендуемая база
- BS-RoFormer-SW как основной 6-stem backbone
- Demucs FT (`htdemucs_ft`, R2) как validation / donor для drums+bass
- (опционально, R10) SCNet medium как третий backbone
- `restoration-worker.stem_ensemble` (R9) как **ML-ensemble layer**
  (`median_fft` / `min_fft` / `max_fft` / `avg_fft` / `min_mag`)
- `restoration-worker.wiener_postfilter` (R8) как post-cleanup поверх
  ML-выходов

> NB: `audio-separator-worker` (`honest_name = bus-mixer-worker`, R12)
> — это детерминистический ffmpeg-микшер шин, **не learnable
> ensemble**. Если в старом тексте написано "Audio Separator как
> ensemble/fallback layer", читай это как
> `restoration-worker.stem_ensemble` + `wiener_postfilter`.

## Уровень 2: stem-specific refinement

### Vocals
- primary: BS-RoFormer vocals-oriented branch
- fallback: hybrid vocals blend
- cleanup: resurrection / dereverb

### Drums
- primary: Demucs FT
- optional cleanup: transient restore / bleed control

### Bass
- primary: Demucs FT
- optional cleanup: low-end consistency pass

### Guitar
- primary: BS-RoFormer-SW
- future split:
  - acoustic_guitar
  - electric_guitar
  - distorted_guitar
  - clean_guitar

### Piano / Keys
- primary: BS-RoFormer-SW
- future split:
  - piano
  - e_piano
  - organ
  - synth_keys

## Уровень 3: harmonic residual / melody residual

Из coarse `other` + части `piano/guitar` строится residual bus для дальнейшего деления на:
- `strings_orchestra`
- `synth_lead`
- `synth_pad`
- `arps_plucks`
- `brass_woodwinds`
- `organ_reeds`
- `fx_risers_impacts`

## Почему именно через residual bus

Потому что эти классы часто не имеют стабильного отдельного fixed output в стандартных моделях.
Они сидят внутри:
- `other`
- harmonic layers
- melodic layers
- ambience / FX tails

## Query-based ветка

Это research-ориентированный контейнер:
- вход: residual bus + prompt/query
- выход: узкий целевой stem

Под это место удобно подключать:
- Banquet (R11: уже wired up в `querysep-worker`, активирован в
  пресете `lyrics_sync_from_track_tandem_banquet_other.yaml`)
- QSCNet-подобные модели
- TUSS-подобные prompt systems

### R7+R9+R11 — рабочая цепочка для long-tail

Стандартная связка по умолчанию:

1. `restoration-worker.other_residual_rebuild` с `mode: direct` (R7) —
   берёт `other` напрямую из ML-модели и чистит вокальный bleed
   адаптивной маской из R-ACT, не накапливая ошибки вычитания.
2. `querysep-worker` (Banquet, R11) — извлекает `strings`, `synth_lead`,
   `synth_pad`, `brass`, `organ`, `fx` из получившегося `other_direct`.
3. `restoration-worker.stem_ensemble` (R9, `median_fft`) — собирает
   финальный `other` из нескольких кандидатов (BS-RoFormer, Demucs,
   опционально SCNet, плюс Banquet-output) для устойчивости к
   miscategorisation.

## Restoration

После separation каждый stem проходит свой cleanup:

- vocals -> dereverb / resurrection
- instrumental mix -> instrumental resurrection (S1)
- strings/synth -> harmonic smoothing
- drums -> transient repair
- broad harmonic bus -> de-reverb if needed
- любой стэм -> `wiener_postfilter` (R8) для подавления musical noise
  и cross-stem leakage; soft-mask Wiener поверх phase оригинального
  микса (профили `soft` / `classical` / `transparent` / `off`)
- mixed/ambiguous bins -> `stem_ensemble` (R9) для агрегации
  нескольких ML-кандидатов на одно и то же место

### S1: instrumental_resurrection стал реальным instrumental output

До S1 `bsroformer-worker.instrumental_resurrection` (и любая
folder-based BS-роформер модель с `target_instrument: vocals`)
выбирал `primary = instruments[0]` для residual subtraction. Если у
модели в YAML стояло `instruments: [vocals, instrumental]` — стэмы
получались правильно по случайности, а если `instruments: [other,
vocals, ...]` — `*_instrumental.wav` физически содержал `mix - other`,
то есть всё, кроме `other`, включая вокал. Результат: pipeline дальше
работал не с инструменталом, а с почти full-mix'ом, и это уезжало в
hybrid как "BS seed" с -5..-10 dB вокального bleed.

Сейчас `services/bsroformer-worker/app/runtime.py` использует
`_resolve_instrumental_strategy(instruments, target_instrument)`,
которая выбирает один из трёх режимов:

- `residual` поверх `vocals` — если модель действительно знает вокал
  как primary (`target_instrument: vocals` или `vocals` есть в списке);
- `direct` — если модель честно отдаёт нужный инструментальный стэм;
- `sum` — fallback по сумме всех ненулевых outputs.

Это инвариант, который S4-юнит-тесты в
`services/bsroformer-worker/tests/test_runtime_instrumental_strategy.py`
теперь стерегут на синтетических vocals/instrumental сигналах.

### S2: safety-net в tandem_instrumental_hybrid

`restoration-worker.run_tandem_instrumental_hybrid` теперь оценивает
`_vocal_contamination_db(track, vocals_guide, sr)` для трёх кандидатов:

- `base_audio` (Demucs base);
- `bs_seed_audio` (resurrected BS instrumental);
- `raw_bs_audio` (исходный BS instrumental).

Если BS-кандидат **хуже** Demucs base по vocal bleed более чем на
`bs_overlay_safety_margin_db` (default `3.0` dB), его вес в смеси
обнуляется. Без этого исторически случался сценарий "overlay
disabled by guideline, но всё равно влит" — и `final.bleed_db_rel_total`
мог уехать с -28 dB до -10 dB на агрессивных треках.

Контроль через preset:

```yaml
- stem: instrumental_mix
  service: restoration-worker
  task: tandem_instrumental_hybrid
  bs_overlay_safety_margin_db: 3.0   # 0 = выключить guard полностью
```

Метаданные шага (`metadata.bs_overlay_safety_net`) теперь
отражают **фактически применённые** веса (`bs_seed_weight`,
`bs_raw_weight`) и причину override
(`bs_overlay_disabled_reason`), а не желаемые из profile.

### S3: новый default preset

С S3 default-пресетом фронтенда стал
`lyrics_sync_from_track_tandem_residual_other_center_hard.yaml`.
Старый `lyrics_sync_from_track_tandem.yaml` остаётся доступным как
"openness over cleanliness" вариант. Замер на Nine Thou validation:
`vocal_bleed.final_vocals.bleed_db_rel_total` улучшилось с
**-22.71 dB** (legacy) до **-28.11 dB** (новый default), при
сохранении той же `vocal_clarity_*` репорт-зоны и без появления
warble penalty.

## Практический rollout

### Phase 1
Запустить только:
- BS-RoFormer-SW
- Demucs FT
- Audio Separator

### Phase 2
Добавить:
- querysep-worker
- detailed target stems
- per-stem acceptance UI

### Phase 3
Добавить:
- restoration experts
- собственные LoRA / fine-tuning
- confidence maps
- oracle-like routing by section

## R10: SCNet medium как третий backbone (опционально)

После завершения P0 + P1 из `stem-tandem-improvement-roadmap_86d167c0.plan.md` появилась
поддержка SCNet medium в `bsroformer-worker` через `_detect_model_family` /
`_create_scnet_model`. Это P2-уровень: модель **не** идёт в комплекте, её
надо положить вручную, потому что веса обновляются часто и тащить их в
docker-build не имеет смысла.

### Установка

1. Скачать веса и `config.yaml` SCNet medium с
   [Politrees/UVR_resources](https://huggingface.co/Politrees/UVR_resources)
   (раздел `configs/SCnet/`).
2. Положить их в `models/scnet-medium-vocals-by-zfturbo/`:

   ```
   models/
     scnet-medium-vocals-by-zfturbo/
       config.yaml
       checkpoint.ckpt
   ```

3. Поставить SCNet-инференс одним из способов в порядке приоритета (worker
   попробует их сам):
   - `pip install scnet` (когда пакет станет доступен в PyPI)
   - вендорная копия `Music-Source-Separation-Training` →
     `services/bsroformer-worker/vendor/scnet/` (поднимется через
     `SCNET_VENDOR_ROOT`)
   - копия в `/app/vendor/Music-Source-Separation-Training` внутри контейнера

4. Прописать слаг `scnet-medium-vocals-by-zfturbo` в нужный preset как
   третьего кандидата для `stem_ensemble` (R9). Пример:

   ```yaml
   - stem: vocals_ensemble_input_scnet
     service: bsroformer-worker
     task: vocals_resurrection
     model: scnet-medium-vocals-by-zfturbo
     input_bus: bs_vocals
   ```

5. Затем в `stem_ensemble` шаге передать все три кандидата
   (`bs_vocals`, `demucs_vocals`, `scnet_vocals`) — median_fft даст
   3-источниковый ensemble в духе MVSEP best-2025.

### Память

- SCNet **medium** влезает в 10–12 ГБ VRAM при `batch_size=1`,
  `chunk_size=485100`, `num_overlap=4`.
- SCNet **XL** — только с `use_amp: true` (fp16) на 16+ ГБ; иначе OOM при
  TTA-проходах из R6.
