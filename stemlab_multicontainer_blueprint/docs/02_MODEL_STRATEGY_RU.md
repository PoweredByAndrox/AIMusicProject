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
- Demucs FT как validation / donor для drums+bass
- Audio Separator как ensemble/fallback layer

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
- Banquet
- QSCNet-подобные модели
- TUSS-подобные prompt systems

## Restoration

После separation каждый stem проходит свой cleanup:

- vocals -> dereverb / resurrection
- instrumental mix -> instrumental resurrection
- strings/synth -> harmonic smoothing
- drums -> transient repair
- broad harmonic bus -> de-reverb if needed

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
