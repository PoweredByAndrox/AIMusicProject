# Roadmap

## Phase 1 — usable local lab
Цель: получить один UI и несколько контейнеров.

Сделать:
- поднять compose,
- подключить BS-RoFormer worker,
- подключить Demucs worker,
- подключить Audio Separator worker,
- dry-run orchestration,
- сохранить manifests.

## Phase 2 — honest multi-architecture per stem
Цель: реально разные архитектуры на разные стэмы.

Сделать:
- drums/bass -> Demucs
- vocals -> BS / hybrid vocals
- piano/guitar -> BS-RoFormer-SW
- other/harmonic -> ensemble cleanup

## Phase 3 — beyond 6 stems
Цель: strings/synth/brass/reeds/organ/fx.

Сделать:
- querysep-worker
- residual buses
- prompt/query extraction
- section-specific routing

## Phase 4 — restoration-first quality
Цель: убрать bleed / reverb / codec damage.

Сделать:
- dereverb models
- resurrection models
- per-stem expert cleanup
- confidence-aware accept/reject

## Phase 5 — собственные модели
Цель: перестать зависеть только от чужих чекпоинтов.

Сделать:
- MoisesDB-based fine-grained training
- LoRA warm-start
- instrument-specific experts
- evaluation pipeline
