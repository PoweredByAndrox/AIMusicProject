# Roadmap

## Phase 1 — usable local lab — DONE
Цель: получить один UI и несколько контейнеров.

Сделано:
- поднят compose,
- подключён BS-RoFormer worker,
- подключён Demucs worker (`htdemucs_ft`, R2),
- подключён `audio-separator-worker` (`bus-mixer-worker`, R12),
- dry-run orchestration,
- сохранение manifests.

## Phase 2 — honest multi-architecture per stem — DONE (R1+R2+R7b+R9+R10)
Цель: реально разные архитектуры на разные стэмы.

Сделано:
- drums/bass -> Demucs FT (R2)
- vocals -> BS / hybrid vocals
- piano/guitar -> BS-RoFormer-SW
- other/harmonic -> `restoration-worker.stem_ensemble` (R9,
  `median_fft`/`min_fft`/`max_fft`/`avg_fft`/`min_mag`)
- (опционально) SCNet medium как третий backbone (R10)
- исправлено Mel-Band instrumental subtraction (R7b)
- честный `num_overlap=4` + `-ar 44100` (R1+R4)
- промежуточный `pcm_f32le` (R-PCM) — никаких 16-бит округлений
  между шагами

## Phase 3 — beyond 6 stems — DONE (R11)
Цель: strings/synth/brass/reeds/organ/fx.

Сделано:
- `querysep-worker` с реальным Banquet-runtime
- residual buses через `other_direct` (R7)
- prompt/query extraction через
  `lyrics_sync_from_track_tandem_banquet_other.yaml` (R11)
- финальная пере-сборка `other` через `stem_ensemble` (R9)

Дальше: section-specific routing (oracle-like).

## Phase 4 — restoration-first quality — IN PROGRESS (R3+R6+R7+R8+R-ACT)
Цель: убрать bleed / reverb / codec damage.

Сделано:
- успокоены `*_unmuffle` пресеты (R3)
- TTA-обёртка (polarity-flip, L/R swap) для separation (R6)
- адаптивные пороги `_activity_mask` (R-ACT)
- прямой `other` от ML-модели (`mode: direct`, R7) вместо
  накапливающего ошибки вычитания
- MWF (multichannel Wiener) post-filter (R8) поверх ML-выходов
  с профилями `soft` / `classical` / `transparent` / `off`
- WhisperX `large-v3` для lyrics (R5)

Осталось:
- dedicated dereverb / resurrection модели
- confidence-aware accept/reject (см. acceptance scoring ниже)

## Phase 4.5 — regression & evaluation infra — DONE (R13+R14)
Цель: ловить unintended drift и мерить настоящее качество.

Сделано:
- `scripts/baseline_snapshot.py` + `scripts/baseline_compare.py` (R13)
  для проверки совместимости после рефакторинга
- `scripts/sisdr_compare.py` (R14) для SI-SDR против ground truth
  (single-track или manifest-aggregate, MUSDB18-HQ-friendly)
- честное переименование `audio-separator-worker → bus-mixer-worker`
  через `honest_name` (R12), чтобы документация и код больше не врали
  про PyPI-пакет `audio-separator`
- доки выровнены с кодом (R15)

## Phase 5 — собственные модели
Цель: перестать зависеть только от чужих чекпоинтов.

Сделать:
- MoisesDB-based fine-grained training
- LoRA warm-start
- instrument-specific experts
- evaluation pipeline (SI-SDR из R14 + WER + listening tests)
