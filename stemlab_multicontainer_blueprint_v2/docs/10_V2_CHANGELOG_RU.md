# Changelog v2

## Added
- `lyrics-worker`
- `/api/v1/uploads/audio`
- `/api/v1/uploads/text`
- `/api/v1/jobs/{job_id}/artifacts`
- media serving через `/media/...`
- preset `lyrics_sync_from_vocals`
- frontend lyrics parser
- frontend artifact browser
- frontend audio player + text sync

## Changed
- `bsroformer-worker` теперь staging'ит single-file input
- gateway сохраняет `named_outputs` в `job.artifacts`
- presets умеют включать lyrics-фазы
- UI теперь опрашивает jobs и показывает реальные артефакты

## Fixed
- архитектурная дыра, где live BS-RoFormer мог читать всю input-папку
- отсутствие честного lyrics/test workflow внутри UI
- отсутствие общего artifact API для audio/text outputs

## Still stub / TODO
- полный actual-output mapping для всех external CLI tools
- fine-grained long-tail separation beyond user-supplied models
- собственные fine-tuning / LoRA / oracle-routing
- waveform preview во фронтенде
- acceptance scoring (на базе SI-SDR из R14)

## P0–P3 reliability sweep (R1–R14, см. `stem-tandem-improvement-roadmap_86d167c0.plan.md`)

### Added
- `restoration-worker.other_residual_rebuild` — режим `mode: direct`
  (по умолчанию) на ML-`other` с очисткой вокального bleed
  (R7).
- `restoration-worker.wiener_postfilter` — soft-mask multichannel
  Wiener поверх ML-стэмов, с numpy-based STFT/iSTFT (no scipy/torch),
  профили `soft` / `classical` / `transparent` / `off` (R8).
- `restoration-worker.stem_ensemble` — FFT-based объединение нескольких
  кандидатов одного стэма: `median_fft` / `min_fft` / `max_fft` /
  `avg_fft` / `min_mag`, с выбором phase reference (mixture vs.
  candidate) (R9).
- `bsroformer-worker` детектит и грузит SCNet medium как третий
  backbone (R10): `_detect_model_family` + `_create_scnet_model`,
  `SCNET_VENDOR_ROOT` в `docker-compose.yml`,
  `models/scnet-medium-vocals-by-zfturbo/` как точка установки.
- `querysep-worker` Banquet wired up в production-пресет
  `lyrics_sync_from_track_tandem_banquet_other.yaml` (R11).
- `audio-separator-worker` теперь публикует `honest_name =
  bus-mixer-worker` в `/capabilities`, в `model_registry.yaml` стоит
  `role: heuristic_bus_mixer`, `package_installed: false` (R12).
- TTA-обёртка (polarity flip + L/R swap) в separation-инференсе (R6).
- Адаптивные пороги `_activity_mask` (R-ACT) в `restoration-worker`.
- `scripts/baseline_snapshot.py` + `scripts/baseline_compare.py` —
  per-stem fingerprint + diff с tolerances (R13).
- `scripts/sisdr_compare.py` — SI-SDR/SDR scorer для single-track или
  manifest-aggregate против ground-truth (R14).

### Changed
- `num_overlap` в separation поднят с `2` до `4`; ffmpeg переведён
  на честный `-ar 44100` без неявных downsample/upsample (R1+R4).
- Demucs default переключен `htdemucs_6s → htdemucs_ft` (R2).
- `*_unmuffle`-пресеты успокоены, чтобы не задирать highpass/EQ до
  слышимых артефактов (R3).
- Промежуточные wav'ы в pipeline переведены `pcm_s16le → pcm_f32le`,
  чтобы не терять headroom между шагами (R-PCM).
- WhisperX `large-v2 → large-v3` в `lyrics-worker` (R5).

### Fixed
- Mel-Band instrumental subtraction (R7b): больше не вычитает на
  чужой шкале мощности, не ломает фазу.

### Docs (R15)
- `README.md`, `PROJECT_STATUS_RU.md`, `docs/01_ARCHITECTURE_RU.md`,
  `docs/02_MODEL_STRATEGY_RU.md`, `docs/05_ROADMAP_RU.md`,
  `docs/06_LIMITATIONS_RU.md`, `docs/07_CURRENT_LAB_MAPPING_RU.md`,
  `docs/08_REFERENCES_RU.md`, `docs/09_LYRICS_WORKER_RU.md` и
  `docs/04_FIX_GUIDE_RU.md` приведены в соответствие с реальным
  кодом после P0–P3 sweep.
