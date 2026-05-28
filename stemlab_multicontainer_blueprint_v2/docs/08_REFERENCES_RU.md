# Reference list для дальнейшей интеграции

## Official / primary sources
- BS-RoFormer-Infer: https://github.com/openmirlab/bs-roformer-infer
- Demucs: https://github.com/facebookresearch/demucs
- Audio Separator (PyPI): https://github.com/nomadkaraoke/python-audio-separator
  — **NB**: проект не использует этот пакет, см. R12 в
  `docs/10_V2_CHANGELOG_RU.md`. `audio-separator-worker` — это
  ffmpeg-based bus-mixer (`honest_name = bus-mixer-worker`).
- SCNet: https://github.com/amanteur/SCNet-PyTorch
  (опциональный backbone, R10; альтернативно через
  Music-Source-Separation-Training: https://github.com/ZFTurbo/Music-Source-Separation-Training)
- WhisperX: https://github.com/m-bain/whisperX
- Whisper: https://github.com/openai/whisper
- stable-ts: https://github.com/jianfch/stable-ts
- whisper-timestamped: https://github.com/linto-ai/whisper-timestamped
- MoisesDB paper: https://arxiv.org/abs/2307.15913
- MUSDB18-HQ: https://zenodo.org/records/3338373 (датасет для R14
  SI-SDR-замеров)
- Banquet paper/repo: https://arxiv.org/abs/2406.18747
- Banquet repo: https://github.com/kwatcharasupat/query-bandit
- TUSS repo: https://github.com/merlresearch/unified-source-separation
- MSR paper: https://arxiv.org/abs/2603.16926
- CP-JKU MSR paper: https://arxiv.org/abs/2603.04032
- SI-SDR: Le Roux et al., "SDR — half-baked or well done?",
  https://arxiv.org/abs/1811.02508

## Что смотреть для интеграции первым
1. BS-RoFormer-SW download + infer
2. Demucs FT runtime (`htdemucs_ft`, R2)
3. SCNet medium как третий backbone (R10, опционально)
4. WhisperX word-level timestamps (`large-v3` по умолчанию, R5)
5. stable-ts reference alignment
6. Banquet / query-based BYOQ inference (R11)
7. TUSS prompt routing
8. per-stem restoration experts (плюс R7/R8/R9 уже встроенные
   `other_direct`, `wiener_postfilter`, `stem_ensemble`)
