from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import numpy as np

import worker.high_quality_stem_export as hq_export
from worker.high_quality_stem_export import (
    HYBRID_BS_STEM_EXPORT_MODEL,
    HYBRID_STAGE1_MODEL,
    HYBRID_STAGE2_MODEL,
    HYBRID_STEM_EXPORT_MODEL,
    _aggressive_melody_first_rebalance,
    _blend_vocal_stems,
    _cleanup_single_vocal_stem,
    _component_transfer_envelope,
    _redistribute_instrumental_melody_from_stems,
    _rebalance_hybrid_stage2_stems,
    _split_harmonic_percussive,
    _separate_track_stems_hybrid_bs_vocals_first,
    _separate_track_stems_hybrid_vocals_first,
    export_vocal_preview_from_audio,
    export_track_stems_full_res,
    export_aggressive_melody_first_from_stem_dir,
    extract_melody_stem_from_stems,
    repair_exported_stem_directory,
    sharpen_stem_bleed,
    export_track_stems_aggressive_melody_first,
)
from worker.stem_separator_defaults import DEFAULT_STEM_EXPORT_MODEL


class TestHighQualityStemExport(unittest.TestCase):
    @patch("worker.high_quality_stem_export._load_bs_roformer_vocals_other")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_hybrid_bs_pipeline_keeps_stage1_vocals_and_stage2_instrumental_stems(self, separate_mock, bs_mock):
        stage1_vocals = np.full((44100, 2), 0.75, dtype=np.float32)
        bs_vocals = np.full((44100, 2), 0.45, dtype=np.float32)
        bs_mock.return_value = {
            "vocals": bs_vocals,
            "instrumental": np.full((44100, 2), 0.45, dtype=np.float32),
        }
        stage2_stems = {
            "drums": np.full((44100, 2), 0.3, dtype=np.float32),
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.2, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.01, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.05, dtype=np.float32),
            "piano": np.full((44100, 2), 0.04, dtype=np.float32),
        }

        def fake_separate(audio_path, *, model_name, device=None, **_kwargs):
            if model_name == HYBRID_STAGE1_MODEL:
                return {
                    "drums": np.full((44100, 2), 0.15, dtype=np.float32),
                    "bass": np.full((44100, 2), 0.1, dtype=np.float32),
                    "other": np.full((44100, 2), 0.2, dtype=np.float32),
                    "vocals": stage1_vocals,
                }
            if model_name == HYBRID_STAGE2_MODEL:
                return stage2_stems
            raise AssertionError(model_name)

        separate_mock.side_effect = fake_separate

        hybrid = _separate_track_stems_hybrid_bs_vocals_first("audio/example.mp3", device="cuda")

        self.assertEqual(list(hybrid.keys()), ["drums", "bass", "other", "vocals", "guitar", "piano"])
        np.testing.assert_allclose(hybrid["vocals"], _blend_vocal_stems(stage1_vocals, bs_vocals))
        np.testing.assert_allclose(hybrid["other"], stage2_stems["other"])
        bs_mock.assert_called_once()

    @patch("worker.high_quality_stem_export._load_bs_roformer_vocals_other")
    @patch("worker.high_quality_stem_export._rebalance_hybrid_stage2_stems")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_hybrid_vocals_first_pipeline_keeps_stage1_vocals_and_stage2_instrumental_stems(self, separate_mock, rebalance_mock, bs_mock):
        stage1_vocals = np.full((44100, 2), 0.75, dtype=np.float32)
        bs_vocals = np.full((44100, 2), 0.55, dtype=np.float32)
        stage1_other = np.full((44100, 2), 0.2, dtype=np.float32)
        stage1_drums = np.full((44100, 2), 0.15, dtype=np.float32)
        stage1_bass = np.full((44100, 2), 0.1, dtype=np.float32)
        stage2_vocals = np.full((44100, 2), 0.01, dtype=np.float32)
        stage2_stems = {
            "drums": np.full((44100, 2), 0.3, dtype=np.float32),
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.2, dtype=np.float32),
            "vocals": stage2_vocals,
            "guitar": np.full((44100, 2), 0.05, dtype=np.float32),
            "piano": np.full((44100, 2), 0.04, dtype=np.float32),
        }

        def fake_separate(audio_path, *, model_name, device=None, **_kwargs):
            if model_name == HYBRID_STAGE1_MODEL:
                return {
                    "drums": stage1_drums,
                    "bass": stage1_bass,
                    "other": stage1_other,
                    "vocals": stage1_vocals,
                }
            if model_name == HYBRID_STAGE2_MODEL:
                return stage2_stems
            raise AssertionError(model_name)

        separate_mock.side_effect = fake_separate
        rebalance_mock.side_effect = lambda stems: stems
        bs_mock.return_value = {
            "vocals": bs_vocals,
            "instrumental": np.full((44100, 2), 0.45, dtype=np.float32),
        }

        hybrid = _separate_track_stems_hybrid_vocals_first("audio/example.mp3", device="cuda")

        self.assertEqual(list(hybrid.keys()), ["drums", "bass", "other", "vocals", "guitar", "piano"])
        np.testing.assert_allclose(hybrid["vocals"], _blend_vocal_stems(stage1_vocals, bs_vocals))
        np.testing.assert_allclose(hybrid["drums"], stage2_stems["drums"])
        np.testing.assert_allclose(hybrid["guitar"], stage2_stems["guitar"])
        self.assertEqual(separate_mock.call_args_list[0].kwargs["model_name"], HYBRID_STAGE1_MODEL)
        self.assertEqual(separate_mock.call_args_list[1].kwargs["model_name"], HYBRID_STAGE2_MODEL)
        rebalance_mock.assert_called_once()
        bs_mock.assert_called_once()

    def test_blend_vocal_stems_prefers_bs_low_band_and_ft_high_band(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        low = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 5000.0 * t).astype(np.float32)
        stage1 = np.stack([0.2 * low + high, 0.2 * low + high], axis=1).astype(np.float32)
        bs = np.stack([low + 0.2 * high, low + 0.2 * high], axis=1).astype(np.float32)

        blended = _blend_vocal_stems(stage1, bs)

        def band_rms(audio: np.ndarray, low_hz: float, high_hz: float) -> float:
            mono = audio.mean(axis=1)
            spectrum = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
            mask = (freqs >= low_hz) & (freqs < high_hz)
            values = spectrum[mask]
            return float(np.sqrt(np.mean(values * values))) if values.size else 0.0

        self.assertGreater(band_rms(blended, 150.0, 400.0), band_rms(stage1, 150.0, 400.0))
        self.assertGreater(band_rms(blended, 4000.0, 7000.0), band_rms(bs, 4000.0, 7000.0))

    def test_blend_vocal_stems_suppresses_tail_when_bs_goes_inactive(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        active_voice = (0.6 * np.sin(2 * np.pi * 220.0 * t) + 0.4 * np.sin(2 * np.pi * 4800.0 * t)).astype(np.float32)
        tail_leak = (0.18 * np.sin(2 * np.pi * 220.0 * t) + 0.26 * np.sin(2 * np.pi * 5200.0 * t)).astype(np.float32)

        stage1_mono = np.concatenate([active_voice[: sample_rate // 2], tail_leak[sample_rate // 2 :]]).astype(np.float32)
        bs_mono = np.concatenate([active_voice[: sample_rate // 2], np.zeros(sample_rate - sample_rate // 2, dtype=np.float32)]).astype(np.float32)
        stage1 = np.stack([stage1_mono, stage1_mono], axis=1).astype(np.float32)
        bs = np.stack([bs_mono, bs_mono], axis=1).astype(np.float32)

        blended = _blend_vocal_stems(stage1, bs)

        def window_rms(audio: np.ndarray, start: int, end: int) -> float:
            mono = audio[start:end].mean(axis=1)
            return float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0

        active_start = int(0.15 * sample_rate)
        active_end = int(0.35 * sample_rate)
        tail_start = int(0.7 * sample_rate)
        tail_end = int(0.9 * sample_rate)

        self.assertGreater(window_rms(blended, active_start, active_end), 0.45 * window_rms(stage1, active_start, active_end))
        self.assertLess(window_rms(blended, tail_start, tail_end), 0.35 * window_rms(stage1, tail_start, tail_end))

    def test_blend_vocal_stems_trims_small_backend_frame_mismatch(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        stage1_mono = (0.45 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        bs_mono = np.pad(stage1_mono, (0, 1)).astype(np.float32)
        stage1 = np.stack([stage1_mono, stage1_mono], axis=1).astype(np.float32)
        bs = np.stack([bs_mono, bs_mono], axis=1).astype(np.float32)

        blended = _blend_vocal_stems(stage1, bs)

        self.assertEqual(blended.shape, stage1.shape)

    @patch("worker.high_quality_stem_export._load_bs_roformer_vocals_other")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_from_audio_hybrid_writes_window_and_manifest(self, separate_mock, bs_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        mono = (0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 4200.0 * t)).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {"vocals": (audio * 0.5).astype(np.float32)}

        def bs_side_effect(audio_path, *, device=None, **_kwargs):
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {
                "vocals": (audio * 0.25).astype(np.float32),
                "instrumental": np.zeros_like(audio, dtype=np.float32),
            }

        separate_mock.side_effect = separate_side_effect
        bs_mock.side_effect = bs_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(source_path, stereo, sample_rate)

            manifest = export_vocal_preview_from_audio(
                source_path,
                output_dir,
                candidate_mode="hybrid_vocals",
                start_sec=0.2,
                end_sec=0.6,
                device="cuda",
            )

            output_audio, output_sr = sf.read(output_dir / "vocals.flac", always_2d=True)
            self.assertEqual(output_sr, sample_rate)
            preview_audio = stereo[int(0.2 * sample_rate):int(0.6 * sample_rate)]
            expected = _blend_vocal_stems(preview_audio * 0.5, preview_audio * 0.25)
            np.testing.assert_allclose(output_audio, expected, atol=5e-5)

            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["candidate_mode"], "hybrid_vocals")
            self.assertEqual(saved_manifest["pipeline"]["candidate_mode"], "hybrid_vocals")
            self.assertEqual(saved_manifest["preview_window_sec"]["start"], 0.2)
            self.assertEqual(saved_manifest["preview_window_sec"]["end"], 0.6)
            self.assertAlmostEqual(saved_manifest["stems"]["vocals"]["duration_sec"], 0.4, places=3)

    @patch("worker.high_quality_stem_export._load_bs_roformer_vocals_other")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_from_audio_ft_uses_stage1_only(self, separate_mock, bs_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        mono = (0.25 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {"vocals": (audio * 0.75).astype(np.float32)}

        separate_mock.side_effect = separate_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(source_path, stereo, sample_rate)

            manifest = export_vocal_preview_from_audio(
                source_path,
                output_dir,
                candidate_mode="ft_vocals",
                start_sec=0.1,
                end_sec=0.5,
                device="cuda",
            )

            output_audio, output_sr = sf.read(output_dir / "vocals.flac", always_2d=True)
            self.assertEqual(output_sr, sample_rate)
            expected = stereo[int(0.1 * sample_rate):int(0.5 * sample_rate)] * 0.75
            np.testing.assert_allclose(output_audio, expected, atol=5e-5)
            bs_mock.assert_not_called()

            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["candidate_mode"], "ft_vocals")
            self.assertEqual(saved_manifest["pipeline"]["vocals_source"], HYBRID_STAGE1_MODEL)

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_from_audio_ft_hard_mask_sets_postprocess(self, separate_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        mixture_mono = (voice + 0.8 * leak).astype(np.float32)
        stereo = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {"vocals": (audio * 0.8).astype(np.float32)}

        separate_mock.side_effect = separate_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(source_path, stereo, sample_rate)

            manifest = export_vocal_preview_from_audio(
                source_path,
                output_dir,
                candidate_mode="ft_vocals",
                start_sec=0.0,
                end_sec=0.5,
                device="cpu",
                hard_mask_threshold=0.6,
                hard_mask_power=3.0,
                hard_mask_residual_floor=0.02,
            )

            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["vocal_postprocess"], "hard_subtractive_vocal_mask_v1")
            self.assertEqual(saved_manifest["pipeline"]["vocal_postprocess"], "hard_subtractive_vocal_mask_v1")
            self.assertEqual(saved_manifest["stems"]["vocals"]["vocal_postprocess"], "hard_subtractive_vocal_mask_v1")

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_from_audio_ft_subtractive_sets_postprocess(self, separate_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        mixture_mono = (voice + 0.8 * leak).astype(np.float32)
        stereo = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {"vocals": (audio * 0.8).astype(np.float32)}

        separate_mock.side_effect = separate_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(source_path, stereo, sample_rate)

            manifest = export_vocal_preview_from_audio(
                source_path,
                output_dir,
                candidate_mode="ft_vocals",
                start_sec=0.0,
                end_sec=0.5,
                device="cpu",
                subtractive_strength=1.2,
                subtractive_power=1.3,
                subtractive_residual_floor=0.12,
            )

            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v1")
            self.assertEqual(saved_manifest["pipeline"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v1")
            self.assertEqual(saved_manifest["stems"]["vocals"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v1")

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_resamples_cleanup_mixture_to_model_rate(self, separate_mock):
        source_sample_rate = 48000
        target_sample_rate = 44100
        duration_sec = 0.5
        source_t = np.linspace(
            0.0,
            duration_sec,
            int(source_sample_rate * duration_sec),
            endpoint=False,
            dtype=np.float32,
        )
        target_t = np.linspace(
            0.0,
            duration_sec,
            int(target_sample_rate * duration_sec),
            endpoint=False,
            dtype=np.float32,
        )
        source_mono = (
            0.35 * np.sin(2 * np.pi * 220.0 * source_t)
            + 0.15 * np.sin(2 * np.pi * 880.0 * source_t)
        ).astype(np.float32)
        model_vocals = np.stack(
            [
                0.25 * np.sin(2 * np.pi * 220.0 * target_t),
                0.25 * np.sin(2 * np.pi * 220.0 * target_t),
            ],
            axis=1,
        ).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            _, sample_rate = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sample_rate, source_sample_rate)
            return {"vocals": model_vocals}

        separate_mock.side_effect = separate_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source_48k.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(
                source_path,
                np.stack([source_mono, source_mono], axis=1),
                source_sample_rate,
            )

            def cleanup_side_effect(vocals_audio, mixture_audio, **_kwargs):
                self.assertEqual(vocals_audio.shape, model_vocals.shape)
                self.assertEqual(mixture_audio.shape, model_vocals.shape)
                return vocals_audio, {"vocal_postprocess": "spectral_subtractive_vocal_denoise_v1"}

            with patch("worker.high_quality_stem_export._cleanup_single_vocal_stem") as cleanup_mock:
                cleanup_mock.side_effect = cleanup_side_effect
                manifest = export_vocal_preview_from_audio(
                    source_path,
                    output_dir,
                    candidate_mode="ft_vocals",
                    start_sec=0.0,
                    end_sec=duration_sec,
                    device="cpu",
                    subtractive_strength=1.2,
                    subtractive_power=1.3,
                    subtractive_residual_floor=0.12,
                )

            output_audio, output_sr = sf.read(output_dir / "vocals.flac", always_2d=True)
            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(output_sr, target_sample_rate)
            self.assertEqual(output_audio.shape[0], model_vocals.shape[0])
            self.assertAlmostEqual(saved_manifest["stems"]["vocals"]["duration_sec"], duration_sec, places=3)
            self.assertEqual(manifest["pipeline"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v1")
            cleanup_mock.assert_called_once()

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_vocal_preview_from_audio_ft_subtractive_v2_sets_postprocess(self, separate_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        mixture_mono = (voice + 0.8 * leak).astype(np.float32)
        stereo = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        def separate_side_effect(audio_path, *, model_name, device=None, **_kwargs):
            self.assertEqual(model_name, HYBRID_STAGE1_MODEL)
            audio, sr = sf.read(str(audio_path), always_2d=True)
            self.assertEqual(sr, sample_rate)
            return {"vocals": (audio * 0.8).astype(np.float32)}

        separate_mock.side_effect = separate_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "out"
            sf.write(source_path, stereo, sample_rate)

            manifest = export_vocal_preview_from_audio(
                source_path,
                output_dir,
                candidate_mode="ft_vocals",
                start_sec=0.0,
                end_sec=0.5,
                device="cpu",
                subtractive_strength=1.6,
                subtractive_power=1.5,
                subtractive_residual_floor=0.08,
                subtractive_active_restore=0.35,
                subtractive_target_rms_ratio=0.92,
                subtractive_max_gain=1.35,
            )

            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v2")
            self.assertEqual(saved_manifest["pipeline"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v2")
            self.assertEqual(saved_manifest["stems"]["vocals"]["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v2")

    def test_cleanup_single_vocal_stem_suppresses_tail_while_preserving_active_phrase(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        active_voice = (0.6 * np.sin(2 * np.pi * 220.0 * t) + 0.25 * np.sin(2 * np.pi * 4200.0 * t)).astype(np.float32)
        tail_leak = (0.08 * np.sin(2 * np.pi * 220.0 * t) + 0.12 * np.sin(2 * np.pi * 5200.0 * t)).astype(np.float32)
        accompaniment = (0.18 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)

        vocals_mono = np.concatenate(
            [
                active_voice[: sample_rate // 2],
                tail_leak[sample_rate // 2 :],
            ]
        ).astype(np.float32)
        mixture_mono = (vocals_mono + accompaniment).astype(np.float32)
        vocals = np.stack([vocals_mono, vocals_mono], axis=1).astype(np.float32)
        mixture = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        cleaned, debug = _cleanup_single_vocal_stem(
            vocals,
            mixture,
            sharpen_alpha=0.0,
            tail_gate_strength=1.0,
            tail_gate_floor_ratio=0.2,
            tail_gate_smoothing_frames=9,
        )

        def window_rms(audio: np.ndarray, start: int, end: int) -> float:
            mono = audio[start:end].mean(axis=1)
            return float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0

        active_start = int(0.15 * sample_rate)
        active_end = int(0.35 * sample_rate)
        tail_start = int(0.7 * sample_rate)
        tail_end = int(0.9 * sample_rate)

        self.assertGreater(window_rms(cleaned, active_start, active_end), 0.45 * window_rms(vocals, active_start, active_end))
        self.assertLess(window_rms(cleaned, tail_start, tail_end), 0.45 * window_rms(vocals, tail_start, tail_end))
        self.assertEqual(debug["vocal_postprocess"], "single_stem_cleanup_v1")

    def test_cleanup_single_vocal_stem_hard_mask_reduces_accompaniment_band(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        vocals_mono = (voice + 0.25 * leak).astype(np.float32)
        mixture_mono = (voice + 0.8 * leak).astype(np.float32)
        vocals = np.stack([vocals_mono, vocals_mono], axis=1).astype(np.float32)
        mixture = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        cleaned, debug = _cleanup_single_vocal_stem(
            vocals,
            mixture,
            hard_mask_threshold=0.6,
            hard_mask_power=3.0,
            hard_mask_residual_floor=0.02,
            dsp_device="cpu",
        )

        def band_rms(audio: np.ndarray, low_hz: float, high_hz: float) -> float:
            mono = audio.mean(axis=1)
            spectrum = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
            mask = (freqs >= low_hz) & (freqs < high_hz)
            values = spectrum[mask]
            return float(np.sqrt(np.mean(values * values))) if values.size else 0.0

        self.assertLess(band_rms(cleaned, 800.0, 950.0), 0.55 * band_rms(vocals, 800.0, 950.0))
        self.assertGreater(band_rms(cleaned, 180.0, 260.0), 0.45 * band_rms(vocals, 180.0, 260.0))
        self.assertEqual(debug["vocal_postprocess"], "hard_subtractive_vocal_mask_v1")

    def test_cleanup_single_vocal_stem_subtractive_reduces_accompaniment_band(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        vocals_mono = (voice + 0.35 * leak).astype(np.float32)
        mixture_mono = (voice + 0.9 * leak).astype(np.float32)
        vocals = np.stack([vocals_mono, vocals_mono], axis=1).astype(np.float32)
        mixture = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        cleaned, debug = _cleanup_single_vocal_stem(
            vocals,
            mixture,
            subtractive_strength=1.2,
            subtractive_power=1.3,
            subtractive_residual_floor=0.12,
            dsp_device="cpu",
        )

        def band_rms(audio: np.ndarray, low_hz: float, high_hz: float) -> float:
            mono = audio.mean(axis=1)
            spectrum = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
            mask = (freqs >= low_hz) & (freqs < high_hz)
            values = spectrum[mask]
            return float(np.sqrt(np.mean(values * values))) if values.size else 0.0

        self.assertLess(band_rms(cleaned, 800.0, 950.0), 0.7 * band_rms(vocals, 800.0, 950.0))
        self.assertGreater(band_rms(cleaned, 180.0, 260.0), 0.5 * band_rms(vocals, 180.0, 260.0))
        self.assertEqual(debug["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v1")

    def test_cleanup_single_vocal_stem_subtractive_v2_restores_level_on_active_voice(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        voice = (0.55 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        leak = (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        vocals_mono = (voice + 0.35 * leak).astype(np.float32)
        mixture_mono = (voice + 0.9 * leak).astype(np.float32)
        vocals = np.stack([vocals_mono, vocals_mono], axis=1).astype(np.float32)
        mixture = np.stack([mixture_mono, mixture_mono], axis=1).astype(np.float32)

        cleaned_v1, _ = _cleanup_single_vocal_stem(
            vocals,
            mixture,
            subtractive_strength=1.6,
            subtractive_power=1.5,
            subtractive_residual_floor=0.08,
            dsp_device="cpu",
        )
        cleaned_v2, debug = _cleanup_single_vocal_stem(
            vocals,
            mixture,
            subtractive_strength=1.6,
            subtractive_power=1.5,
            subtractive_residual_floor=0.08,
            subtractive_active_restore=0.35,
            subtractive_target_rms_ratio=0.92,
            subtractive_max_gain=1.35,
            dsp_device="cpu",
        )

        def rms(audio: np.ndarray) -> float:
            mono = audio.mean(axis=1)
            return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))

        def band_rms(audio: np.ndarray, low_hz: float, high_hz: float) -> float:
            mono = audio.mean(axis=1)
            spectrum = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
            mask = (freqs >= low_hz) & (freqs < high_hz)
            values = spectrum[mask]
            return float(np.sqrt(np.mean(values * values))) if values.size else 0.0

        self.assertGreater(rms(cleaned_v2), rms(cleaned_v1))
        self.assertGreater(band_rms(cleaned_v2, 180.0, 260.0), band_rms(cleaned_v1, 180.0, 260.0))
        self.assertLess(band_rms(cleaned_v2, 800.0, 950.0), 0.9 * band_rms(vocals, 800.0, 950.0))
        self.assertGreater(debug["subtractive_restore_mean"], 0.0)
        self.assertEqual(debug["vocal_postprocess"], "spectral_subtractive_vocal_denoise_v2")

    def test_rebalance_hybrid_stage2_stems_moves_harmonic_bleed_from_drums_to_other_family(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        melodic = 0.22 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        clicks = np.zeros_like(t)
        clicks[::2205] = 1.0
        percussive = clicks.astype(np.float32)

        drums = np.stack([percussive + melodic, percussive + melodic], axis=1).astype(np.float32)
        other = np.stack([0.03 * melodic, 0.03 * melodic], axis=1).astype(np.float32)
        guitar = np.stack([0.01 * melodic, 0.01 * melodic], axis=1).astype(np.float32)
        piano = np.stack([0.008 * melodic, 0.008 * melodic], axis=1).astype(np.float32)
        stems = {
            "drums": drums,
            "bass": np.zeros_like(drums),
            "other": other,
            "vocals": np.zeros_like(drums),
            "guitar": guitar,
            "piano": piano,
        }

        rebalanced = _rebalance_hybrid_stage2_stems(stems)
        drums_harmonic_before, _ = _split_harmonic_percussive(
            stems["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
        )
        drums_harmonic_after, _ = _split_harmonic_percussive(
            rebalanced["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
        )
        other_harmonic_before, _ = _split_harmonic_percussive(
            stems["other"],
            harmonic_margin=2.0,
            percussive_margin=1.0,
        )
        other_harmonic_after, _ = _split_harmonic_percussive(
            rebalanced["other"],
            harmonic_margin=2.0,
            percussive_margin=1.0,
        )

        def rms(audio: np.ndarray) -> float:
            mono = audio.mean(axis=1)
            return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))

        self.assertLess(rms(drums_harmonic_after), rms(drums_harmonic_before))
        self.assertGreater(rms(other_harmonic_after), rms(other_harmonic_before))

    def test_rebalance_hybrid_stage2_stems_moves_percussive_bleed_from_other_to_drums(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        melodic = 0.18 * np.sin(2 * np.pi * 392.0 * t).astype(np.float32)
        clicks = np.zeros_like(t)
        clicks[::2205] = 1.0
        percussive = (0.35 * clicks).astype(np.float32)

        drums = np.stack([0.08 * percussive + 0.02 * melodic, 0.08 * percussive + 0.02 * melodic], axis=1).astype(np.float32)
        other = np.stack([0.25 * melodic + percussive, 0.25 * melodic + percussive], axis=1).astype(np.float32)
        stems = {
            "drums": drums,
            "bass": np.zeros_like(drums),
            "other": other,
            "vocals": np.zeros_like(drums),
            "guitar": np.zeros_like(drums),
            "piano": np.zeros_like(drums),
        }

        rebalanced = _rebalance_hybrid_stage2_stems(stems)
        _, drums_percussive_before = _split_harmonic_percussive(
            stems["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
        )
        _, drums_percussive_after = _split_harmonic_percussive(
            rebalanced["drums"],
            harmonic_margin=1.0,
            percussive_margin=4.0,
        )
        _, other_percussive_before = _split_harmonic_percussive(
            stems["other"],
            harmonic_margin=2.0,
            percussive_margin=1.0,
        )
        _, other_percussive_after = _split_harmonic_percussive(
            rebalanced["other"],
            harmonic_margin=2.0,
            percussive_margin=1.0,
        )

        def rms(audio: np.ndarray) -> float:
            mono = audio.mean(axis=1)
            return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))

        self.assertGreater(rms(drums_percussive_after), rms(drums_percussive_before))
        self.assertLess(rms(other_percussive_after), rms(other_percussive_before))

    def test_component_transfer_envelope_tracks_primary_dominance(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        lead = np.concatenate([
            np.sin(2 * np.pi * 220.0 * t[: sample_rate // 2]).astype(np.float32),
            np.zeros(sample_rate - sample_rate // 2, dtype=np.float32),
        ])
        bed = np.concatenate([
            np.zeros(sample_rate // 2, dtype=np.float32),
            0.4 * np.sin(2 * np.pi * 220.0 * t[sample_rate // 2 :]).astype(np.float32),
        ])
        envelope = _component_transfer_envelope(
            np.stack([lead, lead], axis=1).astype(np.float32),
            np.stack([bed, bed], axis=1).astype(np.float32),
        )
        first_half = float(np.mean(envelope[: sample_rate // 2, 0]))
        second_half = float(np.mean(envelope[sample_rate // 2 :, 0]))
        self.assertGreater(first_half, 0.5)
        self.assertLess(second_half, 0.15)

    def test_extract_melody_stem_from_stems_returns_non_silent_output_for_tonal_mix(self):
        sample_rate = 44100
        t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
        lead = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        harmony = 0.4 * np.sin(2 * np.pi * 659.25 * t).astype(np.float32)
        vocals = 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        stems = {
            "other": np.stack([lead + harmony, lead + harmony], axis=1).astype(np.float32),
            "vocals": np.stack([vocals, vocals], axis=1).astype(np.float32),
        }

        melody, debug = extract_melody_stem_from_stems(stems, sample_rate=sample_rate)

        mono = melody.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))
        self.assertGreater(rms, 0.01)
        self.assertGreater(debug["active_frame_count"], 0)
        self.assertGreater(debug["voiced_ratio"], 0.05)

    def test_redistribute_instrumental_melody_from_stems_reduces_lead_bleed_from_drums_and_other(self):
        sample_rate = 44100
        t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
        lead = 0.45 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        harmony = 0.25 * np.sin(2 * np.pi * 659.25 * t).astype(np.float32)
        drums_hit = np.zeros_like(t, dtype=np.float32)
        drums_hit[::2205] = 1.0
        drums_hit = np.convolve(drums_hit, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)
        stems = {
            "drums": np.stack([drums_hit + 0.3 * lead, drums_hit + 0.3 * lead], axis=1).astype(np.float32),
            "other": np.stack([harmony + lead, harmony + lead], axis=1).astype(np.float32),
            "guitar": np.stack([0.2 * lead, 0.2 * lead], axis=1).astype(np.float32),
            "piano": np.stack([0.15 * lead, 0.15 * lead], axis=1).astype(np.float32),
            "vocals": np.zeros((sample_rate * 2, 2), dtype=np.float32),
        }

        rebalanced, melody, debug = _redistribute_instrumental_melody_from_stems(stems)

        def cosine(left: np.ndarray, right: np.ndarray) -> float:
            num = float(np.dot(left, right))
            den = float(np.linalg.norm(left) * np.linalg.norm(right))
            return num / den if den > 1e-8 else 0.0

        before_drums = cosine(stems["drums"][:, 0], lead)
        after_drums = cosine(rebalanced["drums"][:, 0], lead)
        before_other = cosine(stems["other"][:, 0], lead)
        after_other = cosine(rebalanced["other"][:, 0], lead)
        melody_rms = float(np.sqrt(np.mean(np.square(melody.mean(axis=1), dtype=np.float32))))

        self.assertLess(after_drums, before_drums)
        self.assertLess(after_other, before_other)
        self.assertGreater(melody_rms, 0.01)
        self.assertEqual(debug["melody_strategy"], "instrumental_mask_reassign_v1")

    def test_aggressive_melody_first_rebalance_creates_stronger_melody_and_cleans_drums_other(self):
        sample_rate = 44100
        t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
        lead = 0.45 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        harmony = 0.25 * np.sin(2 * np.pi * 659.25 * t).astype(np.float32)
        kick = np.zeros_like(t, dtype=np.float32)
        kick[::2205] = 1.0
        kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)
        stems = {
            "drums": np.stack([kick + 0.35 * lead, kick + 0.35 * lead], axis=1).astype(np.float32),
            "other": np.stack([lead + harmony + 0.35 * kick, lead + harmony + 0.35 * kick], axis=1).astype(np.float32),
            "guitar": np.stack([0.25 * lead, 0.25 * lead], axis=1).astype(np.float32),
            "piano": np.stack([0.18 * harmony, 0.18 * harmony], axis=1).astype(np.float32),
            "vocals": np.zeros((sample_rate * 2, 2), dtype=np.float32),
            "bass": np.zeros((sample_rate * 2, 2), dtype=np.float32),
        }

        rebalanced, melody, debug = _aggressive_melody_first_rebalance(stems)

        def rms(audio: np.ndarray) -> float:
            mono = audio.mean(axis=1)
            return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))

        drums_h_before, _ = _split_harmonic_percussive(stems["drums"], harmonic_margin=1.0, percussive_margin=4.0)
        drums_h_after, _ = _split_harmonic_percussive(rebalanced["drums"], harmonic_margin=1.0, percussive_margin=4.0)
        other_h_before, _ = _split_harmonic_percussive(stems["other"], harmonic_margin=2.2, percussive_margin=1.0)
        other_h_after, _ = _split_harmonic_percussive(rebalanced["other"], harmonic_margin=2.2, percussive_margin=1.0)

        self.assertLess(rms(drums_h_after), rms(drums_h_before))
        self.assertLess(rms(other_h_after), rms(other_h_before))
        self.assertGreater(rms(melody), 0.03)
        self.assertEqual(debug["melody_strategy"], "aggressive_harmonic_bus_reassign_v1")
        self.assertGreater(debug["drums_harmonic_moved_rms"], 0.01)
        self.assertGreater(debug["other_harmonic_moved_rms"], 0.01)

    def test_sharpen_stem_bleed_reduces_cross_talk_for_simple_mix(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        target = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        leak = np.sin(2 * np.pi * 660.0 * t).astype(np.float32)

        stems = {
            "vocals": np.stack([target + 0.35 * leak, target + 0.35 * leak], axis=1).astype(np.float32),
            "other": np.stack([leak + 0.35 * target, leak + 0.35 * target], axis=1).astype(np.float32),
        }
        sharpened = sharpen_stem_bleed(stems, alpha=2.0, n_fft=1024, hop_length=256)

        def cosine(left: np.ndarray, right: np.ndarray) -> float:
            num = float(np.dot(left, right))
            den = float(np.linalg.norm(left) * np.linalg.norm(right))
            return num / den if den > 1e-8 else 0.0

        before = cosine(stems["vocals"][:, 0], stems["other"][:, 0])
        after = cosine(sharpened["vocals"][:, 0], sharpened["other"][:, 0])
        self.assertLess(after, before)

    @patch("worker.high_quality_stem_export._redistribute_instrumental_melody_from_stems")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_full_res_writes_flacs_and_manifest(self, separate_mock, melody_rebalance_mock):
        separate_mock.return_value = {
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        melody_rebalance_mock.return_value = (
            separate_mock.return_value,
            np.full((44100, 2), 0.125, dtype=np.float32),
            {
                "melody_bus_weights": {"other": 1.0},
                "voiced_ratio": 0.75,
                "mean_confidence": 0.6,
                "median_frequency_hz": 440.0,
                "active_frame_count": 80,
                "melody_strategy": "instrumental_mask_reassign_v1",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_track_stems_full_res(
                "audio/example.mp3",
                temp_dir,
                device="cuda",
                sharpen_bleed=True,
                include_melody_stem=True,
            )
            manifest_path = Path(temp_dir) / "stem_manifest.json"
            bass_path = Path(temp_dir) / "bass.flac"
            other_path = Path(temp_dir) / "other.flac"
            melody_path = Path(temp_dir) / "melody.flac"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(bass_path.exists())
            self.assertTrue(other_path.exists())
            self.assertTrue(melody_path.exists())

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sample_rate"], 44100)
            self.assertEqual(saved["export_subtype"], "PCM_24")
            self.assertTrue(saved["bleed_sharpening"])
            self.assertTrue(saved["melody_stem_included"])
            self.assertEqual(saved["pipeline"]["mode"], HYBRID_STEM_EXPORT_MODEL)
            self.assertEqual(saved["pipeline"]["vocals_source"], "htdemucs_ft+bs_roformer_vocals_other")
            self.assertEqual(saved["pipeline"]["instrumental_strategy"], "stage1_non_vocal_sum")
            self.assertEqual(saved["pipeline"]["postprocess"], "harmonic_percussive_rebalance_v1")
            self.assertEqual(saved["pipeline"]["vocal_fusion"], "spectral_split_bs_low_ft_high_v2")
            self.assertEqual(saved["pipeline"]["melody_strategy"], "instrumental_mask_reassign_v1")
            self.assertEqual(saved["stems"]["bass"]["channels"], 2)
            self.assertEqual(saved["stems"]["other"]["duration_sec"], 1.0)
            self.assertEqual(saved["stems"]["melody"]["median_frequency_hz"], 440.0)

            audio, sample_rate = sf.read(str(other_path), always_2d=True)
            self.assertEqual(sample_rate, 44100)
            self.assertEqual(audio.shape[1], 2)
            self.assertEqual(manifest["stems"]["other"]["channels"], 2)
            self.assertEqual(manifest["stems"]["melody"]["active_frame_count"], 80)
            self.assertEqual(separate_mock.call_args.kwargs["model_name"], DEFAULT_STEM_EXPORT_MODEL)

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    @patch("worker.high_quality_stem_export.verify_gpu_inference_device", create=True)
    def test_export_track_stems_full_res_records_gpu_execution_contract(self, cuda_mock, separate_mock):
        cuda_mock.return_value = {
            "cuda_available": True,
            "requested_device": "cuda",
            "resolved_device": "cuda",
            "device_count": 1,
            "device_name": "Unit Test GPU",
        }
        separate_mock.return_value = {
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.125, dtype=np.float32),
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_track_stems_full_res(
                "audio/example.mp3",
                temp_dir,
                device="cuda",
                include_melody_stem=False,
            )
            saved = json.loads((Path(temp_dir) / "stem_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["execution_contract"]["expected_acceleration_mode"], "gpu_inference_then_cpu_dsp_scoring")
        self.assertEqual(manifest["execution_contract"]["gpu_inference_required"], True)
        self.assertEqual(manifest["execution_contract"]["gpu_inference_skipped"], False)
        self.assertEqual(manifest["execution_contract"]["cuda_preflight"]["device_name"], "Unit Test GPU")
        self.assertEqual(saved["execution_contract"], manifest["execution_contract"])
        cuda_mock.assert_called_once_with("cuda")

    @patch("worker.high_quality_stem_export.torch.cuda.is_available", return_value=False)
    def test_verify_gpu_inference_device_rejects_cuda_when_unavailable(self, _cuda_available_mock):
        with self.assertRaisesRegex(RuntimeError, "CUDA inference required"):
            hq_export.verify_gpu_inference_device("cuda")

    def test_bs_roformer_stage_fails_closed_instead_of_allowing_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "repo"
            config_path = root / "config.yaml"
            checkpoint_path = root / "model.ckpt"
            source_path = root / "source.wav"
            repo_path.mkdir()
            (repo_path / "inference.py").write_text("# fake inference", encoding="utf-8")
            config_path.write_text("inference:\n  num_overlap: 4\n", encoding="utf-8")
            checkpoint_path.write_bytes(b"fake")
            sf.write(source_path, np.zeros((128, 2), dtype=np.float32), hq_export.SEPARATION_SAMPLE_RATE)

            executed_commands: list[list[str]] = []

            def fake_run(command, *, cwd, capture_output, text, check):
                executed_commands.append([str(part) for part in command])
                store_dir = Path(command[command.index("--store_dir") + 1])
                audio = np.zeros((128, 2), dtype=np.float32)
                sf.write(store_dir / "vocals.flac", audio, hq_export.SEPARATION_SAMPLE_RATE)
                sf.write(store_dir / "instrumental.flac", audio, hq_export.SEPARATION_SAMPLE_RATE)

                class Completed:
                    stdout = ""
                    stderr = ""

                return Completed()

            with patch("worker.high_quality_stem_export.resolve_torch_device", return_value="cuda") as resolve_mock:
                with patch("worker.high_quality_stem_export.subprocess.run", side_effect=fake_run):
                    hq_export._load_bs_roformer_vocals_other(
                        source_path,
                        device="cuda",
                        bs_roformer_runtime={
                            "repoPath": str(repo_path),
                            "configPath": str(config_path),
                            "checkpointPath": str(checkpoint_path),
                            "requestedDevice": "cuda",
                            "deviceIds": ["0"],
                            "forceCpu": False,
                            "extractInstrumental": True,
                            "useTta": False,
                            "disableDetailedPbar": True,
                            "filenameTemplate": "{instrument}",
                            "configOverrideYaml": "",
                            "extraCliArgs": [],
                        },
                    )

            self.assertTrue(executed_commands)
            resolve_mock.assert_called_once()
            self.assertEqual(resolve_mock.call_args.kwargs["allow_cpu"], False)
            self.assertIn("--device_ids", executed_commands[0])
            self.assertNotIn("--force_cpu", executed_commands[0])

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_full_res_defaults_to_stem_parser_profile_with_melody(self, separate_mock):
        separate_mock.return_value = {
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.125, dtype=np.float32),
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_track_stems_full_res("audio/example.mp3", temp_dir, device="cuda")
            manifest_path = Path(temp_dir) / "stem_manifest.json"
            melody_path = Path(temp_dir) / "melody.flac"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(melody_path.exists())
            self.assertEqual(manifest["export_subtype"], "PCM_24")
            self.assertTrue(manifest["melody_stem_included"])
            self.assertIn("melody", manifest["stems"])
            self.assertEqual(manifest["pipeline"]["stage1_model"], HYBRID_STAGE1_MODEL)
            self.assertEqual(manifest["pipeline"]["stage2_model"], HYBRID_STAGE2_MODEL)
            self.assertEqual(manifest["pipeline"]["vocals_source"], "htdemucs_ft+bs_roformer_vocals_other")
            self.assertEqual(manifest["pipeline"]["instrumental_strategy"], "stage1_non_vocal_sum")
            self.assertEqual(manifest["pipeline"]["postprocess"], "harmonic_percussive_rebalance_v1")
            self.assertEqual(manifest["pipeline"]["vocal_fusion"], "spectral_split_bs_low_ft_high_v2")
            self.assertEqual(manifest["pipeline"]["melody_strategy"], "instrumental_mask_reassign_v1")
            self.assertTrue(json.loads(manifest_path.read_text(encoding="utf-8"))["melody_stem_included"])
            self.assertEqual(manifest["contract"]["target_reference_assisted"], False)
            self.assertEqual(manifest["contract"]["target_reference_used_at_application"], False)
            self.assertEqual(manifest["contract"]["application_reference_free"], True)
            self.assertEqual(manifest["contract"]["delivery_allowed_without_reference"], True)
            self.assertEqual(manifest["contract"]["fallback_used"], False)
            self.assertEqual(manifest["contract"]["per_track_hardcode_allowed"], False)
            self.assertEqual(manifest["contract"]["not_timestamp_hardcoded"], True)
            self.assertEqual(separate_mock.call_args.kwargs["model_name"], DEFAULT_STEM_EXPORT_MODEL)

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_aggressive_melody_first_writes_no_reference_contract(self, separate_mock):
        separate_mock.return_value = {
            "bass": np.zeros((44100, 2), dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.zeros((44100, 2), dtype=np.float32),
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_track_stems_aggressive_melody_first(
                "audio/example.mp3",
                temp_dir,
                model_name="htdemucs_6s",
                device="cuda",
            )
            saved_manifest = json.loads((Path(temp_dir) / "stem_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_manifest["pipeline"]["mode"], "aggressive_melody_first")
        self.assertEqual(manifest["contract"]["target_reference_assisted"], False)
        self.assertEqual(manifest["contract"]["target_reference_used_at_application"], False)
        self.assertEqual(manifest["contract"]["application_reference_free"], True)
        self.assertEqual(manifest["contract"]["delivery_allowed_without_reference"], True)
        self.assertEqual(manifest["contract"]["fallback_used"], False)
        self.assertEqual(manifest["contract"]["per_track_hardcode_allowed"], False)
        self.assertEqual(manifest["contract"]["not_timestamp_hardcoded"], True)
        self.assertEqual(saved_manifest["contract"], manifest["contract"])

    @patch("builtins.print")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_full_res_emits_progress_messages(self, separate_mock, print_mock):
        separate_mock.return_value = {
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.125, dtype=np.float32),
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            export_track_stems_full_res("audio/example.mp3", temp_dir, device="cuda")

        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("ENTER export_track_stems_full_res", printed)
        self.assertIn("writing_stem stem=vocals", printed)
        self.assertIn("EXIT export_track_stems_full_res", printed)

    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_full_res_records_bs_pipeline_manifest(self, separate_mock):
        separate_mock.return_value = {
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.125, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = export_track_stems_full_res(
                "audio/example.mp3",
                temp_dir,
                device="cuda",
                model_name=HYBRID_BS_STEM_EXPORT_MODEL,
            )
        self.assertEqual(manifest["pipeline"]["mode"], HYBRID_BS_STEM_EXPORT_MODEL)
        self.assertEqual(manifest["pipeline"]["vocals_source"], "htdemucs_ft+bs_roformer_vocals_other")
        self.assertEqual(manifest["pipeline"]["accompaniment_source"], "bs_roformer_vocals_other")
        self.assertEqual(manifest["pipeline"]["instrumental_strategy"], "bs_roformer_instrumental")
        self.assertEqual(manifest["pipeline"]["vocal_fusion"], "spectral_split_bs_low_ft_high_v2")
        self.assertEqual(manifest["pipeline"]["melody_strategy"], "instrumental_mask_reassign_v1")

    @patch("worker.high_quality_stem_export._aggressive_melody_first_rebalance")
    @patch("worker.high_quality_stem_export.separate_track_stems_full_res")
    def test_export_track_stems_aggressive_melody_first_writes_manifest_and_melody(self, separate_mock, aggressive_mock):
        separate_mock.return_value = {
            "drums": np.full((44100, 2), 0.2, dtype=np.float32),
            "bass": np.full((44100, 2), 0.25, dtype=np.float32),
            "other": np.full((44100, 2), 0.5, dtype=np.float32),
            "vocals": np.full((44100, 2), 0.125, dtype=np.float32),
            "guitar": np.full((44100, 2), 0.01, dtype=np.float32),
            "piano": np.full((44100, 2), 0.005, dtype=np.float32),
        }
        aggressive_mock.return_value = (
            separate_mock.return_value,
            np.full((44100, 2), 0.15, dtype=np.float32),
            {
                "melody_strategy": "aggressive_harmonic_bus_reassign_v1",
                "melody_rms": 0.15,
                "drums_harmonic_moved_rms": 0.05,
                "other_harmonic_moved_rms": 0.08,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            melody_path = Path(temp_dir) / "melody.flac"
            manifest_path = Path(temp_dir) / "stem_manifest.json"
            manifest = export_track_stems_aggressive_melody_first(
                "audio/example.mp3",
                temp_dir,
                device="cuda",
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(melody_path.exists())
            self.assertTrue(manifest_path.exists())

        self.assertEqual(manifest["pipeline"]["mode"], "aggressive_melody_first")
        self.assertEqual(saved["pipeline"]["postprocess"], "aggressive_harmonic_bus_reassign_v1")
        self.assertEqual(saved["stems"]["melody"]["melody_strategy"], "aggressive_harmonic_bus_reassign_v1")

    def test_export_aggressive_melody_first_from_stem_dir_preserves_vocals_exact(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        vocals = np.stack([
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
        ], axis=1).astype(np.float32)
        kick = np.zeros_like(t, dtype=np.float32)
        kick[::2205] = 1.0
        kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)
        other = np.stack([
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
        ], axis=1).astype(np.float32)
        drums = np.stack([kick, kick], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "out"
            source_vocals_path = source_dir / "vocals.flac"
            source_dir.mkdir(parents=True, exist_ok=True)
            sf.write(source_vocals_path, vocals, sample_rate)
            sf.write(source_dir / "drums.flac", drums, sample_rate)
            sf.write(source_dir / "other.flac", other, sample_rate)
            sf.write(source_dir / "bass.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "guitar.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "piano.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            (source_dir / "stem_manifest.json").write_text(
                json.dumps({"audio_path": "audio/example.mp3", "model_name": "htdemucs_6s", "bleed_sharpening": True}),
                encoding="utf-8",
            )

            manifest = export_aggressive_melody_first_from_stem_dir(source_dir, output_dir)
            output_vocals_path = output_dir / "vocals.flac"
            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(source_vocals_path.read_bytes(), output_vocals_path.read_bytes())
        self.assertEqual(saved_manifest["pipeline"]["vocals_strategy"], "preserve_source_stem_exact")
        self.assertEqual(manifest["pipeline"]["mode"], "aggressive_melody_first_from_stem_dir")

    def test_export_aggressive_melody_first_from_stem_dir_preview_writes_only_requested_window(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        vocals = np.stack([
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
        ], axis=1).astype(np.float32)
        kick = np.zeros_like(t, dtype=np.float32)
        kick[::2205] = 1.0
        kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)
        other = np.stack([
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
        ], axis=1).astype(np.float32)
        drums = np.stack([kick, kick], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "out"
            source_dir.mkdir(parents=True, exist_ok=True)
            sf.write(source_dir / "vocals.flac", vocals, sample_rate)
            sf.write(source_dir / "drums.flac", drums, sample_rate)
            sf.write(source_dir / "other.flac", other, sample_rate)
            sf.write(source_dir / "bass.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "guitar.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "piano.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            (source_dir / "stem_manifest.json").write_text(
                json.dumps({"audio_path": "audio/example.mp3", "model_name": "htdemucs_6s", "bleed_sharpening": True}),
                encoding="utf-8",
            )

            manifest = export_aggressive_melody_first_from_stem_dir(
                source_dir,
                output_dir,
                start_sec=0.2,
                end_sec=0.6,
                other_move=0.2,
                drums_masked_move=0.05,
                target_melody_rms=0.015,
                max_melody_gain=1.2,
                harmonic_count=1,
                confidence_floor=0.5,
                dsp_device="cpu",
            )
            saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            output_vocals, out_sr = sf.read(output_dir / "vocals.flac", always_2d=True)

        self.assertEqual(out_sr, sample_rate)
        self.assertAlmostEqual(saved_manifest["stems"]["vocals"]["duration_sec"], 0.4, places=3)
        self.assertEqual(saved_manifest["preview_window_sec"]["start"], 0.2)
        self.assertEqual(saved_manifest["preview_window_sec"]["end"], 0.6)
        self.assertEqual(saved_manifest["tuning"]["other_move"], 0.2)
        self.assertEqual(saved_manifest["tuning"]["drums_masked_move"], 0.05)
        self.assertEqual(saved_manifest["tuning"]["harmonic_count"], 1)
        self.assertEqual(saved_manifest["tuning"]["confidence_floor"], 0.5)
        self.assertEqual(saved_manifest["tuning"]["dsp_device"], "cpu")
        self.assertEqual(saved_manifest["stems"]["melody"]["other_move"], 0.2)
        self.assertEqual(saved_manifest["stems"]["melody"]["drums_masked_move"], 0.05)
        self.assertEqual(saved_manifest["stems"]["melody"]["harmonic_count"], 1)
        self.assertEqual(saved_manifest["stems"]["melody"]["confidence_floor"], 0.5)
        self.assertEqual(saved_manifest["stems"]["melody"]["dsp_device"], "cpu")
        np.testing.assert_allclose(output_vocals, vocals[int(0.2 * sample_rate):int(0.6 * sample_rate)], atol=5e-5)
        self.assertEqual(manifest["pipeline"]["vocals_strategy"], "preserve_source_stem_exact")

    @patch("builtins.print")
    def test_export_aggressive_melody_first_from_stem_dir_emits_preview_progress_logs(self, print_mock):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        vocals = np.stack([
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
            0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32),
        ], axis=1).astype(np.float32)
        kick = np.zeros_like(t, dtype=np.float32)
        kick[::2205] = 1.0
        kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)
        other = np.stack([
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
            0.25 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2 * kick,
        ], axis=1).astype(np.float32)
        drums = np.stack([kick, kick], axis=1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "out"
            source_dir.mkdir(parents=True, exist_ok=True)
            sf.write(source_dir / "vocals.flac", vocals, sample_rate)
            sf.write(source_dir / "drums.flac", drums, sample_rate)
            sf.write(source_dir / "other.flac", other, sample_rate)
            sf.write(source_dir / "bass.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "guitar.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "piano.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            (source_dir / "stem_manifest.json").write_text(
                json.dumps({"audio_path": "audio/example.mp3", "model_name": "htdemucs_6s", "bleed_sharpening": True}),
                encoding="utf-8",
            )

            export_aggressive_melody_first_from_stem_dir(
                source_dir,
                output_dir,
                start_sec=0.0,
                end_sec=0.5,
            )

        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("ENTER export_aggressive_melody_first_from_stem_dir", printed)
        self.assertIn("preview_window_resolved start_sec=0.0 end_sec=0.5", printed)
        self.assertIn("ENTER selective_melody_first_rebalance_from_stem_dir", printed)
        self.assertIn("writing_stem_dir_preview stem=melody", printed)
        self.assertIn("EXIT export_aggressive_melody_first_from_stem_dir", printed)

    def test_repair_exported_stem_directory_writes_repaired_stems_and_manifest(self):
        sample_rate = 44100
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        lead = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        kick = np.zeros_like(t, dtype=np.float32)
        kick[::2205] = 1.0
        kick = np.convolve(kick, np.hanning(96).astype(np.float32), mode="same").astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "repaired"
            source_dir.mkdir(parents=True, exist_ok=True)
            sf.write(source_dir / "drums.flac", np.stack([kick + 0.2 * lead, kick + 0.2 * lead], axis=1), sample_rate)
            sf.write(source_dir / "other.flac", np.stack([lead, lead], axis=1), sample_rate)
            sf.write(source_dir / "bass.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            sf.write(source_dir / "vocals.flac", np.zeros((sample_rate, 2), dtype=np.float32), sample_rate)
            (source_dir / "stem_manifest.json").write_text(
                json.dumps({"audio_path": "audio/example.mp3", "model_name": HYBRID_STEM_EXPORT_MODEL, "pipeline": {}}),
                encoding="utf-8",
            )

            manifest = repair_exported_stem_directory(source_dir, output_dir)
            repaired_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "melody.flac").exists())
            self.assertTrue((output_dir / "drums.flac").exists())
            self.assertTrue((output_dir / "other.flac").exists())
            self.assertTrue(manifest["melody_stem_included"])
            self.assertEqual(repaired_manifest["pipeline"]["melody_strategy"], "instrumental_mask_reassign_v1")
            self.assertIn("melody", repaired_manifest["stems"])


if __name__ == "__main__":
    unittest.main()
