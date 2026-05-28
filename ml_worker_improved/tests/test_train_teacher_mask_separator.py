import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_teacher_mask_separator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_teacher_mask_separator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stereo(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal, signal], axis=1).astype(np.float32)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(reference.shape[0], estimate.shape[0])
    ref = reference[:length].astype(np.float64)
    est = estimate[:length].astype(np.float64)
    err = ref - est
    return float(10.0 * np.log10((np.mean(ref**2) + 1.0e-12) / (np.mean(err**2) + 1.0e-12)))


def test_teacher_mask_training_is_reference_free_at_application_and_improves_synthetic_mix(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.38 * np.sin(2.0 * np.pi * 660.0 * time))
    interference = _stereo(0.30 * np.sin(2.0 * np.pi * 130.0 * time))
    source_mix = target + interference
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=96,
            epochs=80,
            batch_size=64,
            learning_rate=0.01,
            seed=123,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    assert _sdr(target, output_audio) > _sdr(target, source_mix) + 8.0
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["profile_source_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["requires_holdout_validation"] is True
    assert manifest["contract"]["stores_timestamp_rules"] is False


def test_teacher_mask_training_does_not_collapse_weak_target_to_silence(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    weak_target = _stereo(0.025 * np.sin(2.0 * np.pi * 930.0 * time))
    strong_interference = _stereo(0.55 * np.sin(2.0 * np.pi * 130.0 * time))
    source_mix = weak_target + strong_interference
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, weak_target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=96,
            epochs=90,
            batch_size=64,
            learning_rate=0.01,
            activity_weight=16.0,
            seed=456,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    output_rms = float(np.sqrt(np.mean(np.square(output_audio), dtype=np.float64)))
    target_rms = float(np.sqrt(np.mean(np.square(weak_target), dtype=np.float64)))
    assert output_rms > target_rms * 0.35
    assert _sdr(weak_target, output_audio) > _sdr(weak_target, source_mix) + 6.0


def test_teacher_mask_training_balances_sparse_target_frames(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 3, dtype=np.float64) / float(sample_rate)
    target_envelope = np.zeros_like(time)
    for start_sec in (0.4, 1.3, 2.2):
        start = int(start_sec * sample_rate)
        target_envelope[start : start + int(0.18 * sample_rate)] = 1.0
    sparse_target = _stereo(0.018 * np.sin(2.0 * np.pi * 930.0 * time) * target_envelope)
    dense_interference = _stereo(
        0.45 * np.sin(2.0 * np.pi * 140.0 * time)
        + 0.30 * np.sin(2.0 * np.pi * 440.0 * time)
        + 0.20 * np.sin(2.0 * np.pi * 1760.0 * time)
    )
    source_mix = sparse_target + dense_interference
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, sparse_target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=128,
            epochs=100,
            batch_size=64,
            learning_rate=0.008,
            activity_weight=28.0,
            active_frame_oversample=3,
            seed=987,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    output_rms = float(np.sqrt(np.mean(np.square(output_audio), dtype=np.float64)))
    target_rms = float(np.sqrt(np.mean(np.square(sparse_target), dtype=np.float64)))
    assert output_rms > target_rms * 0.25
    assert _sdr(sparse_target, output_audio) > _sdr(sparse_target, source_mix) + 8.0


def test_teacher_mask_training_uses_local_context_without_timestamp_rules(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 2, dtype=np.float64) / float(sample_rate)
    context_envelope = np.zeros_like(time)
    for start_sec in (0.35, 1.25):
        start = int(start_sec * sample_rate)
        ramp = np.hanning(int(0.42 * sample_rate))
        context_envelope[start : start + ramp.shape[0]] = ramp
    target = _stereo(0.035 * np.sin(2.0 * np.pi * 980.0 * time) * context_envelope)
    interference = _stereo(
        0.34 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.22 * np.sin(2.0 * np.pi * 510.0 * time)
        + 0.12 * np.sin(2.0 * np.pi * 1320.0 * time)
    )
    source_mix = target + interference
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=128,
            epochs=90,
            batch_size=64,
            learning_rate=0.008,
            activity_weight=24.0,
            active_frame_oversample=4,
            context_frame_radius=2,
            seed=654,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    assert _sdr(target, output_audio) > _sdr(target, source_mix) + 8.0
    assert manifest["training"]["config"]["context_frame_radius"] == 2
    assert manifest["contract"]["stores_timestamp_rules"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False


def test_teacher_mask_training_preserves_very_weak_sparse_targets_with_context(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 3, dtype=np.float64) / float(sample_rate)
    target_envelope = np.zeros_like(time)
    for start_sec in (0.55, 1.65, 2.35):
        start = int(start_sec * sample_rate)
        target_envelope[start : start + int(0.16 * sample_rate)] = np.hanning(int(0.16 * sample_rate))
    very_weak_target = _stereo(0.006 * np.sin(2.0 * np.pi * 1120.0 * time) * target_envelope)
    strong_mix_bed = _stereo(
        0.46 * np.sin(2.0 * np.pi * 150.0 * time)
        + 0.30 * np.sin(2.0 * np.pi * 410.0 * time)
        + 0.18 * np.sin(2.0 * np.pi * 1670.0 * time)
    )
    source_mix = very_weak_target + strong_mix_bed
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, very_weak_target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=128,
            epochs=100,
            batch_size=64,
            learning_rate=0.008,
            activity_weight=32.0,
            magnitude_loss_weight=0.75,
            active_frame_oversample=6,
            context_frame_radius=3,
            seed=321,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    output_rms = float(np.sqrt(np.mean(np.square(output_audio), dtype=np.float64)))
    target_rms = float(np.sqrt(np.mean(np.square(very_weak_target), dtype=np.float64)))
    assert output_rms > target_rms * 0.20
    assert manifest["training"]["diagnostics"]["active_frame_count"] < manifest["training"]["diagnostics"]["frame_count"]
    assert manifest["training"]["config"]["leakage_suppression_weight"] < manifest["training"]["config"]["activity_weight"]
    assert manifest["application"]["diagnostics"]["predicted_mask_mean"] > 1.0e-7


def test_teacher_mask_training_can_use_source_feature_memory_without_timestamp_rules(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate * 3, dtype=np.float64) / float(sample_rate)
    target_envelope = np.zeros_like(time)
    for start_sec in (0.45, 1.35, 2.20):
        start = int(start_sec * sample_rate)
        target_envelope[start : start + int(0.18 * sample_rate)] = np.hanning(int(0.18 * sample_rate))
    very_weak_target = _stereo(0.005 * np.sin(2.0 * np.pi * 1180.0 * time) * target_envelope)
    dense_bed = _stereo(
        0.50 * np.sin(2.0 * np.pi * 130.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 470.0 * time)
        + 0.16 * np.sin(2.0 * np.pi * 1730.0 * time)
    )
    source_mix = very_weak_target + dense_bed
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, very_weak_target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=96,
            epochs=20,
            batch_size=64,
            learning_rate=0.008,
            activity_weight=32.0,
            active_frame_oversample=6,
            context_frame_radius=2,
            source_feature_memory_neighbors=1,
            source_feature_memory_chunk_size=128,
            seed=2468,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    output_rms = float(np.sqrt(np.mean(np.square(output_audio), dtype=np.float64)))
    target_rms = float(np.sqrt(np.mean(np.square(very_weak_target), dtype=np.float64)))
    assert output_rms > target_rms * 0.75
    assert _sdr(very_weak_target, output_audio) > _sdr(very_weak_target, source_mix) + 12.0
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["stores_timestamp_rules"] is False
    assert manifest["contract"]["stores_learned_source_feature_memory"] is True
    assert manifest["contract"]["source_feature_memory_generalization_claim"] is False
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["application"]["diagnostics"]["source_feature_memory_used"] is True
    assert manifest["application"]["diagnostics"]["source_feature_memory_exact_source_match"] is True
    assert manifest["application"]["diagnostics"]["source_feature_memory_neighbor_count"] == 1


def test_teacher_mask_training_can_use_source_feature_complex_memory_without_timestamp_rules(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.025 * np.sin(2.0 * np.pi * 970.0 * time + 0.4))
    interference = _stereo(0.42 * np.sin(2.0 * np.pi * 220.0 * time) + 0.22 * np.sin(2.0 * np.pi * 1460.0 * time))
    source_mix = target + interference
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=64,
            epochs=8,
            batch_size=64,
            source_feature_memory_neighbors=1,
            source_feature_memory_store_complex_masks=True,
            source_feature_memory_chunk_size=128,
            seed=1357,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    assert _sdr(target, output_audio) > 35.0
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["stores_timestamp_rules"] is False
    assert manifest["contract"]["stores_learned_temporal_position_features"] is False
    assert manifest["contract"]["stores_learned_source_feature_memory"] is True
    assert manifest["contract"]["stores_learned_source_feature_complex_memory"] is True
    assert manifest["contract"]["target_reference_reconstructive_prior"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["application"]["diagnostics"]["source_feature_memory_exact_source_match"] is True
    assert manifest["application"]["diagnostics"]["source_feature_memory_complex_mask_used"] is True
    assert manifest["application"]["diagnostics"]["temporal_complex_mask_prior_used"] is False


def test_teacher_mask_checkpoint_persists_source_feature_memory(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.02 * np.sin(2.0 * np.pi * 900.0 * time))
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, target + _stereo(0.3 * np.sin(2.0 * np.pi * 180.0 * time)), sample_rate)
    _write_audio(target_path, target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=4,
            batch_size=64,
            source_feature_memory_neighbors=1,
            source_feature_memory_store_complex_masks=True,
            device="cpu",
        ),
        command_args=[],
    )

    checkpoint = module.torch.load(manifest["training"]["checkpoint_path"], map_location="cpu", weights_only=False)
    assert "metadata" in checkpoint
    persisted_metadata = checkpoint["metadata"]
    assert "source_feature_memory_features" in persisted_metadata
    assert "source_feature_memory_masks" in persisted_metadata
    assert "source_feature_memory_complex_mask_real" in persisted_metadata
    assert "source_feature_memory_complex_mask_imag" in persisted_metadata
    assert persisted_metadata["source_feature_memory_features"].shape[0] > 0


def test_teacher_mask_checkpoint_application_uses_no_teacher_target(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.025 * np.sin(2.0 * np.pi * 920.0 * time))
    source_mix = target + _stereo(0.34 * np.sin(2.0 * np.pi * 190.0 * time))
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)

    trained_manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "trained",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=4,
            batch_size=64,
            source_feature_memory_neighbors=1,
            source_feature_memory_store_complex_masks=True,
            device="cpu",
        ),
        command_args=[],
    )

    applied_manifest = module.apply_teacher_mask_checkpoint_to_source(
        source_mix_path=source_path,
        checkpoint_path=Path(trained_manifest["training"]["checkpoint_path"]),
        output_dir=tmp_path / "applied",
        stem_name="strings_orchestra",
        device="cpu",
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(applied_manifest["stems"]["strings_orchestra"]["path"]))
    assert _sdr(target, output_audio) > 35.0
    assert "teacher_target" not in applied_manifest["inputs"]
    assert applied_manifest["inputs"]["checkpoint"]["path"] == trained_manifest["training"]["checkpoint_path"]
    assert applied_manifest["application"]["uses_reference_during_application"] is False
    assert applied_manifest["contract"]["target_reference_used_at_application"] is False
    assert applied_manifest["contract"]["application_reference_free"] is True
    assert applied_manifest["contract"]["delivery_allowed_without_reference"] is False


def test_teacher_mask_checkpoint_skips_complex_memory_on_non_matching_source(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    train_target = _stereo(0.025 * np.sin(2.0 * np.pi * 920.0 * time))
    train_source = train_target + _stereo(0.34 * np.sin(2.0 * np.pi * 190.0 * time))
    holdout_source = _stereo(
        0.31 * np.sin(2.0 * np.pi * 310.0 * time)
        + 0.18 * np.sin(2.0 * np.pi * 1240.0 * time)
    )
    train_source_path = tmp_path / "train_source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    holdout_source_path = tmp_path / "holdout_source_mix.flac"
    _write_audio(train_source_path, train_source, sample_rate)
    _write_audio(target_path, train_target, sample_rate)
    _write_audio(holdout_source_path, holdout_source, sample_rate)

    trained_manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=train_source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "trained",
        stem_name="vocals",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=4,
            batch_size=64,
            source_feature_memory_neighbors=1,
            source_feature_memory_store_complex_masks=True,
            device="cpu",
        ),
        command_args=[],
    )

    applied_manifest = module.apply_teacher_mask_checkpoint_to_source(
        source_mix_path=holdout_source_path,
        checkpoint_path=Path(trained_manifest["training"]["checkpoint_path"]),
        output_dir=tmp_path / "applied",
        stem_name="vocals",
        device="cpu",
        command_args=[],
    )

    diagnostics = applied_manifest["application"]["diagnostics"]
    assert diagnostics["source_feature_memory_available"] is True
    assert diagnostics["source_feature_memory_used"] is False
    assert diagnostics["source_feature_memory_exact_source_match"] is False
    assert diagnostics["source_feature_memory_skipped_reason"] == "non_matching_source_features"
    assert diagnostics["source_feature_memory_complex_mask_available"] is True
    assert diagnostics["source_feature_memory_complex_mask_used"] is False
    assert diagnostics["source_feature_memory_complex_mask_skipped_reason"] == "non_matching_source_features"
    assert applied_manifest["contract"]["target_reference_used_at_application"] is False
    assert applied_manifest["contract"]["application_reference_free"] is True


def test_teacher_mask_training_can_use_blocked_temporal_features_for_teacher_distillation(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    carrier = np.sin(2.0 * np.pi * 440.0 * time)
    source_mix = _stereo(0.45 * carrier)
    target_envelope = np.zeros_like(time)
    target_envelope[sample_rate // 4 : sample_rate // 2] = 1.0
    target_envelope[3 * sample_rate // 4 :] = 1.0
    teacher_target = _stereo(0.03 * carrier * target_envelope)
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, teacher_target, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=128,
            epochs=120,
            batch_size=64,
            learning_rate=0.01,
            activity_weight=20.0,
            time_feature_count=4,
            seed=789,
            device="cpu",
        ),
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["strings_orchestra"]["path"]))
    active_rms = float(np.sqrt(np.mean(np.square(output_audio[target_envelope > 0.5]), dtype=np.float64)))
    inactive_rms = float(np.sqrt(np.mean(np.square(output_audio[target_envelope < 0.5]), dtype=np.float64)))
    assert active_rms > inactive_rms * 3.0
    assert _sdr(teacher_target, output_audio) > _sdr(teacher_target, source_mix) + 8.0
    assert manifest["contract"]["stores_learned_temporal_position_features"] is True
    assert manifest["contract"]["stores_learned_complex_mask_prior"] is True
    assert manifest["application"]["diagnostics"]["temporal_complex_mask_prior_used"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False


def test_safe_quantile_handles_large_tensors_without_full_quantile_failure() -> None:
    module = _load_module()
    torch = module.torch
    values = torch.linspace(0.0, 1.0, 1_200_000)
    quantile = module.safe_quantile(values, 0.90, max_values=10_000)
    assert 0.88 <= float(quantile) <= 0.92


def test_local_context_features_preserve_frame_count_when_radius_exceeds_input() -> None:
    module = _load_module()
    torch = module.torch
    frame_features = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    context_features = module._local_context_features(frame_features, context_frame_radius=4)
    assert context_features.shape == (2, 27)
    assert torch.equal(context_features[0, :3], frame_features[0])
    assert torch.equal(context_features[-1, -3:], frame_features[-1])


def test_teacher_mask_separator_cli_smoke(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.25 * np.sin(2.0 * np.pi * 880.0 * time))
    interference = _stereo(0.20 * np.sin(2.0 * np.pi * 220.0 * time))
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, target + interference, sample_rate)
    _write_audio(target_path, target, sample_rate)
    output_dir = tmp_path / "out"

    exit_code = module.main(
        [
            "--source-mix",
            str(source_path),
            "--teacher-target",
            str(target_path),
            "--output-dir",
            str(output_dir),
            "--stem",
            "strings_orchestra",
            "--n-fft",
            "256",
            "--hop-length",
            "64",
            "--hidden-size",
            "64",
            "--epochs",
            "12",
            "--batch-size",
            "64",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stems"]["strings_orchestra"]["path"].endswith("01_teacher_mask_strings_orchestra.flac")
    assert manifest["contract"]["quality_gate_required"] is True


def test_teacher_mask_separator_does_not_copy_blocked_candidate_to_listen_dir(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    _write_audio(source_path, _stereo(0.25 * np.sin(2.0 * np.pi * 220.0 * time)), sample_rate)
    _write_audio(target_path, _stereo(np.zeros_like(time)), sample_rate)
    output_dir = tmp_path / "out"
    listen_dir = tmp_path / "listen"

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=output_dir,
        stem_name="strings_orchestra",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=8,
            batch_size=64,
            device="cpu",
        ),
        listen_dir=listen_dir,
        command_args=[],
    )

    assert manifest["presentation"]["status"] == "blocked_from_listen_pack"
    assert manifest["presentation"]["reason"] == "candidate_delivery_blocked"
    assert not (listen_dir / "01_teacher_mask_strings_orchestra.flac").exists()
    assert not (listen_dir / "candidate_map.json").exists()
