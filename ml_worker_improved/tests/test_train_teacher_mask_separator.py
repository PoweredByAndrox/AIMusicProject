import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_select_device_fails_fast_when_cuda_requested_but_unavailable(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        module._select_device("cuda")

    assert module._select_device("cpu").type == "cpu"


def test_select_device_fails_fast_when_mps_requested_but_unavailable(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS was requested"):
        module._select_device("mps")


def test_mps_is_recorded_as_gpu_accelerator() -> None:
    module = _load_module()

    assert module.is_accelerator_device("mps") is True
    assert module.is_accelerator_device(module.torch.device("mps")) is True
    assert module.acceleration_mode("mps", phase="training") == "mps_gpu_training"
    assert module.is_accelerator_device("cpu") is False


def test_cuda_memory_guard_fails_fast_with_progress_when_gpu_memory_is_too_low(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    progress_path = tmp_path / "progress.jsonl"
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", lambda: (512 * 1024 * 1024, 12 * 1024 * 1024 * 1024))

    with pytest.raises(RuntimeError, match="CUDA free memory is below required minimum"):
        module._require_cuda_memory_available(
            module.torch.device("cuda"),
            min_free_memory_mb=4096,
            progress_path=progress_path,
            progress_context={"scope": "memory_guard_test"},
            phase="unit_test",
        )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress_rows[-1]["event"] == "cuda_memory_guard_blocked"
    assert progress_rows[-1]["free_memory_mb"] == 512
    assert progress_rows[-1]["required_free_memory_mb"] == 4096
    assert progress_rows[-1]["scope"] == "memory_guard_test"


def test_cuda_phase_sync_writes_start_and_finish_progress(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    progress_path = tmp_path / "progress.jsonl"
    calls: list[str] = []
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device=None: calls.append(str(device)))

    module._synchronize_cuda_phase(
        progress_path,
        {"scope": "sync_test"},
        event_prefix="unit_phase",
        device=module.torch.device("cuda"),
        frame_count=123,
    )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert calls == ["cuda"]
    assert [row["event"] for row in progress_rows] == [
        "unit_phase_cuda_sync_start",
        "unit_phase_cuda_sync_finish",
    ]
    assert progress_rows[-1]["frame_count"] == 123
    assert progress_rows[-1]["scope"] == "sync_test"


def test_magnitude_network_can_emit_masks_above_one_when_ceiling_allows() -> None:
    module = _load_module()
    model = module.TeacherMaskNetwork(
        input_features=3,
        frequency_bins=2,
        hidden_size=4,
        mask_output_mode=module.MASK_OUTPUT_MODE_MAGNITUDE,
        mask_ceiling=1.7,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.layers[-1].bias.data.fill_(4.0)

    prediction = model(module.torch.zeros((1, 3), dtype=module.torch.float32))

    assert float(prediction.max()) > 1.0
    assert float(prediction.max()) <= 1.70001


def test_source_feature_memory_finds_exact_subsequence_match() -> None:
    module = _load_module()
    query = module.torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
        ],
        dtype=module.torch.float32,
    )
    memory = module.torch.tensor(
        [
            [9.0, 9.0, 9.0],
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [8.0, 8.0, 8.0],
        ],
        dtype=module.torch.float32,
    )

    assert module._features_are_exact_memory_match(query, memory) is False
    assert module._find_exact_memory_subsequence(query, memory) == 1


def test_magnitude_residual_mask_mode_is_identity_centered_and_bounded() -> None:
    module = _load_module()
    model = module.TeacherMaskNetwork(
        input_features=3,
        frequency_bins=2,
        hidden_size=4,
        mask_output_mode=module.MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL,
        mask_ceiling=1.2,
    )
    for parameter in model.parameters():
        parameter.data.zero_()

    prediction = model(module.torch.zeros((1, 3), dtype=module.torch.float32))
    assert module.validate_mask_output_mode("magnitude_residual") == module.MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL
    assert module.torch.allclose(prediction, module.torch.ones((1, 2)))

    model.layers[-1].bias.data.fill_(10.0)
    high_prediction = model(module.torch.zeros((1, 3), dtype=module.torch.float32))
    assert float(high_prediction.max()) <= 1.20001

    model.layers[-1].bias.data.fill_(-10.0)
    low_prediction = model(module.torch.zeros((1, 3), dtype=module.torch.float32))
    assert float(low_prediction.min()) >= 0.79999


def test_magnitude_residual_mask_mode_requires_ceiling_above_identity() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires mask_ceiling > 1.0"):
        module.TeacherMaskNetwork(
            input_features=3,
            frequency_bins=2,
            hidden_size=4,
            mask_output_mode=module.MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL,
            mask_ceiling=1.0,
        )


def test_config_payload_records_source_preservation_weight() -> None:
    module = _load_module()

    payload = module._config_payload(
        module.TeacherMaskTrainConfig(
            source_preservation_weight=2.5,
            device="cpu",
        )
    )

    assert payload["source_preservation_weight"] == 2.5


def test_source_preservation_loss_only_uses_declared_identity_scope() -> None:
    module = _load_module()
    delta = module.torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    weights = module.torch.ones_like(delta)
    identity_scope = module.torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    scoped_loss = module._scoped_source_preservation_loss(delta, weights, identity_scope)
    empty_loss = module._scoped_source_preservation_loss(delta, weights, module.torch.zeros_like(delta))

    assert float(scoped_loss) == 1.0
    assert float(empty_loss) == 0.0


def test_source_feature_memory_projection_accepts_half_feature_storage() -> None:
    module = _load_module()
    model_features = module.torch.ones((3, 4), dtype=module.torch.float16)
    projection_matrix = module.torch.full((4, 2), 0.5, dtype=module.torch.float32)

    projected = module._project_source_feature_memory_features(model_features, projection_matrix)

    assert projected.dtype == module.torch.float32
    assert projected.shape == (3, 2)
    assert module.torch.allclose(projected, module.torch.full((3, 2), 2.0))


def test_safe_quantile_large_cuda_tensor_uses_in_bounds_sampling() -> None:
    module = _load_module()
    if not module.torch.cuda.is_available():
        pytest.skip("CUDA quantile regression requires GPU Docker runtime")
    length = 19_823_500
    sample_count = 1_000_000

    indices = module._evenly_spaced_long_indices(
        length=length,
        count=sample_count,
        device=module.torch.device("cuda"),
    )

    assert int(indices.min().detach().cpu()) == 0
    assert int(indices.max().detach().cpu()) == length - 1
    assert not bool(((indices < 0) | (indices >= length)).any().detach().cpu())
    values = module.torch.zeros(length, dtype=module.torch.float32, device="cuda")
    quantile = module.safe_quantile(values, 0.90, max_values=sample_count)
    module.torch.cuda.synchronize()
    assert float(quantile.detach().cpu()) == 0.0


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
    assert manifest["application"]["diagnostics"]["device"] == "cpu"
    assert manifest["application"]["diagnostics"]["device_used"] == 0.0

    progress_path = tmp_path / "out" / "teacher_mask_separator_progress.jsonl"
    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress_rows[0]["event"] == "start"
    assert progress_rows[0]["expected_acceleration_mode"] == "cpu_test_training"
    events = [row["event"] for row in progress_rows]
    assert "train_loop_start" in events
    assert "train_epoch_start" in events
    assert "train_permutation_start" in events
    assert "train_permutation_finish" in events
    assert "train_batch_index_select_start" in events
    assert "train_batch_index_select_finish" in events
    assert "train_batch_start" in events
    assert "train_batch_feature_gather_start" in events
    assert "train_batch_feature_gather_finish" in events
    assert "train_batch_forward_start" in events
    assert "train_batch_forward_finish" in events
    assert "train_batch_loss_start" in events
    assert "train_batch_loss_finish" in events
    assert "train_batch_backward_start" in events
    assert "train_batch_backward_finish" in events
    assert "train_batch_optimizer_step_start" in events
    assert "train_batch_optimizer_step_finish" in events
    train_epoch_rows = [row for row in progress_rows if row["event"] == "train_epoch"]
    assert [row["epoch"] for row in train_epoch_rows] == list(range(1, 81))
    assert progress_rows[-1]["event"] == "finish"


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


def test_teacher_mask_training_resamples_teacher_target_to_source_rate(tmp_path: Path) -> None:
    module = _load_module()
    source_sample_rate = 8_000
    target_sample_rate = 12_000
    source_time = np.arange(source_sample_rate, dtype=np.float64) / float(source_sample_rate)
    target_time = np.arange(target_sample_rate, dtype=np.float64) / float(target_sample_rate)
    source_target = _stereo(0.22 * np.sin(2.0 * np.pi * 440.0 * source_time))
    source_mix = source_target + _stereo(0.18 * np.sin(2.0 * np.pi * 130.0 * source_time))
    teacher_target = _stereo(0.22 * np.sin(2.0 * np.pi * 440.0 * target_time))
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target_12k.flac"
    _write_audio(source_path, source_mix, source_sample_rate)
    _write_audio(target_path, teacher_target, target_sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "out",
        stem_name="vocal",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=512,
            hop_length=128,
            hidden_size=96,
            epochs=35,
            batch_size=64,
            learning_rate=0.01,
            seed=321,
            device="cpu",
        ),
        command_args=[],
    )

    alignment = manifest["training"]["sample_rate_alignment"]
    assert alignment["source_sample_rate"] == source_sample_rate
    assert alignment["teacher_target_original_sample_rate"] == target_sample_rate
    assert alignment["teacher_target_sample_rate_after_alignment"] == source_sample_rate
    assert alignment["teacher_target_resampled"] is True
    assert Path(manifest["stems"]["vocal"]["path"]).exists()


def test_teacher_mask_dataset_training_generalizes_to_held_out_synthetic_pair(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for interference_hz in (120.0, 180.0):
        target = _stereo(0.26 * np.sin(2.0 * np.pi * 720.0 * time))
        source = target + _stereo(0.34 * np.sin(2.0 * np.pi * interference_hz * time))
        train_pairs.append((source, target))
    holdout_target = _stereo(0.26 * np.sin(2.0 * np.pi * 720.0 * time))
    holdout_source = holdout_target + _stereo(0.34 * np.sin(2.0 * np.pi * 150.0 * time))

    model, metadata, diagnostics = module.train_teacher_mask_dataset(
        train_pairs,
        config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=35,
            batch_size=64,
            learning_rate=0.01,
            activity_weight=10.0,
            seed=314,
            device="cpu",
        ),
        max_frames_per_pair=64,
    )
    output_audio, apply_diagnostics = module.apply_teacher_mask(
        holdout_source,
        model=model,
        metadata=metadata,
        config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=35,
            batch_size=64,
            learning_rate=0.01,
            activity_weight=10.0,
            seed=314,
            device="cpu",
        ),
    )

    assert _sdr(holdout_target, output_audio) > _sdr(holdout_target, holdout_source) + 6.0
    assert diagnostics["training_example_count"] == 2.0
    assert diagnostics["unique_training_pair_count"] == 2.0
    assert diagnostics["frame_count"] == 128.0
    assert diagnostics["device_used"] == 0.0
    assert metadata["input_features"] > 0
    assert apply_diagnostics["device"] == "cpu"


def test_teacher_mask_dataset_progress_logs_every_epoch_for_observability(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 4_000
    time = np.arange(sample_rate // 4, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.18 * np.sin(2.0 * np.pi * 520.0 * time))
    source = target + _stereo(0.11 * np.sin(2.0 * np.pi * 130.0 * time))
    progress_path = tmp_path / "progress.jsonl"

    module.train_teacher_mask_dataset(
        [(source, target)],
        config=module.TeacherMaskTrainConfig(
            n_fft=128,
            hop_length=64,
            hidden_size=24,
            epochs=24,
            batch_size=16,
            learning_rate=0.01,
            seed=917,
            device="cpu",
        ),
        progress_path=progress_path,
        progress_context={"scope": "observability_test"},
        max_frames_per_pair=16,
    )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_epoch_rows = [row for row in progress_rows if row["event"] == "dataset_train_epoch"]
    assert [row["epoch"] for row in train_epoch_rows] == list(range(1, 25))
    assert all(row["scope"] == "observability_test" for row in train_epoch_rows)


def test_teacher_mask_dataset_progress_logs_batch_heartbeat_inside_epoch(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 4_000
    time = np.arange(sample_rate // 3, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.16 * np.sin(2.0 * np.pi * 480.0 * time))
    source = target + _stereo(0.12 * np.sin(2.0 * np.pi * 150.0 * time))
    progress_path = tmp_path / "progress.jsonl"

    module.train_teacher_mask_dataset(
        [(source, target)],
        config=module.TeacherMaskTrainConfig(
            n_fft=128,
            hop_length=32,
            hidden_size=16,
            epochs=2,
            batch_size=5,
            learning_rate=0.01,
            seed=918,
            device="cpu",
        ),
        progress_path=progress_path,
        progress_context={"scope": "batch_observability_test"},
        max_frames_per_pair=24,
    )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [row["event"] for row in progress_rows]
    assert "dataset_feature_stack_start" in events
    assert "dataset_feature_stack_finish" in events
    assert "dataset_feature_chunks_released" in events
    assert "dataset_model_init_start" in events
    assert "dataset_network_construct_start" in events
    assert "dataset_network_construct_finish" in events
    assert "dataset_model_to_device_start" in events
    assert "dataset_model_to_device_finish" in events
    assert "dataset_optimizer_init_start" in events
    assert "dataset_optimizer_init_finish" in events
    assert "dataset_model_init_finish" in events
    assert "dataset_training_tensor_stack_start" in events
    assert "dataset_training_tensor_stack_finish" in events
    assert "dataset_training_chunks_released" in events
    assert "dataset_training_indices_start" in events
    assert "dataset_training_indices_finish" in events
    assert "dataset_train_loop_start" in events
    assert "dataset_train_epoch_start" in events
    assert "dataset_train_permutation_start" in events
    assert "dataset_train_permutation_finish" in events
    assert "dataset_feature_stats_start" in events
    assert "dataset_feature_stats_finish" in events
    assert "dataset_feature_normalize_start" in events
    assert "dataset_feature_normalize_finish" in events
    assert "dataset_train_batch_index_select_start" in events
    assert "dataset_train_batch_index_select_finish" in events
    assert "dataset_train_batch_start" in events
    assert "dataset_train_batch_feature_gather_start" in events
    assert "dataset_train_batch_feature_gather_finish" in events
    assert "dataset_train_batch_forward_start" in events
    assert "dataset_train_batch_forward_finish" in events
    assert "dataset_train_batch_loss_start" in events
    assert "dataset_train_batch_loss_finish" in events
    assert "dataset_train_batch_backward_start" in events
    assert "dataset_train_batch_backward_finish" in events
    assert "dataset_train_batch_optimizer_step_start" in events
    assert "dataset_train_batch_optimizer_step_finish" in events
    permutation_rows = [row for row in progress_rows if row["event"] == "dataset_train_permutation_finish"]
    assert {row["epoch"] for row in permutation_rows} == {1, 2}
    assert all(row["training_device"] == "cpu" for row in permutation_rows)
    batch_start_rows = [row for row in progress_rows if row["event"] == "dataset_train_batch_start"]
    assert batch_start_rows[0]["epoch"] == 1
    assert batch_start_rows[0]["batch_index"] == 1
    select_start_rows = [
        row for row in progress_rows if row["event"] == "dataset_train_batch_index_select_start"
    ]
    assert select_start_rows[0]["epoch"] == 1
    assert select_start_rows[0]["batch_index"] == 1
    heartbeat_rows = [row for row in progress_rows if row["event"] == "dataset_train_batch_heartbeat"]
    assert heartbeat_rows
    assert {row["epoch"] for row in heartbeat_rows} == {1, 2}
    assert all(row["batch_count"] >= row["batch_index"] >= 1 for row in heartbeat_rows)
    last_batch_rows = [row for row in heartbeat_rows if row["batch_index"] == row["batch_count"]]
    assert {row["epoch"] for row in last_batch_rows} == {1, 2}
    assert all(row["scope"] == "batch_observability_test" for row in heartbeat_rows)
    feature_stack_finish = next(row for row in progress_rows if row["event"] == "dataset_feature_stack_finish")
    feature_normalize_finish = next(row for row in progress_rows if row["event"] == "dataset_feature_normalize_finish")
    assert feature_stack_finish["normalization_strategy"] == "chunked_cpu_low_peak"
    assert feature_normalize_finish["normalization_strategy"] == "chunked_cpu_low_peak"


def test_complex_temporal_prior_metadata_is_finite_for_near_floor_source(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 4_000
    time = np.arange(sample_rate // 4, dtype=np.float64) / float(sample_rate)
    source = _stereo(1.0e-7 * np.sin(2.0 * np.pi * 520.0 * time))
    target = _stereo(0.20 * np.sin(2.0 * np.pi * 520.0 * time))

    _model, metadata, _diagnostics = module.train_teacher_mask(
        source,
        target,
        config=module.TeacherMaskTrainConfig(
            n_fft=128,
            hop_length=32,
            hidden_size=16,
            epochs=1,
            batch_size=16,
            learning_rate=0.01,
            mask_output_mode=module.MASK_OUTPUT_MODE_COMPLEX,
            complex_mask_ceiling=2.0,
            time_feature_count=1,
            seed=920,
            device="cpu",
        ),
        progress_path=tmp_path / "progress.jsonl",
        progress_context={"scope": "complex_prior_finite_test"},
    )

    real = np.asarray(metadata["temporal_complex_mask_prior_real"], dtype=np.float32)
    imag = np.asarray(metadata["temporal_complex_mask_prior_imag"], dtype=np.float32)

    assert np.isfinite(real).all()
    assert np.isfinite(imag).all()
    assert float(np.max(np.abs(real))) <= 2.0001
    assert float(np.max(np.abs(imag))) <= 2.0001


def test_teacher_mask_dataset_cuda_keeps_precomputed_tensors_cpu_resident(tmp_path: Path) -> None:
    module = _load_module()
    if not module.torch.cuda.is_available():
        pytest.skip("CUDA dataset residency test requires GPU Docker runtime")
    sample_rate = 4_000
    time = np.arange(sample_rate // 4, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.14 * np.sin(2.0 * np.pi * 420.0 * time))
    source = target + _stereo(0.08 * np.sin(2.0 * np.pi * 120.0 * time))
    progress_path = tmp_path / "progress.jsonl"

    module.train_teacher_mask_dataset(
        [(source, target)],
        config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=8,
            epochs=1,
            batch_size=8,
            learning_rate=0.01,
            seed=919,
            device="cuda",
        ),
        progress_path=progress_path,
        progress_context={"scope": "cuda_residency_test"},
        max_frames_per_pair=16,
    )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [row["event"] for row in progress_rows]
    stack_finish = next(row for row in progress_rows if row["event"] == "dataset_training_tensor_stack_finish")
    transfer_finish = next(row for row in progress_rows if row["event"] == "dataset_train_batch_to_device_finish")

    assert stack_finish["training_tensor_storage_device"] == "cpu"
    assert stack_finish["training_tensor_storage_dtype"] == "torch.float16"
    assert stack_finish["complex_training_tensor_stored"] is False
    assert "dataset_train_batch_to_device_start" in events
    assert "dataset_train_batch_to_device_finish" in events
    assert transfer_finish["training_tensor_storage_device"] == "cpu"
    assert transfer_finish["training_device"] == "cuda"


def test_teacher_mask_dataset_complex_output_mode_applies_complex_mask(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    train_target = _stereo(0.16 * np.sin(2.0 * np.pi * 720.0 * time + 0.45))
    train_source = train_target + _stereo(0.20 * np.sin(2.0 * np.pi * 180.0 * time))
    holdout_target = _stereo(0.16 * np.sin(2.0 * np.pi * 720.0 * time + 0.45))
    holdout_source = holdout_target + _stereo(0.18 * np.sin(2.0 * np.pi * 210.0 * time))
    config = module.TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=64,
        epochs=18,
        batch_size=64,
        learning_rate=0.01,
        mask_output_mode="complex",
        complex_mask_ceiling=2.5,
        seed=918,
        device="cpu",
    )

    model, metadata, diagnostics = module.train_teacher_mask_dataset(
        [(train_source, train_target)],
        config=config,
        max_frames_per_pair=96,
    )
    output_audio, apply_diagnostics = module.apply_teacher_mask(
        holdout_source,
        model=model,
        metadata=metadata,
        config=config,
    )

    assert output_audio.shape == holdout_source.shape
    assert np.isfinite(output_audio).all()
    assert float(np.max(np.abs(output_audio))) > 1.0e-6
    assert metadata["mask_output_mode"] == "complex"
    assert diagnostics["mask_output_mode"] == "complex"
    assert apply_diagnostics["mask_output_mode"] == "complex"
    assert apply_diagnostics["complex_mask_used"] is True
    assert apply_diagnostics["complex_mask_ceiling"] == 2.5


def test_teacher_mask_dataset_complex_output_mode_uses_source_feature_memory(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    train_target = _stereo(0.11 * np.sin(2.0 * np.pi * 620.0 * time + 0.2))
    train_source = train_target + _stereo(0.14 * np.sin(2.0 * np.pi * 170.0 * time))
    holdout_target = _stereo(0.11 * np.sin(2.0 * np.pi * 620.0 * time + 0.2))
    holdout_source = holdout_target + _stereo(0.12 * np.sin(2.0 * np.pi * 190.0 * time))
    config = module.TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=48,
        epochs=8,
        batch_size=64,
        mask_output_mode="complex",
        complex_mask_ceiling=2.0,
        source_feature_memory_neighbors=1,
        source_feature_memory_chunk_size=32,
        source_feature_memory_projection_dim=8,
        source_feature_memory_max_frames=48,
        seed=2424,
        device="cpu",
    )

    model, metadata, diagnostics = module.train_teacher_mask_dataset(
        [(train_source, train_target)],
        config=config,
        max_frames_per_pair=96,
    )
    output_audio, apply_diagnostics = module.apply_teacher_mask(
        holdout_source,
        model=model,
        metadata=metadata,
        config=config,
    )

    assert output_audio.shape == holdout_source.shape
    assert metadata["mask_output_mode"] == "complex"
    assert "source_feature_memory_complex_frame_mask_real" in metadata
    assert "source_feature_memory_complex_frame_mask_imag" in metadata
    assert diagnostics["source_feature_memory_frame_count"] == 48.0
    assert apply_diagnostics["source_feature_memory_used"] is True
    assert apply_diagnostics["source_feature_memory_complex_frame_mask_available"] is True
    assert apply_diagnostics["source_feature_memory_complex_frame_mask_used"] is True
    assert apply_diagnostics["complex_mask_used"] is False
    assert _sdr(holdout_target, output_audio) > _sdr(holdout_target, holdout_source) + 3.0


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


def test_teacher_mask_source_feature_memory_generalizes_to_non_matching_source(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    train_target = _stereo(0.08 * np.sin(2.0 * np.pi * 940.0 * time))
    train_source = train_target + _stereo(0.32 * np.sin(2.0 * np.pi * 160.0 * time))
    holdout_target = _stereo(0.08 * np.sin(2.0 * np.pi * 940.0 * time))
    holdout_source = holdout_target + _stereo(0.30 * np.sin(2.0 * np.pi * 210.0 * time))
    train_source_path = tmp_path / "train_source.flac"
    train_target_path = tmp_path / "train_target.flac"
    holdout_source_path = tmp_path / "holdout_source.flac"
    _write_audio(train_source_path, train_source, sample_rate)
    _write_audio(train_target_path, train_target, sample_rate)
    _write_audio(holdout_source_path, holdout_source, sample_rate)

    trained_manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=train_source_path,
        teacher_target_path=train_target_path,
        output_dir=tmp_path / "trained",
        stem_name="vocal",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=8,
            batch_size=64,
            source_feature_memory_neighbors=1,
            source_feature_memory_chunk_size=64,
            device="cpu",
        ),
        command_args=[],
    )

    applied_manifest = module.apply_teacher_mask_checkpoint_to_source(
        source_mix_path=holdout_source_path,
        checkpoint_path=Path(trained_manifest["training"]["checkpoint_path"]),
        output_dir=tmp_path / "applied",
        stem_name="vocal",
        device="cpu",
        command_args=[],
    )

    output_audio, _ = module.read_audio(Path(applied_manifest["stems"]["vocal"]["path"]))
    assert _sdr(holdout_target, output_audio) > _sdr(holdout_target, holdout_source) + 6.0
    assert applied_manifest["application"]["diagnostics"]["source_feature_memory_used"] is True
    assert applied_manifest["application"]["diagnostics"]["source_feature_memory_exact_source_match"] is False
    assert applied_manifest["application"]["diagnostics"]["source_feature_memory_neighbor_count"] == 1


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


def test_teacher_mask_checkpoint_application_blocks_excessive_source_memory_search(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 2, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.025 * np.sin(2.0 * np.pi * 920.0 * time))
    source_mix = target + _stereo(0.34 * np.sin(2.0 * np.pi * 190.0 * time))
    holdout_mix = _stereo(
        0.31 * np.sin(2.0 * np.pi * 310.0 * time)
        + 0.18 * np.sin(2.0 * np.pi * 1240.0 * time)
    )
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    holdout_path = tmp_path / "holdout_mix.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)
    _write_audio(holdout_path, holdout_mix, sample_rate)

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
            source_feature_memory_chunk_size=64,
            device="cpu",
        ),
        command_args=[],
    )
    progress_path = tmp_path / "applied" / "checkpoint_application_progress.jsonl"

    with pytest.raises(RuntimeError, match="source_feature_memory_cost_guard"):
        module.apply_teacher_mask_checkpoint_to_source(
            source_mix_path=holdout_path,
            checkpoint_path=Path(trained_manifest["training"]["checkpoint_path"]),
            output_dir=tmp_path / "applied",
            stem_name="strings_orchestra",
            device="cpu",
            command_args=[],
            progress_path=progress_path,
            max_source_feature_memory_distance_ops=1,
        )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [row["event"] for row in progress_rows]
    assert "checkpoint_application_start" in events
    assert "apply_feature_extraction_finish" in events
    assert "source_feature_memory_search_start" in events
    assert "source_feature_memory_search_blocked" in events
    assert not (tmp_path / "applied" / "01_teacher_mask_checkpoint_strings_orchestra.flac").exists()


def test_teacher_mask_source_feature_memory_projection_and_frame_cap_are_persisted(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    target = _stereo(0.03 * np.sin(2.0 * np.pi * 910.0 * time))
    source_mix = target + _stereo(0.32 * np.sin(2.0 * np.pi * 170.0 * time))
    holdout_mix = target + _stereo(0.28 * np.sin(2.0 * np.pi * 230.0 * time))
    source_path = tmp_path / "source_mix.flac"
    target_path = tmp_path / "teacher_target.flac"
    holdout_path = tmp_path / "holdout_mix.flac"
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(target_path, target, sample_rate)
    _write_audio(holdout_path, holdout_mix, sample_rate)

    manifest = module.train_and_apply_teacher_mask_separator(
        source_mix_path=source_path,
        teacher_target_path=target_path,
        output_dir=tmp_path / "trained",
        stem_name="vocal",
        train_config=module.TeacherMaskTrainConfig(
            n_fft=256,
            hop_length=64,
            hidden_size=64,
            epochs=4,
            batch_size=64,
            source_feature_memory_neighbors=2,
            source_feature_memory_chunk_size=32,
            source_feature_memory_projection_dim=8,
            source_feature_memory_max_frames=5,
            device="cpu",
        ),
        command_args=[],
    )
    checkpoint = module.torch.load(manifest["training"]["checkpoint_path"], map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    assert checkpoint["config"]["source_feature_memory_projection_dim"] == 8
    assert checkpoint["config"]["source_feature_memory_max_frames"] == 5
    assert metadata["source_feature_memory_features"].shape == (5, 8)
    assert metadata["source_feature_memory_masks"].shape[0] == 5
    assert metadata["source_feature_memory_projection_matrix"].shape[1] == 8
    assert metadata["source_feature_memory_selected_indices"].shape[0] == 5

    applied_manifest = module.apply_teacher_mask_checkpoint_to_source(
        source_mix_path=holdout_path,
        checkpoint_path=Path(manifest["training"]["checkpoint_path"]),
        output_dir=tmp_path / "applied",
        stem_name="vocal",
        device="cpu",
        command_args=[],
    )
    diagnostics = applied_manifest["application"]["diagnostics"]
    assert diagnostics["source_feature_memory_used"] is True
    assert diagnostics["source_feature_memory_frame_count"] == 5
    assert diagnostics["source_feature_memory_projection_dim"] == 8
    assert diagnostics["source_feature_memory_max_frames"] == 5


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
    assert diagnostics["source_feature_memory_used"] is True
    assert diagnostics["source_feature_memory_exact_source_match"] is False
    assert diagnostics["source_feature_memory_skipped_reason"] is None
    assert diagnostics["source_feature_memory_neighbor_count"] == 1
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


def test_conditioned_features_are_permutation_stable() -> None:
    module = _load_module()
    source = module.torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    proxy_one = module.torch.tensor([[0.5, 0.25], [0.75, 0.5]])
    proxy_two = module.torch.tensor([[0.25, 0.5], [0.5, 0.75]])

    forward = module._conditioned_local_context_features(source, [proxy_one, proxy_two], 0)
    reversed_order = module._conditioned_local_context_features(source, [proxy_two, proxy_one], 0)

    assert forward.shape == (2, 6)
    assert module.torch.allclose(forward, reversed_order)


def test_conditioned_checkpoint_requires_exact_proxy_count() -> None:
    module = _load_module()
    sample_rate = 8_000
    time = np.arange(sample_rate // 4, dtype=np.float64) / sample_rate
    source = _stereo(0.2 * np.sin(2.0 * np.pi * 220.0 * time))
    proxy_one = _stereo(0.05 * np.sin(2.0 * np.pi * 660.0 * time))
    proxy_two = _stereo(0.04 * np.sin(2.0 * np.pi * 700.0 * time))
    config = module.TeacherMaskTrainConfig(
        n_fft=256,
        hop_length=64,
        hidden_size=16,
        mask_output_mode=module.MASK_OUTPUT_MODE_MAGNITUDE_RESIDUAL,
        mask_ceiling=1.1,
        device="cpu",
    )
    frequency_bins = config.n_fft // 2 + 1
    input_features = frequency_bins * 3
    model = module.TeacherMaskNetwork(
        input_features=input_features,
        frequency_bins=frequency_bins,
        hidden_size=config.hidden_size,
        mask_output_mode=config.mask_output_mode,
        mask_ceiling=config.mask_ceiling,
    )
    metadata = {
        "feature_mean": np.zeros((1, input_features), dtype=np.float32),
        "feature_std": np.ones((1, input_features), dtype=np.float32),
        "conditioning_audio_count": 2,
        "conditioning_feature_schema": module.CONDITIONING_FEATURE_SCHEMA,
    }

    with pytest.raises(ValueError, match="conditioning audio count mismatch"):
        module.apply_teacher_mask(source, model=model, metadata=metadata, config=config, conditioning_audio=[proxy_one])

    _, diagnostics = module.apply_teacher_mask(
        source,
        model=model,
        metadata=metadata,
        config=config,
        conditioning_audio=[proxy_one, proxy_two],
    )
    assert diagnostics["conditioning_audio_count"] == 2
    assert diagnostics["conditioning_feature_schema"] == module.CONDITIONING_FEATURE_SCHEMA


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
