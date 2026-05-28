from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_teacher_profile_bank.py"


def _load_module():
    if str(TOOLS_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOLS_ROOT))
    spec = importlib.util.spec_from_file_location("build_teacher_profile_bank", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _stereo_tone(sample_rate: int, amplitude: float, frequency_hz: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return np.stack([tone, tone], axis=1).astype(np.float32)


def _write_example(root: Path, stem: str, frequency_hz: float) -> tuple[Path, Path, Path]:
    module = _load_module()
    sample_rate = 16000
    reference = _stereo_tone(sample_rate, amplitude=0.42, frequency_hz=frequency_hz)
    candidate = _stereo_tone(sample_rate, amplitude=0.20, frequency_hz=frequency_hz)
    teacher = _stereo_tone(sample_rate, amplitude=0.36, frequency_hz=frequency_hz)

    ground_truth_manifest = root / "gt" / "ground_truth_manifest.json"
    candidate_manifest = root / "candidate" / "stem_manifest.json"
    teacher_manifest = root / "teacher" / "stem_manifest.json"
    _write_audio(root / "gt" / "source_mix.flac", reference, sample_rate)
    _write_audio(root / "gt" / "stems" / f"{stem}.flac", reference, sample_rate)
    _write_audio(root / "candidate" / f"{stem}.flac", candidate, sample_rate)
    _write_audio(root / "teacher" / f"{stem}.flac", teacher, sample_rate)

    module.write_json(
        ground_truth_manifest,
        {
            "ground_truth_contract": {"verified": True},
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac", "derived": False}},
        },
    )
    module.write_json(
        candidate_manifest,
        {"contract": {"benchmark_compatible": True}, "stems": {stem: {"path": f"{stem}.flac"}}},
    )
    module.write_json(
        teacher_manifest,
        {"contract": {"benchmark_compatible": True}, "stems": {stem: {"path": f"{stem}.flac"}}},
    )
    return ground_truth_manifest, candidate_manifest, teacher_manifest


def test_profile_bank_marks_supervised_source_and_holdout_required(tmp_path: Path) -> None:
    module = _load_module()
    gt_manifest, candidate_manifest, teacher_manifest = _write_example(tmp_path / "drums", "drums", 120.0)

    profile_bank = module.build_teacher_profile_bank(
        training_examples=[
            module.TrainingExample(
                stem_name="drums",
                ground_truth_manifest=gt_manifest,
                base_candidate_manifest=candidate_manifest,
                teacher_manifest=teacher_manifest,
            )
        ],
        output_path=tmp_path / "teacher_profile_bank.json",
        config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=9.0),
    )

    assert profile_bank["contract"]["profile_source_reference_assisted"] is True
    assert profile_bank["contract"]["application_reference_free"] is True
    assert profile_bank["contract"]["target_reference_used_at_application"] is False
    assert profile_bank["contract"]["requires_holdout_validation"] is True
    assert profile_bank["contract"]["delivery_allowed_without_reference"] is False
    assert profile_bank["contract"]["per_track_hardcode_allowed"] is False
    assert profile_bank["contract"]["stores_timestamp_rules"] is False
    assert profile_bank["stems"]["drums"]["training_example_count"] == 1
    assert (tmp_path / "teacher_profile_bank.json").exists()


def test_profile_bank_application_uses_no_reference_and_still_requires_gate(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    gt_manifest, candidate_manifest, teacher_manifest = _write_example(tmp_path / "vocals", "vocals", 440.0)
    profile_bank_path = tmp_path / "teacher_profile_bank.json"
    module.build_teacher_profile_bank(
        training_examples=[
            module.TrainingExample(
                stem_name="vocals",
                ground_truth_manifest=gt_manifest,
                base_candidate_manifest=candidate_manifest,
                teacher_manifest=teacher_manifest,
            )
        ],
        output_path=profile_bank_path,
        config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=9.0),
    )

    manifest = module.apply_teacher_profile_bank(
        profile_bank_path=profile_bank_path,
        candidate_manifest=candidate_manifest,
        output_dir=tmp_path / "applied",
        stem_name="vocals",
        config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=9.0),
    )

    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["profile_source_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["quality_gate_required"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["profile_application"]["uses_reference_during_application"] is False
    assert "ground_truth_manifest" not in manifest["inputs"]
    assert (tmp_path / "applied" / "vocals.flac").exists()
    assert (tmp_path / "applied" / "stem_manifest.json").exists()


def test_profile_bank_rejects_unknown_stem_at_application(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    gt_manifest, candidate_manifest, teacher_manifest = _write_example(tmp_path / "drums", "drums", 120.0)
    profile_bank_path = tmp_path / "teacher_profile_bank.json"
    module.build_teacher_profile_bank(
        training_examples=[
            module.TrainingExample(
                stem_name="drums",
                ground_truth_manifest=gt_manifest,
                base_candidate_manifest=candidate_manifest,
                teacher_manifest=teacher_manifest,
            )
        ],
        output_path=profile_bank_path,
        config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=9.0),
    )

    try:
        module.apply_teacher_profile_bank(
            profile_bank_path=profile_bank_path,
            candidate_manifest=candidate_manifest,
            output_dir=tmp_path / "applied",
            stem_name="vocals",
            config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=9.0),
        )
    except ValueError as exc:
        assert "profile bank has no profile for stem" in str(exc)
    else:
        raise AssertionError("unknown stem must fail before writing a fake candidate")


def test_teacher_profile_selector_rejects_calibrated_score_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    adaptive_manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    module.write_json(
        adaptive_manifest_path,
        {
            "selected_profile": "cal_hybrid",
            "selected_score": -1.5,
            "candidates": [
                {"profile_name": "orig_hybrid", "score": -1.0},
                {"profile_name": "cal_hybrid", "score": -1.5},
                {"profile_name": "orig_scnet", "score": -7.0},
                {"profile_name": "cal_scnet", "score": -6.5},
            ],
        },
    )

    selector_manifest = module.build_teacher_profile_selector_manifest(
        adaptive_manifest_path=adaptive_manifest_path,
        output_manifest_path=tmp_path / "teacher_profile_selector_manifest.json",
        min_score_delta=0.0,
    )

    assert selector_manifest["status"] == "blocked_selected_calibrated_regression"
    assert selector_manifest["contract"]["target_reference_used_at_application"] is False
    assert selector_manifest["contract"]["application_reference_free"] is True
    assert selector_manifest["contract"]["delivery_allowed"] is False
    assert selector_manifest["contract"]["does_not_modify_audio"] is True
    assert selector_manifest["kept_calibrated_profiles"] == ["cal_scnet"]
    assert selector_manifest["rejected_calibrated_profiles"] == ["cal_hybrid"]
    hybrid = next(
        pair
        for pair in selector_manifest["pair_decisions"]
        if pair["calibrated_profile"] == "cal_hybrid"
    )
    assert hybrid["status"] == "reject_calibrated_regression"
    assert hybrid["score_delta"] == -0.5
    assert (tmp_path / "teacher_profile_selector_manifest.json").exists()


def test_teacher_profile_selector_requires_candidate_scores(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    adaptive_manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    module.write_json(
        adaptive_manifest_path,
        {
            "selected_profile": "cal_bad",
            "candidates": [
                {"profile_name": "orig_bad", "score": 0.1},
                {"profile_name": "cal_bad"},
            ],
        },
    )

    try:
        module.build_teacher_profile_selector_manifest(
            adaptive_manifest_path=adaptive_manifest_path,
            output_manifest_path=tmp_path / "teacher_profile_selector_manifest.json",
            min_score_delta=0.0,
        )
    except ValueError as exc:
        assert "candidate entry is missing score" in str(exc)
    else:
        raise AssertionError("selector must fail when adaptive audit has no candidate score")


def test_select_from_audit_cli_does_not_require_audio_profile_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    adaptive_manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    output_manifest_path = tmp_path / "teacher_profile_selector_manifest.json"
    module.write_json(
        adaptive_manifest_path,
        {
            "selected_profile": "cal_a",
            "candidates": [
                {"profile_name": "orig_a", "score": 1.0},
                {"profile_name": "cal_a", "score": 1.1},
            ],
        },
    )

    exit_code = module.main(
        [
            "select-from-audit",
            "--adaptive-manifest",
            str(adaptive_manifest_path),
            "--output-manifest",
            str(output_manifest_path),
        ]
    )

    assert exit_code == 0
    selector_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    assert selector_manifest["status"] == "has_non_regressed_calibrated_candidates"


def test_teacher_profile_selector_blocks_selected_calibrated_without_original_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    adaptive_manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    module.write_json(
        adaptive_manifest_path,
        {
            "selected_profile": "cal_missing",
            "candidates": [
                {"profile_name": "cal_missing", "score": 0.2},
                {"profile_name": "orig_other", "score": 0.1},
                {"profile_name": "cal_other", "score": 0.3},
            ],
        },
    )

    selector_manifest = module.build_teacher_profile_selector_manifest(
        adaptive_manifest_path=adaptive_manifest_path,
        output_manifest_path=tmp_path / "teacher_profile_selector_manifest.json",
        min_score_delta=0.0,
    )

    assert selector_manifest["status"] == "blocked_selected_calibrated_missing_original_pair"
    assert selector_manifest["next_action"] == "reject_selected_calibrated_candidate_and_register_new_vocal_route"
    assert selector_manifest["contract"]["delivery_allowed"] is False


def test_teacher_profile_selector_blocks_unknown_selected_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    adaptive_manifest_path = tmp_path / "adaptive_vocal_manifest.json"
    module.write_json(
        adaptive_manifest_path,
        {
            "selected_profile": "cal_not_in_candidates",
            "candidates": [
                {"profile_name": "orig_a", "score": 0.1},
                {"profile_name": "cal_a", "score": 0.2},
            ],
        },
    )

    selector_manifest = module.build_teacher_profile_selector_manifest(
        adaptive_manifest_path=adaptive_manifest_path,
        output_manifest_path=tmp_path / "teacher_profile_selector_manifest.json",
        min_score_delta=0.0,
    )

    assert selector_manifest["status"] == "blocked_invalid_selected_profile"
    assert selector_manifest["selected_profile_valid"] is False
    assert selector_manifest["next_action"] == "reject_invalid_selected_profile_and_rerun_adaptive_audit"
    assert selector_manifest["contract"]["delivery_allowed"] is False
