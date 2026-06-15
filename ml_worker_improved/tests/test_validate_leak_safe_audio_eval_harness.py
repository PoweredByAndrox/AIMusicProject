import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_validator_accepts_zero_critical_leakage_manifest(tmp_path: Path) -> None:
    from tools.validate_leak_safe_audio_eval_harness import validate_leak_safe_audio_eval_harness

    output_dir = tmp_path / "eval"
    write_json(
        output_dir / "leak_safe_split_manifest.json",
        {
            "schema": "leak_safe_audio_eval_harness.v1",
            "critical_leakage_count": 0,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "model_2c_locked_re_eval": {"release_ready": False},
        },
    )
    write_json(output_dir / "leakage_report.json", {"critical_leakage_count": 0, "checks": {}})
    write_json(output_dir / "hardcase_holdout_manifest.json", {"locked": True, "release_ready": False})
    (output_dir / "audio_fingerprint_index.jsonl").write_text("{}", encoding="utf-8")
    (output_dir / "locked_eval_summary.md").write_text("- release_ready: false\n", encoding="utf-8")

    result = validate_leak_safe_audio_eval_harness(output_dir)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["release_ready"] is False


def test_validator_rejects_critical_leakage_and_release_ready(tmp_path: Path) -> None:
    from tools.validate_leak_safe_audio_eval_harness import validate_leak_safe_audio_eval_harness

    output_dir = tmp_path / "eval"
    write_json(
        output_dir / "leak_safe_split_manifest.json",
        {
            "schema": "leak_safe_audio_eval_harness.v1",
            "critical_leakage_count": 1,
            "release_ready": True,
            "training_started": False,
            "separator_training_started": False,
            "model_2c_locked_re_eval": {"release_ready": False},
        },
    )
    write_json(
        output_dir / "leakage_report.json",
        {
            "critical_leakage_count": 1,
            "checks": {"candidate_id_overlap": {"critical": True}},
        },
    )
    write_json(output_dir / "hardcase_holdout_manifest.json", {"locked": True, "release_ready": False})
    (output_dir / "audio_fingerprint_index.jsonl").write_text("{}", encoding="utf-8")
    (output_dir / "locked_eval_summary.md").write_text("- release_ready: false\n", encoding="utf-8")

    result = validate_leak_safe_audio_eval_harness(output_dir)

    assert result["ok"] is False
    assert "critical_leakage_present" in result["errors"]
    assert "release_ready_true" in result["errors"]


def test_validator_requires_model2c_eval_when_requested(tmp_path: Path) -> None:
    from tools.validate_leak_safe_audio_eval_harness import validate_leak_safe_audio_eval_harness

    output_dir = tmp_path / "eval"
    write_json(
        output_dir / "leak_safe_split_manifest.json",
        {
            "schema": "leak_safe_audio_eval_harness.v1",
            "critical_leakage_count": 0,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "model_2c_re_eval_required": True,
            "model_2c_locked_re_eval": {
                "status": "skipped_no_target_evaluated",
                "release_ready": False,
                "targets": {"minus": {"status": "failed_diagnostic_eval"}},
            },
        },
    )
    write_json(output_dir / "leakage_report.json", {"critical_leakage_count": 0, "checks": {}})
    write_json(output_dir / "hardcase_holdout_manifest.json", {"locked": True, "release_ready": False})
    (output_dir / "audio_fingerprint_index.jsonl").write_text("{}", encoding="utf-8")
    (output_dir / "locked_eval_summary.md").write_text("- release_ready: false\n", encoding="utf-8")

    result = validate_leak_safe_audio_eval_harness(output_dir)

    assert result["ok"] is False
    assert "model_2c_re_eval_not_evaluated" in result["errors"]
    assert "model_2c_re_eval_failed:minus" in result["errors"]
