import json
from pathlib import Path

import numpy as np
import soundfile as sf

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_safe_assets(tmp_path: Path) -> dict[str, str]:
    sample_rate = 8_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    mix_path = tmp_path / "mix.wav"
    vocal_path = tmp_path / "vocal.wav"
    minus_path = tmp_path / "minus.wav"
    map_path = tmp_path / "map.npz"
    sf.write(mix_path, audio, sample_rate)
    sf.write(vocal_path, audio, sample_rate)
    sf.write(minus_path, audio, sample_rate)
    np.savez_compressed(map_path, waveform_residual=np.zeros((4, 1), dtype=np.float32))
    return {
        "mix_path": str(mix_path),
        "vocal_candidate_path": str(vocal_path),
        "minus_candidate_path": str(minus_path),
        "map_path": str(map_path),
    }


def base_safe_row(tmp_path: Path, **overrides: object) -> dict:
    assets = write_safe_assets(tmp_path)
    row = {
        "dataset_id": "dataset-a",
        "track_id": "track-a",
        "run_id": "run-a",
        "window_id": "w0",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "mix_path": assets["mix_path"],
        "vocal_candidate_path": assets["vocal_candidate_path"],
        "minus_candidate_path": assets["minus_candidate_path"],
        "candidate_id": "cand-a",
        "route_id": "route-a",
        "stem_type": "minus",
        "blockers": [],
        "judge_decision": "accept_candidate",
        "pair_compatibility_status": "pair_compatible",
        "error_map_paths": {"npz": assets["map_path"]},
        "safe_for_training": True,
        "unsafe_reason": "",
        "provenance_repaired": True,
        "repaired_fields": ["route_id"],
        "repair_source_files": ["candidate_pool.jsonl"],
        "repair_confidence": 1.0,
        "release_ready": False,
        "training_started": False,
    }
    row.update(overrides)
    return row


def write_repaired_dataset(dataset_dir: Path, rows: list[dict]) -> None:
    write_json(
        dataset_dir / "repaired_audio_error_grounding_manifest.json",
        {
            "rows_total": len(rows),
            "safe_for_training_count_before": 0,
            "safe_for_training_count_after": sum(1 for row in rows if row.get("safe_for_training") is True),
            "unsafe_count_before": len(rows),
            "unsafe_count_after": sum(1 for row in rows if row.get("safe_for_training") is not True),
            "rows_repaired": sum(1 for row in rows if row.get("provenance_repaired") is True),
            "release_ready": False,
            "training_started": False,
        },
    )
    write_json(
        dataset_dir / "provenance_repair_report.json",
        {
            "rows_total": len(rows),
            "safe_for_training_count_before": 0,
            "safe_for_training_count_after": sum(1 for row in rows if row.get("safe_for_training") is True),
            "unsafe_count_before": len(rows),
            "unsafe_count_after": sum(1 for row in rows if row.get("safe_for_training") is not True),
            "rows_repaired": sum(1 for row in rows if row.get("provenance_repaired") is True),
            "rows_unrepairable": 0,
            "release_ready": False,
            "training_started": False,
        },
    )
    write_jsonl(dataset_dir / "repaired_audio_error_window_dataset.jsonl", rows)
    write_jsonl(dataset_dir / "repaired_audio_error_maps_index.jsonl", [])


def test_validator_rejects_unknown_route_id_in_safe_row(tmp_path: Path) -> None:
    from tools.validate_audio_error_dataset_provenance_repair import validate_audio_error_dataset_provenance_repair

    write_repaired_dataset(tmp_path, [base_safe_row(tmp_path, route_id="unknown")])

    result = validate_audio_error_dataset_provenance_repair(tmp_path)

    assert result["ok"] is False
    assert any("safe_row_forbidden_unknown_route_id" in error for error in result["errors"])


def test_validator_rejects_synthetic_candidate_id(tmp_path: Path) -> None:
    from tools.validate_audio_error_dataset_provenance_repair import validate_audio_error_dataset_provenance_repair

    write_repaired_dataset(tmp_path, [base_safe_row(tmp_path, candidate_id="synthetic_cand_001")])

    result = validate_audio_error_dataset_provenance_repair(tmp_path)

    assert result["ok"] is False
    assert any("safe_row_synthetic_candidate_id" in error for error in result["errors"])


def test_validator_rejects_repaired_safe_row_without_source_evidence(tmp_path: Path) -> None:
    from tools.validate_audio_error_dataset_provenance_repair import validate_audio_error_dataset_provenance_repair

    write_repaired_dataset(tmp_path, [base_safe_row(tmp_path, repair_source_files=[])])

    result = validate_audio_error_dataset_provenance_repair(tmp_path)

    assert result["ok"] is False
    assert any("safe_repaired_row_missing_source_evidence" in error for error in result["errors"])


def test_validator_accepts_safe_repaired_row_and_preserves_release_false(tmp_path: Path) -> None:
    from tools.validate_audio_error_dataset_provenance_repair import validate_audio_error_dataset_provenance_repair

    write_repaired_dataset(tmp_path, [base_safe_row(tmp_path)])

    result = validate_audio_error_dataset_provenance_repair(tmp_path)

    assert result["ok"] is True
    assert result["safe_for_training_count_after"] == 1
    assert result["release_ready"] is False
    assert result["training_started"] is False
