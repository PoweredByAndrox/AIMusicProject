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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_dummy_npz(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "waveform_residual": np.zeros((8, 2), dtype=np.float32),
        "stft_residual_magnitude": np.zeros((4, 4), dtype=np.float32),
        "phase_difference_map": np.zeros((4, 4), dtype=np.float32),
        "stereo_width_map": np.zeros((4, 4), dtype=np.float32),
        "vocal_bleed_proxy_map": np.zeros((4, 4), dtype=np.float32),
        "instrumental_bleed_proxy_map": np.zeros((4, 4), dtype=np.float32),
        "missing_detail_proxy_map": np.zeros((4, 4), dtype=np.float32),
        "transient_loss_proxy_map": np.zeros((4, 4), dtype=np.float32),
        "artifact_noise_proxy_map": np.zeros((4, 4), dtype=np.float32),
        "reconstruction_error_map": np.zeros((4, 4), dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)
    return str(path)


def write_audio_triplet(tmp_path: Path) -> dict[str, str]:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    vocal = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    minus = 0.1 * np.sin(2.0 * np.pi * 120.0 * t)
    mix = vocal + minus
    paths = {
        "mix_path": str(tmp_path / "mix.wav"),
        "vocal_candidate_path": str(tmp_path / "vocal.wav"),
        "minus_candidate_path": str(tmp_path / "minus.wav"),
    }
    sf.write(paths["mix_path"], mix.astype(np.float32), sample_rate)
    sf.write(paths["vocal_candidate_path"], vocal.astype(np.float32), sample_rate)
    sf.write(paths["minus_candidate_path"], minus.astype(np.float32), sample_rate)
    return paths


def base_row(tmp_path: Path, npz_path: str, **overrides: object) -> dict:
    audio_paths = write_audio_triplet(tmp_path)
    row = {
        "dataset_id": "dataset-a",
        "track_id": "track-a",
        "run_id": "run-a",
        "window_id": "track-a__cand-a__minus__0000",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "mix_path": audio_paths["mix_path"],
        "vocal_candidate_path": audio_paths["vocal_candidate_path"],
        "minus_candidate_path": audio_paths["minus_candidate_path"],
        "candidate_id": "cand-a",
        "route_id": "",
        "stem_type": "minus",
        "blockers": [],
        "judge_decision": "accept_candidate",
        "pair_compatibility_status": "",
        "residual_energy": 0.0,
        "mixture_consistency_error": 0.0,
        "phase_collapse_score": 0.0,
        "stereo_width_score": 0.0,
        "vocal_bleed_proxy_score": 0.0,
        "instrumental_bleed_proxy_score": 0.0,
        "missing_detail_score": 0.0,
        "transient_damage_score": 0.0,
        "artifact_noise_score": 0.0,
        "error_map_paths": {"npz": npz_path},
        "train_split": "train",
        "safe_for_training": False,
        "unsafe_reason": "missing_route_id;missing_pair_compatibility_status",
        "release_ready": False,
        "training_started": False,
    }
    row.update(overrides)
    return row


def write_dataset(dataset_dir: Path, rows: list[dict]) -> None:
    write_json(
        dataset_dir / "audio_error_grounding_manifest.json",
        {
            "dataset_id": "dataset-a",
            "windows_total": len(rows),
            "safe_for_training_count": sum(1 for row in rows if row.get("safe_for_training") is True),
            "unsafe_count": sum(1 for row in rows if row.get("safe_for_training") is not True),
            "release_ready": False,
            "training_started": False,
        },
    )
    write_jsonl(dataset_dir / "audio_error_window_dataset.jsonl", rows)
    write_jsonl(dataset_dir / "audio_error_maps_index.jsonl", [])


def write_audit(audit_dir: Path, row: dict, *, route: str = "route-a", pair_status: str = "pair_compatible") -> None:
    audit_row = {
        "row_id": "0",
        "track_id": row["track_id"],
        "run_id": row["run_id"],
        "candidate_id": row["candidate_id"],
        "missing_fields": ["route_id", "pair_compatibility_status"],
        "candidate_pool_match_found": True,
        "route_id_recoverable": bool(route),
        "pair_compatibility_recoverable": bool(pair_status),
        "window_judge_match_found": True,
        "repairable": bool(route and pair_status),
        "unrepairable_reason": "",
        "repair_values": {
            "route_id": route,
            "pair_compatibility_status": pair_status,
        },
        "repair_source_files": ["candidate_pool.jsonl", "pair_compatibility.jsonl", "minus_windowed_judge.jsonl"],
        "repair_confidence": 1.0,
    }
    write_json(audit_dir / "audio_error_provenance_audit.json", {"rows_audited": 1, "rows_repairable": int(audit_row["repairable"])})
    write_jsonl(audit_dir / "audio_error_provenance_audit.jsonl", [audit_row])


def test_repair_fills_provenance_from_audit_without_overwriting_original(tmp_path: Path) -> None:
    from tools.repair_audio_error_dataset_provenance import repair_audio_error_dataset_provenance

    dataset_dir = tmp_path / "dataset"
    npz_path = write_dummy_npz(tmp_path / "maps" / "row.npz")
    row = base_row(tmp_path, npz_path)
    write_dataset(dataset_dir, [row])
    audit_dir = tmp_path / "audit"
    write_audit(audit_dir, row)

    result = repair_audio_error_dataset_provenance(
        dataset_dir=dataset_dir,
        audit_dir=audit_dir,
        output_dir=tmp_path / "repaired",
    )
    original_rows = read_jsonl(dataset_dir / "audio_error_window_dataset.jsonl")
    repaired_rows = read_jsonl(tmp_path / "repaired" / "repaired_audio_error_window_dataset.jsonl")

    assert original_rows[0]["route_id"] == ""
    assert original_rows[0]["pair_compatibility_status"] == ""
    assert repaired_rows[0]["route_id"] == "route-a"
    assert repaired_rows[0]["pair_compatibility_status"] == "pair_compatible"
    assert repaired_rows[0]["provenance_repaired"] is True
    assert repaired_rows[0]["repaired_fields"] == ["route_id", "pair_compatibility_status"]
    assert repaired_rows[0]["repair_source_files"]
    assert repaired_rows[0]["safe_for_training"] is True
    assert repaired_rows[0]["unsafe_reason"] == ""
    assert result["rows_repaired"] == 1


def test_repair_keeps_row_unsafe_when_pair_status_is_not_proven(tmp_path: Path) -> None:
    from tools.repair_audio_error_dataset_provenance import repair_audio_error_dataset_provenance

    dataset_dir = tmp_path / "dataset"
    npz_path = write_dummy_npz(tmp_path / "maps" / "row.npz")
    row = base_row(tmp_path, npz_path)
    write_dataset(dataset_dir, [row])
    audit_dir = tmp_path / "audit"
    write_audit(audit_dir, row, route="route-a", pair_status="")

    repair_audio_error_dataset_provenance(
        dataset_dir=dataset_dir,
        audit_dir=audit_dir,
        output_dir=tmp_path / "repaired",
    )
    repaired_rows = read_jsonl(tmp_path / "repaired" / "repaired_audio_error_window_dataset.jsonl")

    assert repaired_rows[0]["route_id"] == ""
    assert repaired_rows[0]["pair_compatibility_status"] == ""
    assert repaired_rows[0]["safe_for_training"] is False
    assert "missing_pair_compatibility_status" in repaired_rows[0]["unsafe_reason"]
