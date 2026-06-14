import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_row(tmp_path: Path, **overrides: object) -> dict:
    row = {
        "dataset_id": "dataset-a",
        "track_id": "track-a",
        "run_id": "run-a",
        "window_id": "track-a__cand-a__minus__0000",
        "window_start_sec": 0.0,
        "window_end_sec": 12.0,
        "mix_path": str(tmp_path / "mix.wav"),
        "vocal_candidate_path": str(tmp_path / "vocal.wav"),
        "minus_candidate_path": str(tmp_path / "minus.wav"),
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
        "error_map_paths": {},
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


def write_source_run(run_dir: Path, row: dict, *, route_id: str = "route-a", pair_status: str = "pair_compatible") -> None:
    write_json(run_dir / "run_manifest.json", {"run_id": "run-a", "track_id": "track-a", "release_ready": False})
    write_jsonl(
        run_dir / "candidate_pool.jsonl",
        [
            {
                "track_id": row["track_id"],
                "run_id": row["run_id"],
                "candidate_id": row["candidate_id"],
                "route_id": route_id,
                "candidate_config_id": route_id,
                "mix_path": row["mix_path"],
                "vocal_path": row["vocal_candidate_path"],
                "minus_path": row["minus_candidate_path"],
                "normalized_vocal_path": row["vocal_candidate_path"],
                "normalized_minus_path": row["minus_candidate_path"],
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "pair_compatibility.jsonl",
        [
            {
                "track_id": row["track_id"],
                "selected_vocal_candidate_id": row["candidate_id"],
                "selected_minus_candidate_id": row["candidate_id"],
                "pair_decision": pair_status,
                "release_ready": False,
            }
        ],
    )
    write_jsonl(
        run_dir / "minus_windowed_judge.jsonl",
        [
            {
                "track_id": row["track_id"],
                "candidate_id": row["candidate_id"],
                "minus_decision": row["judge_decision"],
                "window_metrics": [
                    {
                        "start_sec": row["window_start_sec"],
                        "duration_sec": row["window_end_sec"] - row["window_start_sec"],
                        "minus_decision": row["judge_decision"],
                        "minus_blockers": [],
                    }
                ],
                "release_ready": False,
            }
        ],
    )


def test_audit_marks_missing_route_and_pair_repairable_from_exact_source_rows(tmp_path: Path) -> None:
    from tools.audit_audio_error_dataset_provenance import audit_audio_error_dataset_provenance

    dataset_dir = tmp_path / "dataset"
    row = dataset_row(tmp_path)
    write_dataset(dataset_dir, [row])
    run_dir = tmp_path / "run"
    write_source_run(run_dir, row)

    result = audit_audio_error_dataset_provenance(
        dataset_dir=dataset_dir,
        run_roots=[run_dir],
        output_dir=tmp_path / "audit",
    )
    audit_rows = read_jsonl(tmp_path / "audit" / "audio_error_provenance_audit.jsonl")

    assert result["rows_audited"] == 1
    assert audit_rows[0]["missing_fields"] == ["route_id", "pair_compatibility_status"]
    assert audit_rows[0]["candidate_pool_match_found"] is True
    assert audit_rows[0]["route_id_recoverable"] is True
    assert audit_rows[0]["pair_compatibility_recoverable"] is True
    assert audit_rows[0]["window_judge_match_found"] is True
    assert audit_rows[0]["repairable"] is True
    assert audit_rows[0]["repair_values"]["route_id"] == "route-a"
    assert audit_rows[0]["repair_values"]["pair_compatibility_status"] == "pair_compatible"


def test_audit_refuses_filename_only_route_guess(tmp_path: Path) -> None:
    from tools.audit_audio_error_dataset_provenance import audit_audio_error_dataset_provenance

    dataset_dir = tmp_path / "dataset"
    row = dataset_row(
        tmp_path,
        candidate_id="",
        unsafe_reason="missing_candidate_id;missing_route_id;missing_pair_compatibility_status",
    )
    write_dataset(dataset_dir, [row])
    run_dir = tmp_path / "run"
    source_row = {
        **row,
        "candidate_id": "cand-from-other-dir",
        "vocal_candidate_path": str(tmp_path / "other" / Path(row["vocal_candidate_path"]).name),
        "minus_candidate_path": str(tmp_path / "other" / Path(row["minus_candidate_path"]).name),
    }
    write_source_run(run_dir, source_row, route_id="route-from-filename-only")

    audit_audio_error_dataset_provenance(
        dataset_dir=dataset_dir,
        run_roots=[run_dir],
        output_dir=tmp_path / "audit",
    )
    audit_rows = read_jsonl(tmp_path / "audit" / "audio_error_provenance_audit.jsonl")

    assert audit_rows[0]["candidate_pool_match_found"] is False
    assert audit_rows[0]["route_id_recoverable"] is False
    assert audit_rows[0]["repairable"] is False
    assert "missing_candidate_id" in audit_rows[0]["unrepairable_reason"]
