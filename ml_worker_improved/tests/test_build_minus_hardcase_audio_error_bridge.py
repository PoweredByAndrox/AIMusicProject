import json
from pathlib import Path

from tools import build_minus_hardcase_audio_error_bridge as bridge


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def passed_mount_preflight(output_dir: Path) -> dict:
    return {
        "schema": bridge.MOUNT_PREFLIGHT_SCHEMA,
        "preflight_status": "passed",
        "blocked_reason": "",
        "can_write_route_artifacts": True,
        "mac_windows_project_mount": "/Users/andrejviluha/AIMusicProject_win",
        "mac_output_path": f"/Users/andrejviluha/AIMusicProject_win/artifacts/{output_dir.name}",
        "container_output_path": str(output_dir),
        "windows_project_root": r"D:\AIMusicProject",
        "windows_output_path": rf"D:\AIMusicProject\artifacts\{output_dir.name}",
        "allowed_project_artifact_roots": ["artifacts", "lab_artifacts", "artifact_rag"],
        "relative_output_path": f"artifacts/{output_dir.name}",
        "probe_status": "write_read_delete_passed",
        "probe_path": f"/Users/andrejviluha/AIMusicProject_win/.aimusic_mount_probe/probe_{output_dir.name}.txt",
    }


def install_successful_bridge_fakes(monkeypatch) -> None:
    def fake_grounding(*, run_roots, output_dir, **_kwargs):
        write_json(
            output_dir / "audio_error_grounding_manifest.json",
            {
                "dataset_id": "dataset-bridge",
                "windows_total": 2,
                "safe_for_training_count": 2,
                "error_maps_written": 2,
                "release_ready": False,
            },
        )
        write_jsonl(output_dir / "audio_error_window_dataset.jsonl", [{"row": 1}, {"row": 2}])
        write_jsonl(output_dir / "audio_error_maps_index.jsonl", [{"map": 1}, {"map": 2}])
        return {
            "dataset_id": "dataset-bridge",
            "run_roots_expanded": [str(path) for path in run_roots],
            "windows_total": 2,
            "safe_for_training_count": 2,
            "error_maps_written": 2,
            "release_ready": False,
        }

    def fake_audit(*, dataset_dir, run_roots, output_dir):
        write_json(
            output_dir / "audio_error_provenance_audit.json",
            {"rows_audited": 0, "rows_repairable": 0, "release_ready": False},
        )
        write_jsonl(output_dir / "audio_error_provenance_audit.jsonl", [])
        return {
            "dataset_dir": str(dataset_dir),
            "source_files_indexed": [str(path / "candidate_pool.jsonl") for path in run_roots],
            "rows_audited": 0,
            "rows_repairable": 0,
            "release_ready": False,
        }

    def fake_repair(*, dataset_dir, audit_dir, output_dir):
        assert dataset_dir.name == "ground"
        assert audit_dir.name == "audit"
        write_json(
            output_dir / "repaired_audio_error_grounding_manifest.json",
            {"dataset_id": "dataset-bridge", "release_ready": False},
        )
        write_jsonl(output_dir / "repaired_audio_error_window_dataset.jsonl", [{"safe_for_training": True}])
        write_jsonl(output_dir / "repaired_audio_error_maps_index.jsonl", [{"map": 1}])
        write_json(
            output_dir / "provenance_repair_report.json",
            {
                "safe_for_training_count_before": 2,
                "safe_for_training_count_after": 2,
                "unsafe_count_after": 0,
                "release_ready": False,
            },
        )
        return {
            "safe_for_training_count_before": 2,
            "safe_for_training_count_after": 2,
            "unsafe_count_after": 0,
            "release_ready": False,
        }

    def fake_validate_repair(_dataset_dir):
        return {
            "ok": True,
            "errors": [],
            "rows_total": 2,
            "safe_for_training_count_after": 2,
            "release_ready": False,
        }

    def fake_views(*, dataset_dir, output_dir, run_roots):
        assert dataset_dir.name == "repaired"
        write_jsonl(
            output_dir / "minus_critic_training_view.jsonl",
            [{"row_id": "minus-1", "safe_for_this_view": True, "label": 0, "release_ready": False}],
        )
        manifest = {
            "schema": "audio_error_training_views.v1",
            "dataset_dir": str(dataset_dir),
            "source_run_rows_added": 2,
            "strict_negative_rows_found": 1,
            "strict_positive_rows_found": 1,
            "can_train_minus_critic": True,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
            "run_roots": [str(path) for path in run_roots],
        }
        write_json(output_dir / "audio_error_training_views_manifest.json", manifest)
        return manifest

    def fake_validate_views(_views_dir):
        return {
            "ok": True,
            "errors": [],
            "can_train_minus_critic": True,
            "can_train_vocal_critic": False,
            "can_train_pair_critic": False,
            "can_train_overall_accept": False,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        }

    monkeypatch.setattr(bridge, "build_audio_error_grounding_dataset", fake_grounding)
    monkeypatch.setattr(bridge, "audit_audio_error_dataset_provenance", fake_audit)
    monkeypatch.setattr(bridge, "repair_audio_error_dataset_provenance", fake_repair)
    monkeypatch.setattr(bridge, "validate_audio_error_dataset_provenance_repair", fake_validate_repair)
    monkeypatch.setattr(bridge, "build_audio_error_training_views", fake_views)
    monkeypatch.setattr(bridge, "validate_audio_error_training_views", fake_validate_views)


def test_bridge_links_runtime_roots_to_grounding_repair_and_training_views(tmp_path: Path, monkeypatch) -> None:
    install_successful_bridge_fakes(monkeypatch)
    run_root = tmp_path / "run"
    run_root.mkdir()
    output_dir = tmp_path / "bridge"

    manifest = bridge.bridge_minus_hardcase_audio_error_views(
        run_roots=[run_root],
        output_dir=output_dir,
        mount_preflight=passed_mount_preflight(output_dir),
        require_mount_preflight=True,
        command_args=["--run-root", str(run_root)],
    )

    saved = read_json(output_dir / "bridge_manifest.json")
    progress = read_jsonl(output_dir / "progress.jsonl")
    assert manifest["schema"] == "minus_hardcase_audio_error_bridge.v1"
    assert manifest["output_status"] == "training_views_ready"
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["listen_delivery_artifact"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["separator_training_started"] is False
    assert manifest["source_run_rows_added"] == 2
    assert manifest["strict_negative_rows_found"] == 1
    assert manifest["strict_positive_rows_found"] == 1
    assert manifest["can_train_minus_critic"] is True
    assert manifest["mount_preflight_status"] == "passed"
    assert manifest["mount_probe_status"] == "write_read_delete_passed"
    assert manifest["windows_output_path"].endswith(r"artifacts\bridge")
    assert saved["mount_preflight"]["can_write_route_artifacts"] is True
    assert saved["views_dir"].endswith("views")
    assert progress[0]["mount_preflight_status"] == "passed"
    assert progress[0]["windows_output_path"].endswith(r"artifacts\bridge")
    assert [row["phase"] for row in progress] == [
        "start",
        "grounding_start",
        "grounding_done",
        "provenance_audit_start",
        "provenance_audit_done",
        "provenance_repair_start",
        "provenance_repair_done",
        "training_views_start",
        "training_views_done",
        "complete",
    ]
    assert all(row["gpu_inference_skipped"] is True for row in progress)


def test_bridge_blocks_when_required_mount_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    def fail_if_grounding_runs(**_kwargs):
        raise AssertionError("grounding must not run after mount preflight failure")

    monkeypatch.setattr(bridge, "build_audio_error_grounding_dataset", fail_if_grounding_runs)
    run_root = tmp_path / "run"
    run_root.mkdir()

    manifest = bridge.bridge_minus_hardcase_audio_error_views(
        run_roots=[run_root],
        output_dir=tmp_path / "bridge",
        mount_preflight={
            "schema": bridge.MOUNT_PREFLIGHT_SCHEMA,
            "preflight_status": "blocked",
            "blocked_reason": "mac_windows_project_mount_missing",
            "can_write_route_artifacts": False,
            "probe_status": "not_run",
            "windows_output_path": "",
        },
        require_mount_preflight=True,
    )

    progress = read_jsonl(tmp_path / "bridge" / "progress.jsonl")
    assert manifest["output_status"] == "blocked_mount_preflight"
    assert manifest["delivery_allowed"] is False
    assert manifest["mount_preflight_status"] == "blocked"
    assert manifest["validation_errors"] == ["mount_preflight:mac_windows_project_mount_missing"]
    assert [row["phase"] for row in progress] == ["start", "blocked_mount_preflight", "complete"]


def test_bridge_blocks_when_training_view_validation_fails(tmp_path: Path, monkeypatch) -> None:
    install_successful_bridge_fakes(monkeypatch)

    def fake_validate_views(_views_dir):
        return {
            "ok": False,
            "errors": ["row_missing_fields:minus_critic:0:candidate_audio_path"],
            "can_train_minus_critic": False,
            "release_ready": False,
            "training_started": False,
            "separator_training_started": False,
        }

    monkeypatch.setattr(bridge, "validate_audio_error_training_views", fake_validate_views)

    manifest = bridge.bridge_minus_hardcase_audio_error_views(
        run_roots=[tmp_path / "run"],
        output_dir=tmp_path / "bridge",
    )

    assert manifest["output_status"] == "blocked_validation_failed"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "training_evidence_not_for_listen_pack"
    assert manifest["validation_errors"] == [
        "training_views:row_missing_fields:minus_critic:0:candidate_audio_path"
    ]


def test_bridge_rejects_legacy_flstudio_runtime_rows_before_grounding(tmp_path: Path, monkeypatch) -> None:
    install_successful_bridge_fakes(monkeypatch)
    run_root = tmp_path / "run"
    write_jsonl(
        run_root / "candidate_pool.jsonl",
        [
            {
                "track_id": "legacy-track",
                "candidate_id": "legacy-candidate",
                "mix_path": "D:/AIMusicProject/testing_flstudio_music/raw.flac",
                "minus_path": "D:/AIMusicProject/artifacts/legacy/minus.flac",
            }
        ],
    )

    manifest = bridge.bridge_minus_hardcase_audio_error_views(
        run_roots=[run_root],
        output_dir=tmp_path / "bridge",
    )

    progress = read_jsonl(tmp_path / "bridge" / "progress.jsonl")
    assert manifest["output_status"] == "blocked_forbidden_legacy_corpus"
    assert manifest["delivery_allowed"] is False
    assert manifest["source_run_rows_added"] == 0
    assert manifest["validation_errors"][0].startswith(
        "forbidden_legacy_corpus_marker:testing_flstudio_music:"
    )
    assert [row["phase"] for row in progress] == [
        "start",
        "blocked_forbidden_legacy_corpus",
        "complete",
    ]
