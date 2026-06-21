from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_aris_next_run_plan import build_full_pack_aris_next_run_plan


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _input_file(tmp_path: Path, name: str) -> Path:
    return _write_json(tmp_path / "in" / name, {"schema": f"{name}.v1"})


def _preflight(
    root: Path,
    *,
    status: str,
    candidate_roots: list[str],
    blocked_rows: list[dict] | None = None,
    delivery_allowed: bool = False,
    failed_checks: list[str] | None = None,
) -> Path:
    blocked = _write_jsonl(root / "blocked.jsonl", blocked_rows or [])
    ready_rows = [
        {
            "schema": "full_pack_reference_bundle_preflight_item.v1",
            "bundle_alias": f"r{index:03d}",
            "stem_type": "minus",
            "candidate_root": candidate_root,
            "preflight_status": "ready_for_reference_intake",
            "blockers": [],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        }
        for index, candidate_root in enumerate(candidate_roots, start=1)
    ]
    ready = _write_jsonl(root / "ready.jsonl", ready_rows)
    return _write_json(
        root / "preflight.json",
        {
            "schema": "full_pack_reference_bundle_preflight_manifest.v1",
            "manifest_type": "full_pack_reference_bundle_preflight",
            "preflight_status": status,
            "failed_checks": failed_checks or [],
            "source_handoff_json": str(root / "handoff.json"),
            "handoff_root": str(root.parent / "hand"),
            "handoff_item_count": len(candidate_roots) + len(blocked_rows or []),
            "ready_bundle_count": len(candidate_roots),
            "blocked_bundle_count": len(blocked_rows or []),
            "ready_by_stem": {"minus": len(candidate_roots)} if candidate_roots else {},
            "blocked_by_stem": {"minus": len(blocked_rows or [])} if blocked_rows else {},
            "remaining_by_stem": {"minus": len(blocked_rows or [])} if blocked_rows else {},
            "candidate_roots": candidate_roots,
            "outputs": {
                "manifest_json": str(root / "preflight.json"),
                "ready_jsonl": str(ready),
                "blocked_jsonl": str(blocked),
                "progress_jsonl": str(root / "progress.jsonl"),
                "summary_md": str(root / "summary.md"),
            },
            "delivery_allowed": delivery_allowed,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "audio_decoded": False,
                "dataset_built": False,
                "training_started": False,
                "not_release_artifact": True,
            },
        },
    )


def _recovery_handoff(root: Path, *, candidate_roots: list[str]) -> Path:
    ready_rows = [
        {
            "schema": "full_pack_raw_reference_recovery_material_handoff_item.v1",
            "source_action_item_id": f"a{index:03d}",
            "execution_item_id": f"x{index:03d}",
            "source_bundle_alias": f"b{index:03d}",
            "route_intent": "replace_missing_requested_stem_candidate",
            "required_material": "missing_requested_stem_candidate",
            "target_stem_type": "guitar",
            "requested_stems": ["guitar"],
            "handoff_status": "ready_for_reference_corpus_intake",
            "candidate_root": candidate_root,
            "material_root": candidate_root,
            "ground_truth_manifest_json": str(Path(candidate_root) / "ground_truth_manifest.json"),
            "blockers": [],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        }
        for index, candidate_root in enumerate(candidate_roots, start=1)
    ]
    ready = _write_jsonl(root / "ready.jsonl", ready_rows)
    return _write_json(
        root / "handoff.json",
        {
            "schema": "full_pack_raw_reference_recovery_material_handoff_manifest.v1",
            "manifest_type": "full_pack_raw_reference_recovery_material_handoff",
            "handoff_status": "ready_for_reference_corpus_intake"
            if candidate_roots
            else "blocked_waiting_for_recovery_reference_material",
            "failed_checks": [],
            "handoff_item_count": len(ready_rows),
            "ready_handoff_count": len(ready_rows),
            "blocked_handoff_count": 0,
            "candidate_roots": candidate_roots,
            "outputs": {
                "manifest_json": str(root / "handoff.json"),
                "ready_jsonl": str(ready),
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
            "can_publish_as_result": False,
            "safety": {
                "metadata_only": True,
                "audio_read": False,
                "audio_copied": False,
                "audio_decoded": False,
                "dataset_built": False,
                "training_started": False,
                "not_release_artifact": True,
            },
        },
    )


def _build(
    tmp_path: Path,
    preflight_json: Path,
    *,
    next_output_dir: Path | None = None,
    raw_reference_recovery_material_handoff_json: Path | None = None,
) -> dict:
    return build_full_pack_aris_next_run_plan(
        preflight_json=preflight_json,
        machine_gate_json=_input_file(tmp_path, "gate.json"),
        repair_queue_json=_input_file(tmp_path, "repair.json"),
        full_pack_loop_plan_json=_input_file(tmp_path, "plan.json"),
        reference_gap_plan_json=_input_file(tmp_path, "gap.json"),
        existing_audio_training_examples_jsonl=_write_jsonl(tmp_path / "in" / "audio.jsonl", []),
        raw_reference_recovery_material_handoff_json=raw_reference_recovery_material_handoff_json,
        output_dir=tmp_path / "out",
        next_output_dir=next_output_dir or tmp_path / "loop006",
        min_unique_tracks=5,
        training_output_root="/app/lab_artifacts/tr",
        command_args=["--preflight-json", str(preflight_json)],
    )


def test_next_run_plan_builds_ready_metadata_loop_command_from_preflight(tmp_path: Path) -> None:
    ready_roots = [str(tmp_path / "hand" / "refs" / "r001"), str(tmp_path / "hand" / "refs" / "r002")]
    preflight_json = _preflight(
        tmp_path / "pre",
        status="ready_for_reference_intake",
        candidate_roots=ready_roots,
    )

    manifest = _build(tmp_path, preflight_json, next_output_dir=tmp_path / "fullloop006")

    assert manifest["schema"] == "full_pack_aris_next_run_plan_manifest.v1"
    assert manifest["next_run_status"] == "ready_for_next_metadata_loop"
    assert manifest["candidate_roots"] == ready_roots
    assert manifest["ready_reference_bundle_count"] == 2
    assert manifest["blocked_reference_bundle_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_decoded"] is False

    args = manifest["next_command_args"]
    assert args.count("--candidate-root") == 2
    assert str(tmp_path / "fullloop006") in args
    assert "--min-unique-tracks" in args
    assert "--training-output-root" in args

    outputs = manifest["outputs"]
    assert Path(outputs["manifest_json"]).name == "next.json"
    assert Path(outputs["next_command_args_jsonl"]).name == "args.jsonl"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())
    assert [row["arg"] for row in _read_jsonl(Path(outputs["next_command_args_jsonl"]))] == args


def test_next_run_plan_includes_recovery_material_handoff_roots(tmp_path: Path) -> None:
    root = str(tmp_path / "clean" / "g001")
    preflight_json = _preflight(
        tmp_path / "pre",
        status="blocked_waiting_for_reference_bundles",
        candidate_roots=[],
    )
    handoff_json = _recovery_handoff(tmp_path / "rhand", candidate_roots=[root])

    manifest = _build(
        tmp_path,
        preflight_json,
        raw_reference_recovery_material_handoff_json=handoff_json,
    )

    assert manifest["next_run_status"] == "ready_for_next_metadata_loop"
    assert manifest["candidate_roots"] == [root]
    assert manifest["raw_reference_recovery_material_candidate_roots"] == [root]
    args = manifest["next_command_args"]
    assert args.count("--candidate-root") == 1
    assert args[args.index("--candidate-root") + 1] == root
    assert args.count("--raw-reference-recovery-material-candidate-root") == 1
    assert args[args.index("--raw-reference-recovery-material-candidate-root") + 1] == root
    assert manifest["delivery_allowed"] is False


def test_next_run_plan_uses_recovery_handoff_when_regular_preflight_is_empty(tmp_path: Path) -> None:
    root = str(tmp_path / "clean" / "g001")
    preflight_json = _preflight(
        tmp_path / "pre",
        status="blocked",
        candidate_roots=[],
        failed_checks=[
            "handoff_status_not_ready",
            "handoff_failed_checks_not_empty",
            "handoff_empty",
        ],
    )
    handoff_json = _recovery_handoff(tmp_path / "rhand", candidate_roots=[root])

    manifest = _build(
        tmp_path,
        preflight_json,
        raw_reference_recovery_material_handoff_json=handoff_json,
    )

    assert manifest["next_run_status"] == "ready_for_next_metadata_loop"
    assert manifest["failed_checks"] == []
    assert manifest["preflight_failed_checks_treated_as_nonfatal"] is True
    assert manifest["candidate_roots"] == [root]
    assert manifest["ready_reference_bundle_count"] == 0
    assert manifest["ready_raw_reference_recovery_material_count"] == 1
    assert manifest["raw_reference_recovery_material_candidate_roots"] == [root]
    assert root in manifest["next_command_args"]
    assert "--raw-reference-recovery-material-candidate-root" in manifest["next_command_args"]


def test_next_run_plan_blocks_when_preflight_has_no_ready_roots(tmp_path: Path) -> None:
    blocked_rows = [
        {
            "schema": "full_pack_reference_bundle_preflight_item.v1",
            "bundle_alias": "r001",
            "stem_type": "minus",
            "candidate_root": "",
            "preflight_status": "blocked",
            "blockers": ["bundle_directory_missing"],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        }
    ]
    preflight_json = _preflight(
        tmp_path / "pre",
        status="blocked_waiting_for_reference_bundles",
        candidate_roots=[],
        blocked_rows=blocked_rows,
    )

    manifest = _build(tmp_path, preflight_json)

    assert manifest["next_run_status"] == "blocked_waiting_for_reference_bundles"
    assert manifest["candidate_roots"] == []
    assert manifest["next_command_args"] == []
    assert manifest["blocked_by_stem"] == {"minus": 1}
    assert manifest["blocked_reason_counts"] == {"bundle_directory_missing": 1}
    assert _read_jsonl(Path(manifest["outputs"]["blocked_jsonl"])) == blocked_rows


def test_next_run_plan_blocks_unsafe_preflight_manifest(tmp_path: Path) -> None:
    preflight_json = _preflight(
        tmp_path / "pre",
        status="ready_for_reference_intake",
        candidate_roots=[str(tmp_path / "hand" / "refs" / "r001")],
        delivery_allowed=True,
    )

    manifest = _build(tmp_path, preflight_json)

    assert manifest["next_run_status"] == "blocked"
    assert manifest["candidate_roots"] == []
    assert manifest["next_command_args"] == []
    assert "preflight_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_json(Path(manifest["outputs"]["manifest_json"]))["next_run_status"] == "blocked"
