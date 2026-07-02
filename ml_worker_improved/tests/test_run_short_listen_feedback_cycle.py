import json
from pathlib import Path

from tools.run_short_listen_feedback_cycle import run_short_listen_feedback_cycle


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _pack(root: Path) -> Path:
    pack = root / "pack.json"
    _write_json(
        pack,
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "aggregate_validation_pending",
            "delivery_allowed": False,
            "presentation_status": "aggregate_validation_pending_not_for_listen_pack",
            "release_status": "aggregate_validation_pending",
            "release_ready": False,
            "can_publish_as_result": False,
            "listenable": False,
            "deliverable": False,
            "aggregate_validation_ready": True,
            "failed_checks": [],
            "coverage_json": str(root / "coverage.json"),
            "items": [
                {
                    "audio_path": "audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                    "track_alias": "vrep",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "audio_path": "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
            "artifact_name_policy": {
                "schema": "short_windows_safe_artifact_names.v1",
                "raw_track_or_stem_ids_in_path": False,
                "provenance_in_manifest": True,
            },
        },
    )
    _write_json(
        root / "coverage.json",
        {
            "schema": "aggregate_short_listen_pack_coverage.v1",
            "pack_id": "lp_test",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "stem_groups": [
                {
                    "group_alias": "six",
                    "item_count": 7,
                    "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
                    "semantic_quality_verified": False,
                }
            ],
        },
    )
    _write_json(
        root / "validation" / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )
    return pack


def _make_machine_ready_pack(pack: Path) -> list[str]:
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["items"].extend(
        [
            {
                "audio_path": "audio/vrep/sub/v002.flac",
                "source_alias": "vrep2",
                "stem_type": "vocal",
                "track_alias": "vrep",
                "release_ready": False,
                "can_publish_as_result": False,
            },
            {
                "audio_path": "audio/vrep/sub/v003.flac",
                "source_alias": "vrep3",
                "stem_type": "vocal",
                "track_alias": "vrep",
                "release_ready": False,
                "can_publish_as_result": False,
            },
        ]
    )
    for item in payload["items"]:
        item["source_manifest"] = str(pack.parent / "src" / f"{item['source_alias']}.json")
        item["source_audio_path"] = str(pack.parent / item["audio_path"])
        item["source_row"] = {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
        }
    payload["item_count"] = 25
    payload["release_status"] = "aggregate_validation_pending"
    payload["deliverable"] = False
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    coverage = json.loads((pack.parent / "coverage.json").read_text(encoding="utf-8"))
    coverage["stem_groups"] = [
        {
            "group_alias": f"six{index:03d}",
            "item_count": 7,
            "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
            "semantic_quality_verified": False,
        }
        for index in range(1, 4)
    ]
    (pack.parent / "coverage.json").write_text(json.dumps(coverage, sort_keys=True), encoding="utf-8")
    return [str(item["audio_path"]) for item in payload["items"]]


def _machine_gate(root: Path, *, pack: Path) -> Path:
    validation = pack.parent / "validation" / "validation.json"
    _write_json(
        root / "semantic" / "review.json",
        {
            "schema": "short_listen_semantic_review.v1",
            "review_status": "passed",
            "failed_checks": [],
            "pack_manifest_json": str(pack),
        },
    )
    _write_json(
        root / "minus" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed",
            "failed_checks": [],
            "pack_manifest_json": str(pack),
        },
    )
    return _write_json(
        root / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted",
            "final_result_allowed": True,
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "pack_manifest_json": str(pack),
            "validation_json": str(validation),
            "minus_audit_json": str(root / "minus" / "audit.json"),
            "minus_audit_status": "completed",
            "semantic_review_json": str(root / "semantic" / "review.json"),
            "semantic_review_evidence_count": 2,
        },
    )


def test_run_short_listen_feedback_cycle_builds_repair_queue_for_rejection(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    _make_machine_ready_pack(pack)
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(
        feedback_rows,
        [
            {
                "audio_path": "audio/mpr/min/m001.flac",
                "verdict": "rejected",
                "issue_tags": ["vocal_bleed", "muffled_minus"],
                "note": "residual vocal and muffled middle",
            }
        ],
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        machine_gate_json=_machine_gate(tmp_path, pack=pack),
        output_dir=tmp_path / "cycle",
    )

    assert manifest["schema"] == "short_listen_feedback_cycle_manifest.v1"
    assert manifest["cycle_status"] == "repair_queue_ready"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert manifest["stages"]["feedback_manifest"]["overall_listen_verdict"] == "rejected"
    assert manifest["stages"]["goal_audit"]["goal_status"] == "human_listen_rejected"
    assert manifest["stages"]["repair_queue"]["queue_status"] == "ready_for_repair_cycle"
    rows = [json.loads(line) for line in (tmp_path / "cycle" / "repair_queue" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["audio_path"] == "audio/mpr/min/m001.flac"
    assert rows[0]["route_intent"] == "minus_repair"
    assert (tmp_path / "cycle" / "cycle.json").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "cycle").rglob("*") if path.is_file() or path.is_dir())


def test_run_short_listen_feedback_cycle_blocks_rejected_feedback_without_machine_gate(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    _make_machine_ready_pack(pack)
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(
        feedback_rows,
        [
            {
                "audio_path": "audio/mpr/min/m001.flac",
                "verdict": "rejected",
                "issue_tags": ["vocal_bleed"],
            }
        ],
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "failed_safety_contract"
    assert "feedback_manifest:machine_gate_json_missing" in manifest["failed_checks"]
    assert "goal_audit" not in manifest["stages"]
    assert "repair_queue" not in manifest["stages"]


def test_run_short_listen_feedback_cycle_fails_closed_on_invalid_feedback(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(feedback_rows, [{"audio_path": "audio/not/in/pack.flac", "verdict": "accepted", "issue_tags": []}])

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "failed_safety_contract"
    assert "feedback_manifest:feedback_audio_path_not_in_pack:audio/not/in/pack.flac" in manifest["failed_checks"]
    assert "repair_queue" not in manifest["stages"]


def test_run_short_listen_feedback_cycle_fails_closed_on_missing_machine_gate_after_acceptance(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    audio_paths = _make_machine_ready_pack(pack)
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(
        feedback_rows,
        [{"audio_path": audio_path, "verdict": "accepted", "issue_tags": []} for audio_path in audio_paths],
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "failed_safety_contract"
    assert "feedback_manifest:machine_gate_json_missing" in manifest["failed_checks"]
    assert "goal_audit" not in manifest["stages"]
    assert "repair_queue" not in manifest["stages"]


def test_run_short_listen_feedback_cycle_accepts_only_with_machine_gate_after_human_acceptance(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    audio_paths = _make_machine_ready_pack(pack)
    feedback_rows = tmp_path / "feedback_items.jsonl"
    _write_jsonl(
        feedback_rows,
        [{"audio_path": audio_path, "verdict": "accepted", "issue_tags": []} for audio_path in audio_paths],
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        machine_gate_json=_machine_gate(tmp_path, pack=pack),
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "human_accepted_no_repair_needed"
    assert manifest["failed_checks"] == []
    assert manifest["machine_gate_json"].endswith("gate/gate.json")
    assert manifest["stages"]["goal_audit"]["goal_status"] == "machine_and_human_accepted"
    assert "repair_queue" not in manifest["stages"]


def test_run_short_listen_feedback_cycle_accepts_utf8_bom_jsonl(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    _make_machine_ready_pack(pack)
    feedback_rows = tmp_path / "feedback_items.jsonl"
    feedback_rows.write_text(
        '\ufeff{"audio_path":"audio/mpr/min/m001.flac","verdict":"rejected","issue_tags":["muffled_minus"]}\n',
        encoding="utf-8",
    )

    manifest = run_short_listen_feedback_cycle(
        pack_manifest_json=pack,
        aggregate_validation_json=pack.parent / "validation" / "validation.json",
        feedback_jsonl=feedback_rows,
        machine_gate_json=_machine_gate(tmp_path, pack=pack),
        output_dir=tmp_path / "cycle",
    )

    assert manifest["cycle_status"] == "repair_queue_ready"
    assert manifest["failed_checks"] == []
