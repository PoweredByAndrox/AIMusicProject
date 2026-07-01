from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "evaluate_minus_repair_delta_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_minus_repair_delta_gate", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def self_judge_manifest(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    rows_path = write_jsonl(tmp_path / name / "rows.jsonl", rows)
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row.get("issues", []):
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1
    blocker_count = sum(1 for row in rows if row.get("blocker"))
    return write_json(
        tmp_path / name / "manifest.json",
        {
            "mode": "instrumental_self_judge_gate",
            "candidate": {
                "label": name,
                "instrumental_path": f"/host_artifacts/{name}/m001.flac",
            },
            "overall": {
                "status": "proxy_pass" if blocker_count == 0 else "proxy_blocked",
                "blocker_count": blocker_count,
                "blocker_ratio": blocker_count / max(1, len(rows)),
                "segment_count": len(rows),
                "issue_counts": issue_counts,
            },
            "segments_jsonl": str(rows_path),
            "delivery_contract": {
                "delivery_allowed": False,
                "presentation_status": "diagnostic_self_judge_not_for_listen_pack",
            },
        },
    )


def row(segment_id: str, *, blocker: bool, issues: list[str]) -> dict:
    return {
        "segment_id": segment_id,
        "candidate_label": "candidate",
        "window_sec": [0.0, 5.0],
        "blocker": blocker,
        "issues": issues,
        "severity_score": float(len(issues)),
    }


def test_delta_gate_blocks_without_blocker_reduction(tmp_path: Path) -> None:
    module = _load_module()
    before = self_judge_manifest(
        tmp_path,
        "before",
        [
            row("segment_0000", blocker=True, issues=["low_rms_muffling_proxy"]),
            row("segment_0001", blocker=True, issues=["vocal_projection_proxy"]),
        ],
    )
    after = self_judge_manifest(
        tmp_path,
        "after",
        [
            row("segment_0000", blocker=True, issues=["low_rms_muffling_proxy"]),
            row("segment_0001", blocker=True, issues=["presence_loss_proxy"]),
        ],
    )

    manifest = module.evaluate_minus_repair_delta_gate(
        before_self_judge_manifest=before,
        after_self_judge_manifest=after,
        output_dir=tmp_path / "gate",
        candidate_id="it018-air",
    )

    assert manifest["gate_status"] == "blocked_no_blocker_reduction"
    assert manifest["failure_taxonomy"] == ["blocker_count_not_reduced"]
    assert manifest["delta"]["blocker_count_delta"] == 0
    assert manifest["delta"]["unsafe_issue_count_delta"] == -1
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False
    assert read_json(tmp_path / "gate" / "gate.json")["gate_status"] == "blocked_no_blocker_reduction"
    assert (tmp_path / "gate" / "rows.jsonl").exists()
    assert (tmp_path / "gate" / "summary.md").exists()


def test_delta_gate_blocks_unsafe_increase_even_with_blocker_reduction(tmp_path: Path) -> None:
    module = _load_module()
    before = self_judge_manifest(
        tmp_path,
        "before",
        [
            row("segment_0000", blocker=True, issues=["low_rms_muffling_proxy"]),
            row("segment_0001", blocker=True, issues=["presence_loss_proxy"]),
            row("segment_0002", blocker=True, issues=["air_loss_proxy"]),
        ],
    )
    after = self_judge_manifest(
        tmp_path,
        "after",
        [
            row("segment_0000", blocker=False, issues=[]),
            row("segment_0001", blocker=True, issues=["presence_loss_proxy", "vocal_projection_proxy"]),
            row("segment_0002", blocker=True, issues=["air_loss_proxy"]),
        ],
    )

    manifest = module.evaluate_minus_repair_delta_gate(
        before_self_judge_manifest=before,
        after_self_judge_manifest=after,
        output_dir=tmp_path / "gate",
        candidate_id="it018-air",
    )

    assert manifest["gate_status"] == "blocked_unsafe_risk_increase"
    assert manifest["failure_taxonomy"] == ["unsafe_issue_count_increased"]
    assert manifest["delta"]["blocker_count_delta"] == -1
    assert manifest["delta"]["unsafe_issue_count_delta"] == 1
    assert manifest["delivery_allowed"] is False


def test_delta_gate_passes_candidate_routing_only_on_blocker_reduction_without_unsafe_increase(
    tmp_path: Path,
) -> None:
    module = _load_module()
    before = self_judge_manifest(
        tmp_path,
        "before",
        [
            row("segment_0000", blocker=True, issues=["low_rms_muffling_proxy"]),
            row("segment_0001", blocker=True, issues=["presence_loss_proxy"]),
            row("segment_0002", blocker=True, issues=["vocal_projection_proxy"]),
        ],
    )
    after = self_judge_manifest(
        tmp_path,
        "after",
        [
            row("segment_0000", blocker=False, issues=[]),
            row("segment_0001", blocker=True, issues=["presence_loss_proxy"]),
            row("segment_0002", blocker=False, issues=[]),
        ],
    )

    manifest = module.evaluate_minus_repair_delta_gate(
        before_self_judge_manifest=before,
        after_self_judge_manifest=after,
        output_dir=tmp_path / "gate",
        candidate_id="it018-air",
    )

    assert manifest["gate_status"] == "passed_for_candidate_routing"
    assert manifest["failure_taxonomy"] == []
    assert manifest["delta"]["blocker_count_delta"] == -2
    assert manifest["delta"]["unsafe_issue_count_delta"] == -1
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "delta_gate_not_for_listen_pack"
    assert manifest["requires_downstream_acceptance"] is True


def test_delta_gate_blocks_missing_segments_jsonl(tmp_path: Path) -> None:
    module = _load_module()
    before = write_json(
        tmp_path / "before" / "manifest.json",
        {
            "mode": "instrumental_self_judge_gate",
            "overall": {"status": "proxy_blocked", "blocker_count": 1, "segment_count": 1},
            "delivery_contract": {"delivery_allowed": False},
        },
    )
    after = self_judge_manifest(
        tmp_path,
        "after",
        [row("segment_0000", blocker=False, issues=[])],
    )

    manifest = module.evaluate_minus_repair_delta_gate(
        before_self_judge_manifest=before,
        after_self_judge_manifest=after,
        output_dir=tmp_path / "gate",
        candidate_id="it018-air",
    )

    assert manifest["gate_status"] == "blocked_missing_evidence"
    assert "before_segments_jsonl_missing" in manifest["failure_taxonomy"]
    assert manifest["delivery_allowed"] is False
