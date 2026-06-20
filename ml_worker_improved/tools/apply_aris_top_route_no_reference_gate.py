from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import validate_artifact_relative_path
from docker_runtime import require_docker_runtime
from run_no_reference_result_judge import judge_candidate_row


MANIFEST_NAME = "manifest.json"
GATED_POOL_NAME = "gated_pool.jsonl"
JUDGES_NAME = "judges.jsonl"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"
START_TIME = time.monotonic()

JudgeRunner = Callable[..., dict[str, Any]]


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _ensure_path_inside_rag(*, artifact_rag_root: Path, path: Path, label: str) -> Path:
    root = _resolved(artifact_rag_root)
    target = _resolved(path)
    if target != root and root not in target.parents:
        raise ValueError(f"{label} must be inside artifact_rag_root")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_progress(path: Path, event: str, **fields: Any) -> None:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "cpu_dsp_scoring",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "no_reference_gate_reads_generated_audio",
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _failure_labels(judged: dict[str, Any], *, decision_key: str, accepted_value: str, blocker_keys: Sequence[str]) -> list[str]:
    if str(judged.get(decision_key) or "") == accepted_value:
        return []
    labels: list[str] = [str(judged.get(decision_key) or "gate_failed")]
    for key in blocker_keys:
        value = judged.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    labels.append(str(item.get("label") or item.get("reason") or "gate_blocker"))
                else:
                    labels.append(str(item))
    for key in ("failed_checks", "reasons"):
        value = judged.get(key)
        if isinstance(value, list):
            labels.extend(str(item) for item in value if str(item))
    return sorted({label for label in labels if label})


def _mix_path_from_candidate(row: dict[str, Any]) -> str:
    manifest_path_value = str(row.get("candidate_manifest_path") or "")
    if manifest_path_value:
        manifest_path = Path(manifest_path_value)
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            raw_ref = manifest.get("raw_reference") if isinstance(manifest.get("raw_reference"), dict) else {}
            if raw_ref.get("path"):
                return str(raw_ref["path"])
    return str(row.get("source_audio_path") or "")


def _judge_pair(
    *,
    track_id: str,
    candidate_id: str,
    mix_path: str,
    vocal_path: str,
    minus_path: str,
    judge_runner: JudgeRunner,
) -> dict[str, Any]:
    judge_row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": mix_path,
        "vocal_path": vocal_path,
        "instrumental_path": minus_path,
    }
    vocal = judge_runner(judge_row, score_mode="vocal")
    minus = judge_runner(judge_row, score_mode="minus")
    pair = judge_runner(judge_row, score_mode="both")
    failures: list[str] = []
    failures.extend(
        _failure_labels(
            vocal,
            decision_key="vocal_decision",
            accepted_value="accept_candidate",
            blocker_keys=("vocal_blockers",),
        )
    )
    failures.extend(
        _failure_labels(
            minus,
            decision_key="minus_decision",
            accepted_value="accept_candidate",
            blocker_keys=("minus_blockers",),
        )
    )
    detail_gate = minus.get("minus_musical_detail_preservation_gate")
    if isinstance(detail_gate, dict) and detail_gate.get("status") != "proxy_pass":
        failures.extend(
            _failure_labels(
                {"detail_gate": "minus_detail_gate_blocked", "blockers": detail_gate.get("blockers") or []},
                decision_key="detail_gate",
                accepted_value="proxy_pass",
                blocker_keys=("blockers",),
            )
        )
    failures.extend(
        _failure_labels(
            pair,
            decision_key="pair_decision",
            accepted_value="pair_compatible",
            blocker_keys=("pair_blockers",),
        )
    )
    failed_checks = sorted({label for label in failures if label})
    return {
        "schema": "aris_top_route_no_reference_pair_gate.v1",
        "track_id": track_id,
        "candidate_id": candidate_id,
        "accepted": not failed_checks,
        "failed_checks": failed_checks,
        "reference_gate": "not_available",
        "no_reference_gate": "passed" if not failed_checks else "failed",
        "pair_consistency_gate": "passed" if not failed_checks else "failed",
        "vocal_judge": vocal,
        "minus_judge": minus,
        "pair_judge": pair,
    }


def _summary_text(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ARIS Top Route No-Reference Gate",
            "",
            f"- status: `{manifest['status']}`",
            f"- track_count: `{manifest['track_count']}`",
            f"- accepted_pair_count: `{manifest['accepted_pair_count']}`",
            f"- rejected_pair_count: `{manifest['rejected_pair_count']}`",
            "- delivery_allowed: `false`",
            "",
        ]
    )


def apply_aris_top_route_no_reference_gate(
    *,
    artifact_rag_root: Path,
    candidate_pool_jsonl: Path,
    output_dir: Path,
    judge_runner: JudgeRunner | None = None,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    pool_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=candidate_pool_jsonl,
        label="candidate_pool_jsonl",
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    _append_progress(progress_path, "run_start", candidate_pool_jsonl=str(pool_path), output_dir=str(output))
    judge = judge_runner or judge_candidate_row
    rows = list(_iter_jsonl(pool_path))
    rows_by_track: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_track.setdefault(str(row.get("track_alias") or ""), []).append(row)
    gated_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    accepted_pair_count = 0
    rejected_pair_count = 0
    for track_alias in sorted(rows_by_track):
        track_rows = rows_by_track[track_alias]
        vocal = next((row for row in track_rows if str(row.get("stem")) == "vocal"), None)
        minus = next((row for row in track_rows if str(row.get("stem")) == "minus"), None)
        _append_progress(progress_path, "track_gate_start", track_alias=track_alias)
        if vocal is None or minus is None:
            failed_checks = []
            if vocal is None:
                failed_checks.append("vocal_candidate_missing")
            if minus is None:
                failed_checks.append("minus_candidate_missing")
            gate = {
                "schema": "aris_top_route_no_reference_pair_gate.v1",
                "track_id": str(track_rows[0].get("track_id") if track_rows else track_alias),
                "candidate_id": track_alias,
                "accepted": False,
                "failed_checks": failed_checks,
                "reference_gate": "not_available",
                "no_reference_gate": "failed",
                "pair_consistency_gate": "failed",
            }
        else:
            gate = _judge_pair(
                track_id=str(vocal.get("track_id") or minus.get("track_id") or track_alias),
                candidate_id=f"{track_alias}-top",
                mix_path=_mix_path_from_candidate(vocal),
                vocal_path=str(vocal.get("target_audio_path") or ""),
                minus_path=str(minus.get("target_audio_path") or ""),
                judge_runner=judge,
            )
        accepted = bool(gate["accepted"])
        accepted_pair_count += 1 if accepted else 0
        rejected_pair_count += 0 if accepted else 1
        judge_rows.append(gate)
        for row in track_rows:
            gated = dict(row)
            gated.update(
                {
                    "candidate_status": "passed" if accepted else "failed",
                    "reference_gate": gate["reference_gate"],
                    "no_reference_gate": gate["no_reference_gate"],
                    "pair_consistency_gate": gate["pair_consistency_gate"],
                    "failed_checks": list(gate["failed_checks"]),
                    "delivery_allowed": False,
                    "presentation_status": "blocked_from_listen_pack",
                    "release_ready": False,
                }
            )
            gated_rows.append(gated)
        _append_progress(
            progress_path,
            "track_gate_finish",
            track_alias=track_alias,
            status="accepted" if accepted else "rejected",
            failed_checks=list(gate["failed_checks"]),
        )
    gated_path = output / GATED_POOL_NAME
    judges_path = output / JUDGES_NAME
    _write_jsonl(gated_path, gated_rows)
    _write_jsonl(judges_path, judge_rows)
    status = "accepted" if accepted_pair_count and rejected_pair_count == 0 else "blocked"
    manifest = {
        "schema": "aris_top_route_no_reference_gate_manifest.v1",
        "manifest_type": "aris_top_route_no_reference_gate",
        "status": status,
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "candidate_pool_jsonl": str(pool_path),
        "gated_candidate_pool_jsonl": str(gated_path),
        "judges_jsonl": str(judges_path),
        "progress_jsonl": str(progress_path),
        "summary_md": str(output / SUMMARY_NAME),
        "track_count": len(rows_by_track),
        "candidate_count": len(gated_rows),
        "accepted_pair_count": accepted_pair_count,
        "rejected_pair_count": rejected_pair_count,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "audio_exported": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "command_args": list(command_args or []),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(manifest), encoding="utf-8")
    _append_progress(progress_path, "run_finish", status=status, manifest_json=str(output / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply no-reference gates to ARIS top-route candidate pool.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="apply_aris_top_route_no_reference_gate.py",
        entrypoint_path="/app/tools/apply_aris_top_route_no_reference_gate.py",
    )
    args = parse_args(argv)
    manifest = apply_aris_top_route_no_reference_gate(
        artifact_rag_root=args.artifact_rag_root,
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        output_dir=args.output_dir,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
