from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import validate_artifact_relative_path
from docker_runtime import require_docker_runtime


MANIFEST_NAME = "manifest.json"
WORK_ORDER_NAME = "order.json"
PROGRESS_NAME = "progress.jsonl"
SUMMARY_NAME = "summary.md"
START_TIME = time.monotonic()
TARGET_STEMS = ["vocal", "minus"]


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
        "expected_acceleration_mode": "external_only_cpu_manifest_build",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "work_order_manifest_only",
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _short_output_root(index: int) -> str:
    return f"audio/t{index:03d}"


def _work_item(route: dict[str, Any], *, index: int) -> dict[str, Any]:
    route_id = str(route.get("route_id") or f"route_{index:03d}")
    output_root = _short_output_root(index)
    validate_artifact_relative_path(output_root)
    return {
        "schema": "aris_top_route_work_item.v1",
        "id": f"w{index:03d}",
        "route_id": route_id,
        "rank": index,
        "rank_score": int(route.get("rank_score") or 0),
        "candidate_matrix": {
            "families": [str(item) for item in route.get("candidate_families") or []],
            "stems": list(TARGET_STEMS),
            "pair_required": True,
            "segment_level_routing_required": route_id == "holdout_error_map_segment_router",
        },
        "required_gates": [str(item) for item in route.get("required_gates") or []],
        "matched_failures": dict(route.get("matched_failures") or {}),
        "output_root": output_root,
        "status": "ready_for_gpu_candidate_generation",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "forbidden_as_delivery": ["metadata_only", "fallback", "preview_only", "best_of_bad_candidate"],
    }


def _source_matrix(matrix_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(matrix_manifest.get("status") or ""),
        "delivery_status": str(matrix_manifest.get("delivery_status") or ""),
        "accepted_count": int(matrix_manifest.get("accepted_count") or 0),
        "rejected_count": int(matrix_manifest.get("rejected_count") or 0),
        "delivery_allowed": matrix_manifest.get("delivery_allowed") is True,
        "presentation_status": str(matrix_manifest.get("presentation_status") or ""),
        "target_stems": [str(item) for item in matrix_manifest.get("target_stems") or TARGET_STEMS],
    }


def _summary_text(work_order: dict[str, Any]) -> str:
    route_lines = [
        f"- `{item['route_id']}`: families={','.join(item['candidate_matrix']['families'])}; gates={','.join(item['required_gates'])}"
        for item in work_order["work_items"]
    ]
    return "\n".join(
        [
            "# ARIS Top Route Work Order",
            "",
            "Status: ready_for_gpu_candidate_generation; no audio exported; no listen result.",
            "",
            "## Routes",
            "",
            *route_lines,
            "",
            "## Contract",
            "",
            "- Docker GPU candidate generation is required for the next runner.",
            "- Rejected, metadata-only, fallback, preview, and best-of-bad rows are not deliverable.",
            "- Qwen/controller, diffusion repair, and repeat teacher-mask are forbidden primary directions.",
            "",
        ]
    )


def build_aris_top_route_work_order(
    *,
    artifact_rag_root: Path,
    funnel_json: Path,
    matrix_manifest_json: Path,
    output_dir: Path,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    funnel_path = _ensure_path_inside_rag(artifact_rag_root=root, path=funnel_json, label="funnel_json")
    matrix_path = _ensure_path_inside_rag(
        artifact_rag_root=root,
        path=matrix_manifest_json,
        label="matrix_manifest_json",
    )
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    _append_progress(progress_path, "run_start", funnel_json=str(funnel_path), matrix_manifest_json=str(matrix_path))

    funnel = _read_json(funnel_path)
    matrix_manifest = _read_json(matrix_path)
    top_routes = list(funnel.get("top_routes") or [])[:2]
    if not top_routes:
        raise ValueError("funnel_json must contain at least one top route")
    work_items = [_work_item(route, index=index) for index, route in enumerate(top_routes, start=1)]
    selected_route_ids = [item["route_id"] for item in work_items]
    _append_progress(progress_path, "routes_locked", selected_route_ids=selected_route_ids, work_item_count=len(work_items))

    work_order = {
        "schema": "aris_top_route_work_order.v1",
        "status": "ready_for_gpu_candidate_generation",
        "source_funnel_json": str(funnel_path),
        "source_matrix_manifest_json": str(matrix_path),
        "source_matrix": _source_matrix(matrix_manifest),
        "selected_route_ids": selected_route_ids,
        "work_items": work_items,
        "blocked_old_directions": dict(funnel.get("blocked_old_directions") or {}),
        "execution_contract": {
            "rag_controlled_outputs_only": True,
            "docker_only": True,
            "gpu_candidate_generation_required": True,
            "manual_listening_required": False,
            "reference_or_no_reference_gates_required": True,
            "paired_vocal_minus_required": True,
            "rejected_artifacts_not_deliverable": True,
            "metadata_only_rows_blocked": True,
            "fallback_as_success_forbidden": True,
            "preview_as_success_forbidden": True,
            "per_track_hardcode_forbidden": True,
            "qwen_controller_forbidden": True,
            "diffusion_repair_forbidden": True,
            "repeat_teacher_mask_forbidden": True,
        },
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
    }
    work_order_path = output / WORK_ORDER_NAME
    _write_json(work_order_path, work_order)
    _write_jsonl(output / "items.jsonl", work_items)

    manifest = {
        "schema": "aris_top_route_work_order_manifest.v1",
        "manifest_type": "aris_top_route_work_order",
        "status": "ready_for_gpu_candidate_generation",
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "funnel_json": str(funnel_path),
        "matrix_manifest_json": str(matrix_path),
        "work_order_json": str(work_order_path),
        "items_jsonl": str(output / "items.jsonl"),
        "summary_md": str(output / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "selected_route_ids": selected_route_ids,
        "work_item_count": len(work_items),
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "audio_read": False,
        "audio_exported": False,
        "models_changed": False,
        "thresholds_relaxed": False,
        "command_args": list(command_args or []),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(_summary_text(work_order), encoding="utf-8")
    _append_progress(progress_path, "run_finish", status=manifest["status"], manifest_json=str(output / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize strict ARIS top-route work order from idea funnel.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--funnel-json", type=Path, required=True)
    parser.add_argument("--matrix-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_aris_top_route_work_order.py",
        entrypoint_path="/app/tools/build_aris_top_route_work_order.py",
    )
    args = parse_args(argv)
    manifest = build_aris_top_route_work_order(
        artifact_rag_root=args.artifact_rag_root,
        funnel_json=args.funnel_json,
        matrix_manifest_json=args.matrix_manifest_json,
        output_dir=args.output_dir,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
