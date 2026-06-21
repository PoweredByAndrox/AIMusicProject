from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import SAFETY_FLAGS, append_progress, iter_jsonl, load_rag_current, read_json, stable_hash, write_json
from tools.docker_runtime import require_docker_runtime


PACK_NAME = "pack.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"


def _git_status_short(cwd: Path) -> tuple[bool, list[str]]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return False, []
    if result.returncode != 0:
        return False, []
    return True, [line for line in result.stdout.splitlines() if line.strip()]


def _rag_delivery_state(rag_current_json: Path, rag: dict[str, Any]) -> dict[str, Any]:
    idx_rel = str(rag.get("idx") or "")
    idx_path = rag_current_json.parent / idx_rel
    if not idx_rel or not idx_path.exists():
        return {
            "available": False,
            "reason": "rag_idx_missing",
            "delivery_status_counts": {},
            "aggregate_validation_ref_count": 0,
            "blocked_delivery_ref_count": 0,
        }
    try:
        idx = read_json(idx_path)
        outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
        manifest_refs_path = idx_path.parent / str(outputs.get("manifest_refs") or "manifest_refs.jsonl")
        refs = list(iter_jsonl(manifest_refs_path)) if manifest_refs_path.exists() else []
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"rag_delivery_state_read_failed:{type(exc).__name__}",
            "delivery_status_counts": {},
            "aggregate_validation_ref_count": 0,
            "blocked_delivery_ref_count": 0,
        }
    delivery_status_counts: dict[str, int] = {}
    aggregate_validation_ref_count = 0
    blocked_delivery_ref_count = 0
    for ref in refs:
        status = ref.get("status_fields") if isinstance(ref.get("status_fields"), dict) else {}
        statuses: list[str] = []
        value = str(status.get("delivery_status") or "").strip()
        if value:
            statuses.append(value)
        for sample in status.get("sample_rows") if isinstance(status.get("sample_rows"), list) else []:
            if isinstance(sample, dict) and sample.get("delivery_status"):
                statuses.append(str(sample["delivery_status"]))
        for delivery_status in statuses:
            delivery_status_counts[delivery_status] = delivery_status_counts.get(delivery_status, 0) + 1
        if status.get("schema") == "aggregate_short_listen_pack_validation.v1":
            aggregate_validation_ref_count += 1
        if "blocked" in statuses or status.get("delivery_allowed") is False and status.get("release_ready") is False and status.get("delivery_status") == "blocked":
            blocked_delivery_ref_count += 1
    return {
        "available": True,
        "reason": "",
        "delivery_status_counts": dict(sorted(delivery_status_counts.items())),
        "aggregate_validation_ref_count": aggregate_validation_ref_count,
        "blocked_delivery_ref_count": blocked_delivery_ref_count,
        "manifest_ref_count": len(refs),
    }


def _write_summary(path: Path, pack: dict[str, Any]) -> None:
    lines = [
        "# Audio Quality Source Lock",
        "",
        f"- current_rag_id: `{pack['current_rag_id']}`",
        f"- code_index_status: `{pack['code_index_status']}`",
        f"- code_index_fallback: `{pack['code_index_fallback']}`",
        f"- docker_service: `{pack['docker_service']}`",
        f"- gpu_status: `{pack['gpu_status']}`",
        f"- git_dirty_count: `{pack['git_dirty_count']}`",
        f"- rag_delivery_status_counts: `{pack['rag_delivery_state']['delivery_status_counts']}`",
        f"- aggregate_validation_ref_count: `{pack['rag_delivery_state']['aggregate_validation_ref_count']}`",
        "- delivery_allowed: `false`",
        "- release_ready: `false`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audio_quality_source_lock(
    *,
    rag_current_json: Path,
    output_dir: Path,
    code_index_status: str,
    code_index_fallback: str,
    docker_service: str = "ml-worker",
    gpu_status: str = "not_checked",
    project_root: Path | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "start", start_time=start_time, rag_current_json=str(rag_current_json))
    rag = load_rag_current(rag_current_json)
    rag_delivery_state = _rag_delivery_state(rag_current_json, rag)
    root = project_root or APP_ROOT.parent
    git_status_available, git_rows = _git_status_short(root)
    pack = {
        "schema": "audio_quality_source_lock.v1",
        "manifest_type": "audio_quality_source_lock",
        "lock_id": "aql_" + stable_hash([rag, git_rows, code_index_status, docker_service, gpu_status])[:12],
        "current_rag_id": rag["rag_id"],
        "rag_current_json": str(rag_current_json),
        "rag_idx": rag["idx"],
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "code_index_status": str(code_index_status),
        "code_index_fallback": str(code_index_fallback),
        "docker_service": str(docker_service),
        "gpu_status": str(gpu_status),
        "git_status_available": git_status_available,
        "git_dirty_count": len(git_rows) if git_status_available else None,
        "git_status_short": git_rows[:200],
        "rag_delivery_state": rag_delivery_state,
        "outputs": {
            "pack": str(output_dir / PACK_NAME),
            "summary": str(output_dir / SUMMARY_NAME),
            "progress": str(progress_path),
        },
        "presentation_status": "audio_quality_source_lock_not_for_listen_pack",
        **SAFETY_FLAGS,
    }
    write_json(output_dir / PACK_NAME, pack)
    _write_summary(output_dir / SUMMARY_NAME, pack)
    append_progress(progress_path, "finish", start_time=start_time, current_rag_id=pack["current_rag_id"], git_dirty_count=len(git_rows))
    return pack


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a source-of-truth lock pack for audio quality critic execution.")
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-index-status", default="unknown")
    parser.add_argument("--code-index-fallback", default="")
    parser.add_argument("--docker-service", default="ml-worker")
    parser.add_argument("--gpu-status", default="not_checked")
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_audio_quality_source_lock.py", entrypoint_path="/app/tools/build_audio_quality_source_lock.py")
    args = parse_args(argv)
    pack = build_audio_quality_source_lock(
        rag_current_json=args.rag_current_json,
        output_dir=args.output_dir,
        code_index_status=str(args.code_index_status),
        code_index_fallback=str(args.code_index_fallback),
        docker_service=str(args.docker_service),
        gpu_status=str(args.gpu_status),
        project_root=args.project_root,
    )
    print(json.dumps({"pack_json": str(args.output_dir / PACK_NAME), "current_rag_id": pack["current_rag_id"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
