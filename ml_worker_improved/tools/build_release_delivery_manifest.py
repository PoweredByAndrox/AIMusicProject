from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_deliverable(track: dict[str, Any]) -> bool:
    return (
        track.get("delivery_status") == "deliverable"
        and track.get("release_status") == "release_ready"
        and isinstance(track.get("deliverable_vocals_flac"), str)
    )


def _diagnostic_vocals_path(track: dict[str, Any]) -> Any:
    return track.get("diagnostic_vocals_flac") or track.get("selected_vocals_flac")


def _diagnostic_listen_path(track: dict[str, Any]) -> Any:
    return track.get("diagnostic_listen_vocals_flac") or track.get("listen_selected_vocals_flac")


def _blocked_reasons(track: dict[str, Any], delivery_status: str) -> list[str]:
    reasons = [str(reason) for reason in _as_list(track.get("blocker_reasons"))]
    if delivery_status == "diagnostic_only" and not reasons:
        if track.get("gate_status") == "missing_release_gate" or track.get("delivery_status") is None:
            reasons.append("missing_delivery_gate")
        elif track.get("release_status_drift") is True:
            reasons.append("release_status_drift")
        else:
            reasons.append("not_release_ready")
    if delivery_status == "route_failed":
        reasons.append("route_failed")
    return sorted(set(reasons))


def _exit_code(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 1
    return 0


def build_delivery_track(track: dict[str, Any]) -> dict[str, Any]:
    exit_code = _exit_code(track.get("exit_code", 0))
    route_failed = exit_code != 0
    if route_failed:
        delivery_status = "route_failed"
        release_status = "route_failed"
    elif _is_deliverable(track):
        delivery_status = "deliverable"
        release_status = "release_ready"
    else:
        delivery_status = "diagnostic_only"
        release_status = "diagnostic_only"

    return {
        "source_path": track.get("source_path"),
        "manifest_path": track.get("manifest_path"),
        "output_dir": track.get("output_dir"),
        "listen_dir": track.get("listen_dir"),
        "selected_profile": track.get("selected_profile"),
        "selected_score": track.get("selected_score"),
        "exit_code": exit_code,
        "contract_status": track.get("contract_status"),
        "gate_status": track.get("gate_status"),
        "release_status": release_status,
        "artifact_release_status": track.get("artifact_release_status"),
        "release_status_drift": bool(track.get("release_status_drift")),
        "delivery_status": delivery_status,
        "deliverable": {
            "vocals_flac": track.get("deliverable_vocals_flac") if delivery_status == "deliverable" else None,
            "listen_vocals_flac": (
                track.get("deliverable_listen_vocals_flac") if delivery_status == "deliverable" else None
            ),
        },
        "diagnostic": {
            "vocals_flac": _diagnostic_vocals_path(track) if delivery_status != "deliverable" else None,
            "listen_vocals_flac": _diagnostic_listen_path(track) if delivery_status != "deliverable" else None,
            "issue_clips_manifest": track.get("listen_segment_issue_clips_manifest"),
        },
        "blockers": {
            "reasons": _blocked_reasons(track, delivery_status),
            "labels": [str(label) for label in _as_list(track.get("blocker_labels"))],
            "blocked_window_count": int(track.get("blocked_window_count", 0) or 0),
            "worst_windows": _as_list(track.get("worst_windows")),
            "uncovered_labels": [str(label) for label in _as_list(track.get("uncovered_labels"))],
        },
    }


def build_release_delivery_manifest(
    *,
    batch_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    tracks = [
        build_delivery_track(track)
        for track in _as_list(batch_summary.get("tracks"))
        if isinstance(track, dict)
    ]
    delivery_counts = Counter(str(track.get("delivery_status", "unknown")) for track in tracks)
    release_counts = Counter(str(track.get("release_status", "unknown")) for track in tracks)
    manifest = {
        "schema_version": 1,
        "mode": "release_delivery_manifest",
        "source_summary_path": str(batch_summary_path),
        "source_summary_mode": batch_summary.get("mode"),
        "track_count": len(tracks),
        "deliverable_count": int(delivery_counts.get("deliverable", 0)),
        "diagnostic_count": int(delivery_counts.get("diagnostic_only", 0)),
        "failed_count": int(delivery_counts.get("route_failed", 0)),
        "delivery_status_counts": dict(sorted(delivery_counts.items())),
        "release_status_counts": dict(sorted(release_counts.items())),
        "tracks": tracks,
    }
    _write_json(output_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact release delivery manifest from an adaptive vocal batch summary.",
    )
    parser.add_argument("batch_summary_path", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_release_delivery_manifest.py",
        entrypoint_path="/app/tools/build_release_delivery_manifest.py",
    )
    args = parse_args(argv)
    manifest = build_release_delivery_manifest(
        batch_summary_path=args.batch_summary_path,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "track_count": manifest["track_count"],
                "deliverable_count": manifest["deliverable_count"],
                "diagnostic_count": manifest["diagnostic_count"],
                "failed_count": manifest["failed_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
