#!/usr/bin/env python3
"""Run the short listen review packet sequence from one Docker command."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_short_listen_goal_status import audit_short_listen_goal_status  # noqa: E402
from tools.build_short_listen_feedback_template import (  # noqa: E402
    MANIFEST_NAME as TEMPLATE_MANIFEST_NAME,
    TEMPLATE_NAME,
    build_short_listen_feedback_template,
)
from tools.build_short_listen_review_handoff import (  # noqa: E402
    CHECKLIST_NAME,
    MANIFEST_NAME as HANDOFF_MANIFEST_NAME,
    PLAYLIST_NAME,
    build_short_listen_review_handoff,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.full_pack_stem_contract import (  # noqa: E402
    missing_required_stems,
    normalize_stem_list,
    pack_items_for_stem_contract,
    present_stems_from_items,
    required_stem_missing_checks,
    required_stems_from_cli,
)
from tools.validate_short_listen_review_handoff import (  # noqa: E402
    VALIDATION_NAME as HANDOFF_VALIDATION_NAME,
    validate_short_listen_review_handoff,
)

START_TIME = time.monotonic()
MANIFEST_NAME = "packet.json"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_review_packet_orchestration",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "short_listen_review_packet_no_inference",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "short_listen_review_packet_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def stage_failed(stage_name: str, payload: dict[str, Any]) -> list[str]:
    failed = [str(item) for item in payload.get("failed_checks") or [] if str(item)]
    return [f"{stage_name}:{item}" for item in failed]


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# short listen review packet",
        "",
        f"- packet_status: `{manifest['packet_status']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Stages",
        "",
    ]
    for name, payload in manifest["stages"].items():
        status = (
            payload.get("template_status")
            or payload.get("handoff_status")
            or payload.get("validation_status")
            or payload.get("goal_status")
            or "unknown"
        )
        lines.append(f"- `{name}`: `{status}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def run_short_listen_review_packet(
    *,
    pack_manifest_json: Path,
    output_dir: Path,
    aggregate_validation_json: Path | None = None,
    required_stems: Sequence[str] | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", pack_manifest_json=str(pack_manifest_json))

    stages: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    pack_manifest = read_json(pack_manifest_json)
    requested_required_stems = required_stems_from_cli(requested=required_stems, require_full_pack=False)
    manifest_required_stems = normalize_stem_list(pack_manifest.get("required_stems") if isinstance(pack_manifest.get("required_stems"), list) else [])
    contract_required_stems = normalize_stem_list([*manifest_required_stems, *requested_required_stems])
    contract_items = pack_items_for_stem_contract(pack_manifest)
    present_stems = present_stems_from_items(contract_items, contract_required_stems)
    missing_stems = missing_required_stems(contract_required_stems, present_stems)
    failed.extend(required_stem_missing_checks(missing_stems, prefix="full_pack_required_stem_missing"))

    if not failed:
        template_dir = output_dir / "feedback_template"
        template = build_short_listen_feedback_template(
            pack_manifest_json=pack_manifest_json,
            output_dir=template_dir,
            command_args=["--pack-manifest-json", str(pack_manifest_json), "--output-dir", str(template_dir)],
        )
        stages["feedback_template"] = {
            "template_status": template.get("template_status"),
            "item_count": template.get("item_count"),
            "failed_checks": template.get("failed_checks") or [],
            "manifest_json": str(template_dir / TEMPLATE_MANIFEST_NAME),
            "template_jsonl": str(template_dir / TEMPLATE_NAME),
        }
        failed.extend(stage_failed("feedback_template", template))

    if not failed:
        handoff_dir = output_dir / "review_handoff"
        handoff = build_short_listen_review_handoff(
            pack_manifest_json=pack_manifest_json,
            feedback_template_manifest_json=template_dir / TEMPLATE_MANIFEST_NAME,
            output_dir=handoff_dir,
            command_args=[
                "--pack-manifest-json",
                str(pack_manifest_json),
                "--feedback-template-manifest-json",
                str(template_dir / TEMPLATE_MANIFEST_NAME),
                "--output-dir",
                str(handoff_dir),
            ],
        )
        stages["review_handoff"] = {
            "handoff_status": handoff.get("handoff_status"),
            "item_count": handoff.get("item_count"),
            "failed_checks": handoff.get("failed_checks") or [],
            "manifest_json": str(handoff_dir / HANDOFF_MANIFEST_NAME),
            "playlist_m3u": str(handoff_dir / PLAYLIST_NAME),
            "listen_checklist_md": str(handoff_dir / CHECKLIST_NAME),
        }
        failed.extend(stage_failed("review_handoff", handoff))

    if not failed:
        validation_dir = output_dir / "review_handoff_validation"
        validation = validate_short_listen_review_handoff(
            handoff_manifest_json=output_dir / "review_handoff" / HANDOFF_MANIFEST_NAME,
            output_dir=validation_dir,
            command_args=[
                "--handoff-manifest-json",
                str(output_dir / "review_handoff" / HANDOFF_MANIFEST_NAME),
                "--output-dir",
                str(validation_dir),
            ],
        )
        stages["review_handoff_validation"] = {
            "validation_status": validation.get("validation_status"),
            "item_count": validation.get("item_count"),
            "failed_checks": validation.get("failed_checks") or [],
            "validation_json": str(validation_dir / HANDOFF_VALIDATION_NAME),
        }
        failed.extend(stage_failed("review_handoff_validation", validation))

    if not failed:
        audit_validation_json = aggregate_validation_json or pack_manifest_json.parent / "validation" / "validation.json"
        audit_dir = output_dir / "goal_audit"
        if pack_manifest.get("schema") == "short_listen_subset_pack_manifest.v1":
            stages["goal_audit"] = {
                "goal_status": "not_run",
                "goal_complete": False,
                "remaining_blockers": ["aggregate_goal_audit_not_applicable_to_subset_pack"],
                "failed_checks": [],
                "audit_json": "",
            }
        elif audit_validation_json.exists():
            audit = audit_short_listen_goal_status(
                pack_manifest_json=pack_manifest_json,
                validation_json=audit_validation_json,
                output_dir=audit_dir,
            )
            stages["goal_audit"] = {
                "goal_status": audit.get("goal_status"),
                "goal_complete": audit.get("goal_complete"),
                "remaining_blockers": audit.get("remaining_blockers") or [],
                "failed_checks": [],
                "audit_json": str(audit_dir / "goal_status_audit.json"),
            }
        else:
            failed.append("goal_audit:aggregate_validation_json_missing")
            stages["goal_audit"] = {
                "goal_status": "not_run",
                "goal_complete": False,
                "remaining_blockers": ["aggregate_validation_json_missing"],
                "failed_checks": ["aggregate_validation_json_missing"],
            }

    goal_status = str(stages.get("goal_audit", {}).get("goal_status") or "")
    if contract_required_stems and goal_status in {"failed_safety_contract", "machine_evidence_incomplete"}:
        failed.append(f"goal_audit:{goal_status}")

    failed = stable_unique(failed)
    packet_status = "ready_for_listen_review" if not failed else "failed_safety_contract"
    manifest = {
        "schema": "short_listen_review_packet_manifest.v1",
        "packet_status": packet_status,
        "pack_manifest_json": str(pack_manifest_json),
        "output_dir": str(output_dir),
        "stages": stages,
        "failed_checks": failed,
        "required_stems": contract_required_stems,
        "present_stems": present_stems,
        "missing_required_stems": missing_stems,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "review_packet_not_for_listen_pack",
        "progress_jsonl": str(progress_path),
        "outputs": {
            "review_packet_manifest_json": str(output_dir / MANIFEST_NAME),
            "review_packet_summary_md": str(output_dir / SUMMARY_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "not_claimed": ["release_ready", "publishable_result", "human_listen_verdict", "semantic_ground_truth"],
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(progress_path, "finish", packet_status=packet_status, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-command short listen review packet sequence.")
    parser.add_argument("--pack-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-validation-json", type=Path)
    parser.add_argument("--required-stem", action="append", default=[])
    parser.add_argument("--require-full-pack-stems", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_short_listen_review_packet.py",
        entrypoint_path="/app/tools/run_short_listen_review_packet.py",
    )
    args = parse_args(argv)
    manifest = run_short_listen_review_packet(
        pack_manifest_json=args.pack_manifest_json,
        output_dir=args.output_dir,
        aggregate_validation_json=args.aggregate_validation_json,
        required_stems=required_stems_from_cli(
            requested=tuple(str(item) for item in args.required_stem),
            require_full_pack=bool(args.require_full_pack_stems),
        ),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "packet_status": manifest["packet_status"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
