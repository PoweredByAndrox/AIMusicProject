from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from adaptive_policy import decision_from_evidence  # noqa: E402

MAX_JSON_BYTES = 20 * 1024 * 1024
EXACT_RESULT_JSON_NAMES = {
    "adaptive_vocal_manifest.json",
    "adaptive_manifest.json",
    "stem_manifest.json",
}
RESULT_JSON_SUFFIXES = (
    "_manifest.json",
    "_report.json",
)


def read_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        return {
            "_large_json_skipped": True,
            "status": "read_skipped_large_json",
            "bytes": size,
            "max_json_bytes": MAX_JSON_BYTES,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.json")


def is_result_json(path: Path) -> bool:
    name = path.name
    return name in EXACT_RESULT_JSON_NAMES or any(name.endswith(suffix) for suffix in RESULT_JSON_SUFFIXES)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def model_zoo_exhaustion_labels(payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    next_action = payload.get("next_action")
    overall = payload.get("overall")
    contract = payload.get("contract")
    bounded_rerun = contract.get("bounded_auto_rerun") if isinstance(contract, dict) else None

    for node in (next_action, overall):
        if isinstance(node, dict) and node.get("status") == "no_accepted_candidate":
            labels.append("no_accepted_candidate")

    if isinstance(bounded_rerun, dict):
        exhaustion_action = bounded_rerun.get("exhaustion_action")
        if isinstance(exhaustion_action, str) and "candidate_space_exhausted" in exhaustion_action:
            labels.append("candidate_space_exhausted")

    if payload.get("selected_candidate") is None and labels:
        labels.append("selected_candidate_null")
    return labels


def embedded_decision_contract(payload: dict[str, Any]) -> dict[str, Any] | None:
    contract = payload.get("decision_contract")
    if not isinstance(contract, dict):
        return None
    if contract.get("mode") != "adaptive_decision_contract":
        return None
    required_keys = ("dominant_failure_class", "decision", "required_next_action")
    if not all(isinstance(contract.get(key), str) and contract.get(key) for key in required_keys):
        return None
    return contract


def collect_issue_labels(payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []

    def add(value: Any) -> None:
        for item in as_list(value):
            if isinstance(item, str) and item.strip():
                labels.append(item.strip())

    add(payload.get("failed_check_names"))
    add(payload.get("issue_labels"))
    add(payload.get("blocking_issue_labels"))
    add(model_zoo_exhaustion_labels(payload))

    for key in (
        "detected_issue_action_contract",
        "selected_minus_release_gate",
        "forensic_delivery_contract",
        "artifact_contract",
        "quality_gate",
        "verdict",
    ):
        node = payload.get(key)
        if isinstance(node, dict):
            add(node.get("failed_check_names"))
            add(node.get("issue_labels"))
            add(node.get("blocking_issue_labels"))
            add(node.get("reason"))
            add(node.get("status"))

    for candidate in as_list(payload.get("candidates")):
        if not isinstance(candidate, dict):
            continue
        add(candidate.get("failed_check_names"))
        add(candidate.get("issue_labels"))
        add(candidate.get("reasons"))
        manifest = candidate.get("manifest")
        if isinstance(manifest, dict):
            add(manifest.get("failed_check_names"))
            add(manifest.get("issue_labels"))

    for stem in as_list(payload.get("stems")):
        if not isinstance(stem, dict):
            continue
        verdict = stem.get("verdict")
        if isinstance(verdict, dict):
            add(verdict.get("failed_check_names"))
            add(verdict.get("issue_labels"))
        for windows_key in ("quality_windows", "micro_quality_windows"):
            windows = stem.get(windows_key)
            if isinstance(windows, dict):
                for window in as_list(windows.get("worst_windows")):
                    if isinstance(window, dict):
                        add(window.get("issue_labels"))
    return labels


def collect_failed_checks(payload: dict[str, Any]) -> list[str]:
    checks: list[str] = []

    def add(value: Any) -> None:
        for item in as_list(value):
            if isinstance(item, str) and item.strip():
                checks.append(item.strip())

    add(payload.get("failed_check_names"))
    add(model_zoo_exhaustion_labels(payload))
    for key in ("selected_minus_release_gate", "forensic_delivery_contract", "artifact_contract", "quality_gate", "verdict"):
        node = payload.get(key)
        if isinstance(node, dict):
            add(node.get("failed_check_names"))
            add(node.get("blocking_issue_labels"))
    for candidate in as_list(payload.get("candidates")):
        if isinstance(candidate, dict):
            add(candidate.get("failed_check_names"))
            add(candidate.get("blocking_issue_labels"))
    return checks


def infer_status(payload: dict[str, Any]) -> str:
    for key in ("next_action", "overall"):
        node = payload.get(key)
        if isinstance(node, dict):
            value = node.get("status")
            if isinstance(value, str) and value:
                return value
    for key in ("delivery_status", "status", "presentation_status"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("selected_minus_release_gate", "forensic_delivery_contract", "artifact_contract", "quality_gate", "verdict"):
        node = payload.get(key)
        if isinstance(node, dict):
            for nested_key in ("delivery_status", "status", "presentation_status"):
                value = node.get(nested_key)
                if isinstance(value, str) and value:
                    return value
    return "unknown"


def summarize_manifest(path: Path, root: Path) -> dict[str, Any]:
    payload = read_json(path)
    issue_labels = collect_issue_labels(payload)
    failed_checks = collect_failed_checks(payload)
    status = infer_status(payload)
    decision = embedded_decision_contract(payload) or decision_from_evidence(
        issue_labels=issue_labels,
        failed_checks=failed_checks,
        candidate_status=status,
    )
    return {
        "manifest_path": str(path.relative_to(root)),
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
        "source_audio_path": payload.get("source_audio_path") or payload.get("raw_audio_path"),
        "selected_profile": payload.get("selected_profile"),
        "selected_score": payload.get("selected_score"),
        "selected_manifest_path": payload.get("selected_manifest_path"),
        "status": status,
        "top_issue_labels": [name for name, _ in Counter(issue_labels).most_common(20)],
        "top_failed_checks": [name for name, _ in Counter(failed_checks).most_common(20)],
        "decision_contract": decision,
    }


def build_report(artifact_root: Path) -> dict[str, Any]:
    manifests = []
    for path in iter_json_files(artifact_root):
        name = path.name
        if not is_result_json(path):
            continue
        try:
            manifests.append(summarize_manifest(path, artifact_root))
        except Exception as exc:  # keep report generation robust for old artifacts
            manifests.append({"manifest_path": str(path.relative_to(artifact_root)), "error": str(exc)})

    class_counts = Counter(
        item.get("decision_contract", {}).get("dominant_failure_class", "read_error") for item in manifests
    )
    decision_counts = Counter(item.get("decision_contract", {}).get("decision", "read_error") for item in manifests)
    return {
        "schema_version": 1,
        "mode": "adaptive_decision_report",
        "artifact_root": str(artifact_root),
        "summary": {
            "manifest_count": len(manifests),
            "dominant_failure_class_counts": dict(class_counts.most_common()),
            "decision_counts": dict(decision_counts.most_common()),
        },
        "manifests": manifests,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Adaptive Decision Report", ""]
    summary = report.get("summary", {})
    lines.append(f"Manifest count: `{summary.get('manifest_count', 0)}`")
    lines.append("")
    lines.append("## Dominant failure classes")
    for key, value in summary.get("dominant_failure_class_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Decisions")
    for key, value in summary.get("decision_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Top blocked/hard-case manifests")
    lines.append("| manifest | class | decision | next action |")
    lines.append("|---|---|---|---|")
    shown = 0
    for item in report.get("manifests", []):
        decision = item.get("decision_contract", {})
        if decision.get("decision") in {"promote", None}:
            continue
        lines.append(
            f"| `{item.get('manifest_path')}` | `{decision.get('dominant_failure_class')}` | "
            f"`{decision.get('decision')}` | `{decision.get('required_next_action')}` |"
        )
        shown += 1
        if shown >= 100:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an adaptive decision report from existing manifests.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.artifact_root)
    write_json(args.output_dir / "adaptive_decision_report.json", report)
    write_markdown(args.output_dir / "adaptive_decision_report.md", report)


if __name__ == "__main__":
    main()
