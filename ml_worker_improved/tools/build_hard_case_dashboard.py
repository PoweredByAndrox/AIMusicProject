from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def collect_strings(payload: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    found: list[str] = []

    def add(value: Any) -> None:
        for item in as_list(value):
            if isinstance(item, str) and item.strip():
                found.append(item.strip())

    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys:
                    add(value)
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def infer_status(payload: dict[str, Any]) -> str:
    for key in ("next_action", "overall"):
        node = payload.get(key)
        if isinstance(node, dict):
            value = node.get("status")
            if isinstance(value, str):
                return value
    direct = payload.get("status") or payload.get("delivery_status") or payload.get("presentation_status")
    if isinstance(direct, str):
        return direct
    for key in ("selected_minus_release_gate", "forensic_delivery_contract", "artifact_contract", "quality_gate", "verdict"):
        node = payload.get(key)
        if isinstance(node, dict):
            value = node.get("status") or node.get("delivery_status") or node.get("presentation_status")
            if isinstance(value, str):
                return value
    return "unknown"


def row_for_manifest(path: Path, root: Path) -> dict[str, str]:
    payload = read_json(path)
    issue_labels = collect_strings(payload, ("issue_labels", "blocking_issue_labels", "reasons"))
    failed_checks = collect_strings(payload, ("failed_check_names", "blocking_issue_labels"))
    exhaustion_labels = model_zoo_exhaustion_labels(payload)
    issue_labels.extend(exhaustion_labels)
    failed_checks.extend(exhaustion_labels)
    status = infer_status(payload)
    decision = embedded_decision_contract(payload) or decision_from_evidence(
        issue_labels=issue_labels,
        failed_checks=failed_checks,
        candidate_status=status,
    )
    source_audio = payload.get("source_audio_path") or payload.get("raw_audio_path") or ""
    top_issues = [name for name, _ in Counter(issue_labels).most_common(8)]
    top_checks = [name for name, _ in Counter(failed_checks).most_common(8)]
    return {
        "manifest_path": str(path.relative_to(root)),
        "source_audio_path": str(source_audio),
        "selected_profile": str(payload.get("selected_profile") or ""),
        "status": status,
        "dominant_failure_class": str(decision.get("dominant_failure_class")),
        "decision": str(decision.get("decision")),
        "required_next_action": str(decision.get("required_next_action")),
        "allowed_actions": ";".join(decision.get("allowed_actions", [])),
        "forbidden_actions": ";".join(decision.get("forbidden_actions", [])),
        "top_issue_labels": ";".join(top_issues),
        "top_failed_checks": ";".join(top_checks),
    }


def build_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*.json"):
        if not is_result_json(path):
            continue
        try:
            row = row_for_manifest(path, root)
        except Exception as exc:
            row = {
                "manifest_path": str(path.relative_to(root)),
                "source_audio_path": "",
                "selected_profile": "",
                "status": "read_error",
                "dominant_failure_class": "read_error",
                "decision": "reject",
                "required_next_action": f"fix_manifest_read_error: {exc}",
                "allowed_actions": "inspect_manifest_schema",
                "forbidden_actions": "generate_more_ab_candidates;switch_backbone",
                "top_issue_labels": "",
                "top_failed_checks": "",
            }
        rows.append(row)
    rows.sort(key=lambda item: (item["decision"] == "promote", item["dominant_failure_class"], item["manifest_path"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest_path",
        "source_audio_path",
        "selected_profile",
        "status",
        "dominant_failure_class",
        "decision",
        "required_next_action",
        "allowed_actions",
        "forbidden_actions",
        "top_issue_labels",
        "top_failed_checks",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["dominant_failure_class"] for row in rows)
    lines = ["# Hard Case Dashboard", "", "## Dominant failure class counts"]
    for key, value in counts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Non-promoted items", "| manifest | class | decision | next action |", "|---|---|---|---|"])
    shown = 0
    for row in rows:
        if row["decision"] == "promote":
            continue
        lines.append(
            f"| `{row['manifest_path']}` | `{row['dominant_failure_class']}` | "
            f"`{row['decision']}` | `{row['required_next_action']}` |"
        )
        shown += 1
        if shown >= 200:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-case dashboard from existing manifests.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.artifact_root)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)


if __name__ == "__main__":
    main()
