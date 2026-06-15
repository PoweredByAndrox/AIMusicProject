from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FILES = (
    "leak_safe_split_manifest.json",
    "audio_fingerprint_index.jsonl",
    "leakage_report.json",
    "hardcase_holdout_manifest.json",
    "locked_eval_summary.md",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_leak_safe_audio_eval_harness(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    for filename in REQUIRED_FILES:
        if not (output_dir / filename).exists():
            errors.append(f"missing_required_file:{filename}")

    manifest_path = output_dir / "leak_safe_split_manifest.json"
    leakage_path = output_dir / "leakage_report.json"
    holdout_path = output_dir / "hardcase_holdout_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    leakage = read_json(leakage_path) if leakage_path.exists() else {}
    holdout = read_json(holdout_path) if holdout_path.exists() else {}

    if manifest.get("schema") != "leak_safe_audio_eval_harness.v1":
        errors.append("invalid_manifest_schema")
    if int(manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("critical_leakage_present")
    if int(leakage.get("critical_leakage_count") or 0) != 0:
        errors.append("leakage_report_critical_leakage_present")
    if manifest.get("release_ready") is True or leakage.get("release_ready") is True or holdout.get("release_ready") is True:
        errors.append("release_ready_true")
    if manifest.get("training_started") is True:
        errors.append("training_started_true")
    if manifest.get("critic_training_started") is True:
        errors.append("critic_training_started_true")
    if manifest.get("separator_training_started") is True:
        errors.append("separator_training_started_true")
    if manifest.get("repair_adapter_training_started") is True:
        errors.append("repair_adapter_training_started_true")
    if holdout and holdout.get("locked") is not True:
        errors.append("hardcase_holdout_not_locked")
    if (manifest.get("model_2c_locked_re_eval") or {}).get("release_ready") is True:
        errors.append("model_2c_re_eval_release_ready_true")
    model_eval = manifest.get("model_2c_locked_re_eval") or {}
    if manifest.get("model_2c_re_eval_required") is True and model_eval.get("status") != "evaluated":
        errors.append("model_2c_re_eval_not_evaluated")
    for target, payload in (model_eval.get("targets") or {}).items():
        if isinstance(payload, dict) and str(payload.get("status") or "").startswith("failed"):
            errors.append(f"model_2c_re_eval_failed:{target}")

    for name, check in (leakage.get("checks") or {}).items():
        if isinstance(check, dict) and check.get("critical") is True:
            errors.append(f"critical_check_failed:{name}")

    result = {
        "schema": "leak_safe_audio_eval_validation.v1",
        "output_dir": str(output_dir),
        "ok": not errors,
        "errors": errors,
        "critical_leakage_count": int(manifest.get("critical_leakage_count") or leakage.get("critical_leakage_count") or 0),
        "release_ready": False,
        "training_started": False,
        "critic_training_started": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(output_dir / "leak_safe_eval_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a leak-safe audio fingerprint evaluation harness.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_leak_safe_audio_eval_harness(args.output_dir)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
