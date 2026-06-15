from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_tensor_audio_error_critic(model_dir: Path, *, require_trained: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = model_dir / "tensor_audio_error_critic_manifest.json"
    if not manifest_path.exists():
        errors.append("missing_tensor_audio_error_critic_manifest")
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
    if manifest.get("schema") != "tensor_audio_error_critic.v1":
        errors.append("invalid_manifest_schema")
    if manifest.get("tensor_maps_required") is not True:
        errors.append("tensor_maps_not_required")
    if manifest.get("scalar_fallback_used") is True:
        errors.append("scalar_fallback_used")
    if int(manifest.get("critical_leakage_count") or 0) != 0:
        errors.append("critical_leakage_present")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_true")
    if manifest.get("separator_training_started") is True:
        errors.append("separator_training_started_true")
    if manifest.get("repair_adapter_training_started") is True:
        errors.append("repair_adapter_training_started_true")
    if require_trained and manifest.get("training_completed") is not True:
        errors.append("training_not_completed")
    if manifest.get("training_completed") is True and not (model_dir / "tensor_audio_error_critic.joblib").exists():
        errors.append("missing_trained_model_file")
    result = {
        "schema": "tensor_audio_error_critic_validation.v1",
        "model_dir": str(model_dir),
        "ok": not errors,
        "errors": errors,
        "release_ready": False,
        "separator_training_started": False,
        "repair_adapter_training_started": False,
    }
    write_json(model_dir / "tensor_audio_error_critic_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a shadow tensor audio error critic artifact.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--require-trained", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_critic(args.model_dir, require_trained=args.require_trained)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
