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


def validate_tensor_audio_error_critic(
    model_dir: Path,
    *,
    calibration_dir: Path | None = None,
    require_trained: bool = False,
    require_usable_calibration: bool = False,
) -> dict[str, Any]:
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
    calibration_status = ""
    if calibration_dir is None and require_usable_calibration:
        errors.append("missing_required_calibration_dir")
    if calibration_dir is not None:
        calibration_path = calibration_dir / "tensor_audio_error_thresholds.json"
        if not calibration_path.exists():
            errors.append("missing_tensor_threshold_calibration")
        else:
            calibration = read_json(calibration_path)
            calibration_status = str(calibration.get("calibration_status") or "")
            if calibration.get("schema") != "tensor_audio_error_critic_calibration.v1":
                errors.append("invalid_calibration_schema")
            if calibration.get("tensor_maps_required") is not True:
                errors.append("calibration_tensor_maps_not_required")
            if calibration.get("shadow_only") is not True:
                errors.append("calibration_not_shadow_only")
            if str(calibration.get("model_dir") or "") != str(model_dir):
                errors.append("calibration_model_dir_mismatch")
            if calibration.get("release_ready") is True:
                errors.append("calibration_release_ready_true")
            if calibration.get("scalar_fallback_used") is True:
                errors.append("calibration_scalar_fallback_used")
            if require_usable_calibration and calibration_status != "calibrated":
                errors.append("calibration_not_usable")
    result = {
        "schema": "tensor_audio_error_critic_validation.v1",
        "model_dir": str(model_dir),
        "ok": not errors,
        "errors": errors,
        "calibration_status": calibration_status,
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
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--require-trained", action="store_true")
    parser.add_argument("--require-usable-calibration", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tensor_audio_error_critic(
        args.model_dir,
        calibration_dir=args.calibration_dir,
        require_trained=args.require_trained,
        require_usable_calibration=args.require_usable_calibration,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
