from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFETY_FLAGS = {
    "training_started": False,
    "critic_training_started": False,
    "separator_training_started": False,
    "repair_adapter_training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "can_publish_as_result": False,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected_json_object:{path}:{line_number}")
        yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def threshold_metrics(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row.get("label") or 0) for row in rows]
    scores = [float(row.get("predicted_accept_probability") or 0.0) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    accepted = [score >= threshold for score in scores]
    false_accept = sum(1 for label, accept in zip(labels, accepted) if label == 0 and accept)
    false_reject = sum(1 for label, accept in zip(labels, accepted) if label == 1 and not accept)
    true_accept = sum(1 for label, accept in zip(labels, accepted) if label == 1 and accept)
    return {
        "threshold": float(threshold),
        "rows": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "accepted_rows": sum(accepted),
        "true_accept": true_accept,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": float(false_accept / negatives) if negatives else None,
        "false_reject_rate": float(false_reject / positives) if positives else None,
        "accept_coverage": float(true_accept / positives) if positives else 0.0,
    }


def choose_threshold(
    rows: Sequence[dict[str, Any]],
    *,
    max_false_accept_rate: float,
) -> dict[str, Any]:
    if not rows:
        return {"status": "not_usable", "reason": "no_rows", "threshold": 1.000001, "accept_coverage": 0.0}
    candidates = sorted({0.0, 0.5, 1.000001, *(float(row.get("predicted_accept_probability") or 0.0) for row in rows)})
    candidates.extend(min(1.000001, value + 1.0e-6) for value in list(candidates))
    best: dict[str, Any] | None = None
    for threshold in sorted(set(candidates)):
        metrics = threshold_metrics(rows, threshold)
        false_accept_rate = metrics.get("false_accept_rate")
        if false_accept_rate is not None and float(false_accept_rate) > float(max_false_accept_rate):
            continue
        if best is None or (metrics["accept_coverage"], -metrics["threshold"]) > (best["accept_coverage"], -best["threshold"]):
            best = metrics
    if best is None:
        best = threshold_metrics(rows, 1.000001)
    best["status"] = "usable" if float(best.get("accept_coverage") or 0.0) > 0.0 else "not_usable"
    return best


def calibrate_tensor_audio_error_critic(
    *,
    model_dir: Path,
    output_dir: Path,
    max_false_accept_rate: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(model_dir / "tensor_audio_error_critic_manifest.json")
    if manifest.get("release_ready") is True:
        raise ValueError("release_ready_model_not_allowed")
    if manifest.get("scalar_fallback_used") is True:
        raise ValueError("scalar_fallback_model_not_allowed")
    rows = list(iter_jsonl(model_dir / "tensor_audio_error_critic_eval.jsonl"))
    global_metrics = choose_threshold(rows, max_false_accept_rate=max_false_accept_rate)
    by_stem: dict[str, Any] = {}
    for stem in sorted({str(row.get("stem_type") or "unknown") for row in rows}):
        by_stem[stem] = choose_threshold(
            [row for row in rows if str(row.get("stem_type") or "unknown") == stem],
            max_false_accept_rate=max_false_accept_rate,
        )
    not_usable_reasons: list[str] = []
    if global_metrics.get("status") != "usable":
        not_usable_reasons.append("no_positive_accept_coverage_at_false_accept_budget")
    result = {
        "schema": "tensor_audio_error_critic_calibration.v1",
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "max_false_accept_rate": float(max_false_accept_rate),
        "calibration_status": "calibrated" if not not_usable_reasons else "not_usable",
        "not_usable_reasons": not_usable_reasons,
        "global": global_metrics,
        "by_stem": by_stem,
        "shadow_only": True,
        "scalar_fallback_used": False,
        "tensor_maps_required": True,
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "tensor_audio_error_thresholds.json", result)
    write_jsonl(
        output_dir / "tensor_audio_error_threshold_sweep.jsonl",
        [
            {"scope": "global", **global_metrics, "release_ready": False},
            *({"scope": f"stem:{stem}", **payload, "release_ready": False} for stem, payload in by_stem.items()),
        ],
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate shadow tensor critic thresholds without changing runtime gates.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibrate_tensor_audio_error_critic(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_false_accept_rate=args.max_false_accept_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
