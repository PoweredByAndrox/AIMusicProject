"""Train a torch no-reference stem contamination detector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_stem_contamination_dataset import validate_stem_contamination_row
from tools.docker_runtime import require_docker_runtime
from tools.stem_contamination_common import FORBIDDEN_SOURCES
from tools.stem_contamination_model import (
    FEATURE_NAMES,
    TORCH_AVAILABLE,
    StemContaminationNet,
    extract_no_reference_features,
)

if TORCH_AVAILABLE:  # pragma: no cover - exercised in Docker tests.
    import torch
    from torch import nn
else:  # pragma: no cover
    torch = None
    nn = None


MODEL_FILE = "model.pt"
MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.json"
PROGRESS_FILE = "progress.jsonl"


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _stable_id(rows: list[dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps([row.get("row_id") for row in rows], sort_keys=True).encode("utf-8")
    return "scd_" + hashlib.sha256(payload).hexdigest()[:12]


def _training_examples(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[float], int, dict[tuple[str, str], int]]:
    features: list[list[float]] = []
    labels: list[float] = []
    positive_rows = 0
    pair_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        validate_stem_contamination_row(row)
        tested_stem = str(row.get("base_stem") or "")
        foreign_source = str(row.get("foreign_source") or "")
        if foreign_source not in FORBIDDEN_SOURCES.get(tested_stem, ()):
            continue
        if row.get("label_type") != "synthetic_clean" or row.get("is_ground_truth") is not True:
            continue
        dirty, dirty_rate = _read_audio(Path(str(row["input_stem"])))
        leak, leak_rate = _read_audio(Path(str(row["target_leak"])))
        if dirty_rate != leak_rate:
            raise ValueError(f"sample rate mismatch for row {row['row_id']}: {dirty_rate} != {leak_rate}")
        min_len = min(len(dirty), len(leak))
        if min_len <= 0:
            raise ValueError(f"empty audio for row {row['row_id']}")
        dirty = dirty[:min_len]
        leak = leak[:min_len]
        clean = dirty - leak
        features.append(extract_no_reference_features(dirty, dirty_rate, tested_stem=tested_stem, foreign_source=foreign_source))
        labels.append(1.0)
        features.append(extract_no_reference_features(clean, dirty_rate, tested_stem=tested_stem, foreign_source=foreign_source))
        labels.append(0.0)
        positive_rows += 1
        pair = (tested_stem, foreign_source)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return features, labels, positive_rows, pair_counts


def _trained_pairs(pair_counts: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    return [
        {"tested_stem": tested_stem, "foreign_source": foreign_source, "positive_row_count": pair_counts[(tested_stem, foreign_source)]}
        for tested_stem, foreign_source in sorted(pair_counts)
    ]


def train_stem_contamination_detector(
    *,
    rows_jsonl: Path,
    output_dir: Path,
    device: str = "cuda",
    epochs: int = 120,
    hidden_size: int = 8,
    threshold: float = 0.5,
    min_positive_rows: int = 2,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    for name in (MODEL_FILE, MANIFEST_FILE, METRICS_FILE, PROGRESS_FILE):
        path = output_dir / name
        if path.exists():
            path.unlink()
    _append_progress(progress_path, {"event": "start", "rows_jsonl": str(rows_jsonl), "device": device})
    rows = _iter_jsonl(rows_jsonl)
    features, labels, positive_rows, pair_counts = _training_examples(rows)
    trained_pairs = _trained_pairs(pair_counts)
    reason = ""
    if not TORCH_AVAILABLE:
        reason = "torch_unavailable"
    elif positive_rows < min_positive_rows:
        reason = "positive_rows_below_minimum"
    elif len({int(label) for label in labels}) < 2:
        reason = "missing_positive_or_negative_class"
    if reason:
        manifest = {
            "schema": "stem_contamination_detector_model.v1",
            "manifest_type": "stem_contamination_detector_model",
            "model_id": _stable_id(rows),
            "model_file": "",
            "training_status": "skipped",
            "critic_not_usable_reason": reason,
            "detector_mode": "no_reference_neural_v1",
            "tested_stem": trained_pairs[0]["tested_stem"] if len(trained_pairs) == 1 else "",
            "foreign_source": trained_pairs[0]["foreign_source"] if len(trained_pairs) == 1 else "",
            "trained_pairs": trained_pairs,
            "pair_count": len(trained_pairs),
            "shadow_only": True,
            "promotion_allowed": False,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
        }
        _write_json(output_dir / MANIFEST_FILE, manifest)
        _write_json(output_dir / METRICS_FILE, {"training_status": "skipped", "reason": reason})
        _append_progress(progress_path, {"event": "skipped", "reason": reason})
        return manifest
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for stem contamination detector training but torch.cuda.is_available() is false")

    torch.manual_seed(7)
    torch_device = torch.device(device)
    x_raw = np.asarray(features, dtype=np.float32)
    feature_mean = x_raw.mean(axis=0)
    feature_std = np.maximum(x_raw.std(axis=0), 1.0e-6)
    x_train = torch.tensor((x_raw - feature_mean) / feature_std, dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(np.asarray(labels, dtype=np.float32)[:, None], dtype=torch.float32, device=torch_device)
    model = StemContaminationNet(input_width=len(FEATURE_NAMES), hidden_size=int(hidden_size)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-2)
    loss_fn = nn.BCELoss()
    final_loss = 0.0
    for epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if epoch == 0 or epoch + 1 == int(epochs):
            _append_progress(progress_path, {"event": "epoch", "epoch": epoch + 1, "loss": round(final_loss, 8)})

    model_id = _stable_id(rows)
    bundle = {
        "model_id": model_id,
        "input_width": len(FEATURE_NAMES),
        "hidden_size": int(hidden_size),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [float(value) for value in feature_mean.tolist()],
        "feature_std": [float(value) for value in feature_std.tolist()],
        "state_dict": model.to("cpu").state_dict(),
    }
    model_path = output_dir / MODEL_FILE
    torch.save(bundle, model_path)
    manifest = {
        "schema": "stem_contamination_detector_model.v1",
        "manifest_type": "stem_contamination_detector_model",
        "model_id": model_id,
        "model_family": "torch_no_reference_stem_contamination",
        "model_file": str(model_path),
        "rows_jsonl": str(rows_jsonl),
        "training_status": "trained",
        "training_completed": True,
        "critic_not_usable_reason": "",
        "detector_mode": "no_reference_neural_v1",
        "tested_stem": trained_pairs[0]["tested_stem"] if len(trained_pairs) == 1 else "",
        "foreign_source": trained_pairs[0]["foreign_source"] if len(trained_pairs) == 1 else "",
        "trained_pairs": trained_pairs,
        "pair_count": len(trained_pairs),
        "threshold": float(threshold),
        "positive_row_count": positive_rows,
        "train_example_count": len(labels),
        "device": device,
        "epochs": int(epochs),
        "hidden_size": int(hidden_size),
        "final_loss": round(final_loss, 8),
        "shadow_only": True,
        "promotion_allowed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "sklearn_used": False,
        "proxy_only_model": False,
    }
    _write_json(output_dir / MANIFEST_FILE, manifest)
    _write_json(
        output_dir / METRICS_FILE,
        {
            "training_status": "trained",
            "positive_row_count": positive_rows,
            "pair_count": len(trained_pairs),
            "trained_pairs": trained_pairs,
            "train_example_count": len(labels),
            "final_loss": round(final_loss, 8),
        },
    )
    _append_progress(progress_path, {"event": "finish", "elapsed_sec": round(time.monotonic() - started, 6), "model_id": model_id})
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-positive-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="train_stem_contamination_detector.py", entrypoint_path="/app/tools/train_stem_contamination_detector.py")
    args = parse_args(argv)
    if str(args.device) != "cuda" and not bool(args.allow_non_gpu):
        raise ValueError("stem contamination detector training requires --device cuda unless --allow-non-gpu is set")
    manifest = train_stem_contamination_detector(
        rows_jsonl=args.rows_jsonl,
        output_dir=args.out,
        device=str(args.device),
        epochs=int(args.epochs),
        hidden_size=int(args.hidden_size),
        threshold=float(args.threshold),
        min_positive_rows=int(args.min_positive_rows),
    )
    print(json.dumps({"manifest_json": str(args.out / MANIFEST_FILE), "training_status": manifest["training_status"]}, sort_keys=True))
    return 0 if manifest["training_status"] == "trained" else 2


if __name__ == "__main__":
    raise SystemExit(main())
