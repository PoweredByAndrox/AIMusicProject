from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    DATASET_FILE,
    MANIFEST_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODEL_MANIFEST_SCHEMA,
    PROGRESS_FILE,
    PROMOTION_LABEL_TYPES,
    SAFETY_FLAGS,
    TARGET_NAMES,
    append_progress,
    feature_vector_from_row,
    iter_jsonl,
    label_type_counts,
    pad_feature_vectors,
    read_json,
    split_counts,
    stable_hash,
    write_json,
)
from tools.docker_runtime import require_docker_runtime

try:  # pragma: no cover - exercised in Docker tests.
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class AudioQualityCriticNet(nn.Module):
        def __init__(self, input_width: int, hidden_size: int, output_width: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_width, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_width),
            )

        def forward(self, features: Any) -> Any:
            return torch.sigmoid(self.net(features))

else:

    class AudioQualityCriticNet:  # pragma: no cover - only used when torch import fails.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("torch_unavailable")


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(dataset_dir / MANIFEST_FILE)
    rows = list(iter_jsonl(dataset_dir / DATASET_FILE))
    return manifest, rows


def _target_vector(row: dict[str, Any]) -> list[float]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    return [float(targets.get(name, 0.0) or 0.0) for name in TARGET_NAMES]


def _write_skipped_manifest(
    *,
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    reason: str,
    rows: list[dict[str, Any]],
    device: str,
    epochs: int,
    hidden_size: int,
) -> dict[str, Any]:
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_critic_model",
        "model_id": f"aqc_skip_{stable_hash([dataset_manifest.get('dataset_id'), reason])[:12]}",
        "model_family": "torch_audio_quality_critic",
        "model_file": "",
        "dataset_manifest_json": str(Path(str(dataset_manifest.get("outputs", {}).get("manifest_json", "")))),
        "source_lock_pack_json": str(dataset_manifest.get("source_lock_pack_json") or ""),
        "training_status": "skipped",
        "critic_not_usable_reason": reason,
        "training_completed": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "sklearn_used": False,
        "proxy_only_model": False,
        "input_tensor_required": True,
        "train_row_count": len(rows),
        "verified_training_row_count": sum(1 for row in rows if row.get("label_type") in PROMOTION_LABEL_TYPES),
        "label_type_counts": label_type_counts(rows),
        "split_counts": split_counts(rows),
        "device": device,
        "epochs": int(epochs),
        "hidden_size": int(hidden_size),
        "false_accept_rate": 1.0,
        "critical_leakage_count": 0,
        "rag_refs_complete": bool(dataset_manifest.get("current_rag_id")),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_json(output_dir / METRICS_FILE, {"training_status": "skipped", "reason": reason, **SAFETY_FLAGS})
    return manifest


def train_audio_quality_critic(
    *,
    dataset_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    epochs: int = 16,
    hidden_size: int = 32,
    min_train_rows: int = 2,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (MANIFEST_FILE, METRICS_FILE, MODEL_FILE, PROGRESS_FILE):
        path = output_dir / name
        if path.exists():
            path.unlink()
    progress_path = output_dir / PROGRESS_FILE
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        dataset_dir=str(dataset_dir),
        expected_acceleration_mode="gpu_training" if device == "cuda" else "cpu_diagnostic_training",
        gpu_training_required=device == "cuda",
    )
    dataset_manifest, rows = _load_dataset(dataset_dir)
    train_rows = [row for row in rows if row.get("safe_for_training") is True and row.get("split") == "train"]
    reason = ""
    if not TORCH_AVAILABLE:
        reason = "torch_unavailable"
    elif not train_rows:
        reason = "missing_train_split_rows"
    elif len(train_rows) < min_train_rows:
        reason = "train_rows_below_minimum"
    elif len({int(row.get("label") or 0) for row in train_rows}) < 2:
        reason = "missing_positive_or_negative_class"
    if reason:
        manifest = _write_skipped_manifest(
            dataset_manifest=dataset_manifest,
            output_dir=output_dir,
            reason=reason,
            rows=train_rows,
            device=device,
            epochs=epochs,
            hidden_size=hidden_size,
        )
        append_progress(progress_path, "skipped", start_time=start_time, reason=reason, train_row_count=len(train_rows))
        return manifest
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for audio quality critic training but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    vectors, input_width = pad_feature_vectors([feature_vector_from_row(row) for row in train_rows])
    x_train = torch.tensor(vectors, dtype=torch.float32, device=torch_device)
    y_train = torch.tensor([_target_vector(row) for row in train_rows], dtype=torch.float32, device=torch_device)
    model = AudioQualityCriticNet(input_width=input_width, hidden_size=int(hidden_size), output_width=len(TARGET_NAMES)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = nn.BCELoss()
    final_loss = 0.0
    for epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        append_progress(progress_path, "epoch", start_time=start_time, epoch=epoch + 1, loss=round(final_loss, 8))
    model_id = "aqc_" + stable_hash(
        {
            "dataset_id": dataset_manifest.get("dataset_id"),
            "row_ids": [row["row_id"] for row in train_rows],
            "input_width": input_width,
            "target_names": TARGET_NAMES,
        }
    )[:12]
    bundle = {
        "model_id": model_id,
        "input_width": input_width,
        "hidden_size": int(hidden_size),
        "target_names": list(TARGET_NAMES),
        "state_dict": model.to("cpu").state_dict(),
        "shadow_only": True,
        "sklearn_used": False,
        "proxy_only_model": False,
    }
    torch.save(bundle, output_dir / MODEL_FILE)
    verified_count = sum(1 for row in train_rows if row.get("label_type") in PROMOTION_LABEL_TYPES)
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_critic_model",
        "model_id": model_id,
        "model_family": "torch_audio_quality_critic",
        "model_file": str(output_dir / MODEL_FILE),
        "dataset_manifest_json": str(dataset_dir / MANIFEST_FILE),
        "source_lock_pack_json": str(dataset_manifest.get("source_lock_pack_json") or ""),
        "training_status": "trained",
        "training_completed": True,
        "critic_not_usable_reason": "",
        "shadow_only": True,
        "promotion_allowed": False,
        "sklearn_used": False,
        "proxy_only_model": False,
        "input_tensor_required": True,
        "target_names": list(TARGET_NAMES),
        "train_row_count": len(train_rows),
        "verified_training_row_count": verified_count,
        "weak_training_row_count": sum(1 for row in train_rows if row.get("label_type") not in PROMOTION_LABEL_TYPES),
        "label_type_counts": label_type_counts(train_rows),
        "split_counts": split_counts(train_rows),
        "device": device,
        "epochs": int(epochs),
        "hidden_size": int(hidden_size),
        "final_loss": round(final_loss, 8),
        "false_accept_rate": 1.0 if verified_count == 0 else None,
        "critical_leakage_count": 0,
        "rag_refs_complete": bool(dataset_manifest.get("current_rag_id")),
        "presentation_status": "audio_quality_critic_shadow_model_not_for_listen_pack",
        "safety": {
            "weak_labels_cannot_promote": True,
            "shadow_until_locked_holdout": True,
            "torch_model_not_sklearn_proxy": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    write_json(
        output_dir / METRICS_FILE,
        {
            "training_status": "trained",
            "train_row_count": len(train_rows),
            "verified_training_row_count": verified_count,
            "final_loss": round(final_loss, 8),
            **SAFETY_FLAGS,
        },
    )
    append_progress(progress_path, "finish", start_time=start_time, model_id=model_id, train_row_count=len(train_rows))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the torch audio quality critic shadow model.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--min-train-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="train_audio_quality_critic.py", entrypoint_path="/app/tools/train_audio_quality_critic.py")
    args = parse_args(argv)
    if str(args.device) != "cuda" and not bool(args.allow_non_gpu):
        raise ValueError("production audio quality critic training requires --device cuda unless --allow-non-gpu is set")
    manifest = train_audio_quality_critic(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=str(args.device),
        epochs=int(args.epochs),
        hidden_size=int(args.hidden_size),
        min_train_rows=int(args.min_train_rows),
    )
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "training_status": manifest["training_status"]}, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["training_status"] == "trained" else 2


if __name__ == "__main__":
    raise SystemExit(main())
