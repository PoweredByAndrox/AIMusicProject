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
    MANIFEST_FILE,
    MODEL_FILE,
    PROGRESS_FILE,
    SAFETY_FLAGS,
    SCORE_MANIFEST_SCHEMA,
    SCORES_FILE,
    TARGET_NAMES,
    append_progress,
    as_list,
    feature_vector_from_row,
    iter_jsonl,
    pad_feature_vectors,
    read_json,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime
from tools.train_audio_quality_critic import AudioQualityCriticNet

try:  # pragma: no cover - exercised in Docker tests.
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


ACCEPT_PROXY_DECISIONS = {"accept_candidate", "accepted", "accepted_for_listen_pack", "pair_compatible"}
BLOCK_PROXY_DECISIONS = {"blocked_hard_case", "blocked_pair_incompatible", "block_candidate", "rejected"}


def _load_model(model_dir: Path) -> tuple[dict[str, Any], Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch_unavailable")
    manifest = read_json(model_dir / MANIFEST_FILE)
    bundle = torch.load(model_dir / MODEL_FILE, map_location="cpu", weights_only=False)
    model = AudioQualityCriticNet(
        input_width=int(bundle["input_width"]),
        hidden_size=int(bundle["hidden_size"]),
        output_width=len(bundle["target_names"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return manifest, model


def _predict(model: Any, rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    vectors, width = pad_feature_vectors([feature_vector_from_row(row) for row in rows])
    expected_width = int(next(model.parameters()).shape[1])
    if width < expected_width:
        vectors = [vector + [0.0] * (expected_width - width) for vector in vectors]
    elif width > expected_width:
        vectors = [vector[:expected_width] for vector in vectors]
    with torch.no_grad():
        predictions = model(torch.tensor(vectors, dtype=torch.float32)).detach().cpu().tolist()
    return [
        {name: round(float(value), 6) for name, value in zip(TARGET_NAMES, prediction)}
        for prediction in predictions
    ]


def _proxy_accepts(row: dict[str, Any]) -> bool:
    decision = str(row.get("proxy_decision") or row.get("decision") or row.get("pair_decision") or "").strip()
    if decision in ACCEPT_PROXY_DECISIONS:
        return True
    if decision in BLOCK_PROXY_DECISIONS:
        return False
    return bool(row.get("proxy_accept") is True)


def _score_row(row: dict[str, Any], prediction: dict[str, float], *, accept_threshold: float) -> dict[str, Any]:
    critic_accepts = float(prediction["accept_probability"]) >= float(accept_threshold)
    proxy_accepts = _proxy_accepts(row)
    blockers = [str(item) for item in as_list(row.get("blocker_reasons")) + as_list(row.get("failed_checks"))]
    safety_blocked = bool(blockers)
    disagreement = proxy_accepts != critic_accepts
    if disagreement or safety_blocked:
        selected_action = "block_for_review"
    elif critic_accepts:
        selected_action = "shadow_accept_observation"
    else:
        selected_action = "shadow_block_observation"
    return {
        **row,
        **prediction,
        "critic_accepts": critic_accepts,
        "proxy_accepts": proxy_accepts,
        "proxy_critic_disagreement": disagreement,
        "safety_blocked": safety_blocked,
        "selected_action": selected_action,
        "shadow_only": True,
        "delivery_allowed": False,
        "presentation_status": "audio_quality_critic_shadow_score_not_for_listen_pack",
        "release_ready": False,
    }


def score_audio_quality_windows(
    *,
    candidate_jsonl: Path,
    model_dir: Path,
    output_dir: Path,
    accept_threshold: float = 0.5,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(progress_path, "start", start_time=start_time, candidate_jsonl=str(candidate_jsonl), model_dir=str(model_dir))
    model_manifest, model = _load_model(model_dir)
    rows = list(iter_jsonl(candidate_jsonl))
    predictions = _predict(model, rows)
    scored_rows = [_score_row(row, prediction, accept_threshold=accept_threshold) for row, prediction in zip(rows, predictions)]
    write_jsonl(output_dir / SCORES_FILE, scored_rows)
    disagreement_count = sum(1 for row in scored_rows if row["proxy_critic_disagreement"])
    manifest = {
        "schema": SCORE_MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_critic_score",
        "candidate_jsonl": str(candidate_jsonl),
        "model_dir": str(model_dir),
        "model_id": str(model_manifest.get("model_id") or ""),
        "source_lock_pack_json": str(model_manifest.get("source_lock_pack_json") or ""),
        "scores_jsonl": str(output_dir / SCORES_FILE),
        "row_count": len(scored_rows),
        "disagreement_count": disagreement_count,
        "accept_threshold": float(accept_threshold),
        "shadow_only": True,
        "presentation_status": "audio_quality_critic_shadow_scores_not_for_listen_pack",
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(progress_path, "finish", start_time=start_time, row_count=len(scored_rows), disagreement_count=disagreement_count)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate windows with the audio quality critic shadow model.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="score_audio_quality_windows.py", entrypoint_path="/app/tools/score_audio_quality_windows.py")
    args = parse_args(argv)
    manifest = score_audio_quality_windows(
        candidate_jsonl=args.candidate_jsonl,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        accept_threshold=float(args.accept_threshold),
    )
    print(json.dumps({"manifest_json": str(args.output_dir / MANIFEST_FILE), "disagreement_count": manifest["disagreement_count"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
