from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.qwen_controller_qlora_preflight import TARGET_MODEL_ID, snapshot_file_report  # noqa: E402

MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshot_download(**kwargs: Any) -> str:
    if kwargs.get("local_files_only") is False and os.environ.get("HF_HUB_OFFLINE") == "0":
        try:
            import huggingface_hub.constants as hub_constants  # type: ignore

            hub_constants.HF_HUB_OFFLINE = False
        except Exception:
            pass
    from huggingface_hub import snapshot_download as hf_snapshot_download  # type: ignore

    return str(hf_snapshot_download(**kwargs))


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_model_cache_download",
        "gpu_training_required": False,
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "qwen_model_cache_only",
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "qwen_model_cache_only",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def cache_roots(*, cache_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(cache_dir)
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home))
        roots.append(Path(hf_home) / "hub")
    return roots


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def enforce_cache_root(snapshot: dict[str, Any], *, cache_dir: Path | None) -> dict[str, Any]:
    roots = cache_roots(cache_dir=cache_dir)
    snapshot_path_text = str(snapshot.get("snapshot_path") or "")
    snapshot_path = Path(snapshot_path_text)
    allowed = True if not roots or not snapshot_path_text else any(is_within_root(snapshot_path, root) for root in roots)
    snapshot["cache_roots"] = [str(root) for root in roots]
    snapshot["cache_root_allowed"] = allowed
    if not allowed:
        snapshot["cache_available"] = False
        snapshot["blocker"] = "model_snapshot_outside_cache_root"
    return snapshot


class Heartbeat:
    def __init__(self, progress_path: Path, *, enabled: bool, interval_sec: float, model_id: str) -> None:
        self.progress_path = progress_path
        self.enabled = enabled
        self.interval_sec = max(float(interval_sec), 1.0)
        self.model_id = model_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, name="qwen-cache-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            progress(
                self.progress_path,
                "snapshot_download_heartbeat",
                target_model_id=self.model_id,
            )


def run_cache_qwen_controller_model(
    *,
    output_dir: Path,
    allow_network: bool = False,
    target_model_id: str = TARGET_MODEL_ID,
    revision: str = "",
    cache_dir: Path | None = None,
    max_workers: int = 4,
    heartbeat_interval_sec: float = 30.0,
) -> dict[str, Any]:
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()

    hf_hub_offline_before = str(os.environ.get("HF_HUB_OFFLINE") or "")
    hf_hub_disable_progress_bars_before = str(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") or "")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if allow_network:
        os.environ["HF_HUB_OFFLINE"] = "0"
    hf_hub_offline_during = str(os.environ.get("HF_HUB_OFFLINE") or "")
    hf_hub_disable_progress_bars_during = str(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") or "")
    local_files_only = not allow_network
    download_attempted = bool(allow_network)
    progress(
        progress_path,
        "start",
        target_model_id=target_model_id,
        allow_network=allow_network,
        local_files_only=local_files_only,
        output_manifest_path=str(output_dir / MANIFEST_NAME),
    )
    kwargs: dict[str, Any] = {
        "repo_id": target_model_id,
        "local_files_only": local_files_only,
        "max_workers": max_workers,
    }
    if revision:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    progress(
        progress_path,
        "snapshot_download_start",
        target_model_id=target_model_id,
        allow_network=allow_network,
        local_files_only=local_files_only,
        revision=revision,
        cache_dir=str(cache_dir or ""),
        max_workers=max_workers,
    )
    snapshot_path: Path | None = None
    download_error: dict[str, str] | None = None
    try:
        with Heartbeat(progress_path, enabled=allow_network, interval_sec=heartbeat_interval_sec, model_id=target_model_id):
            snapshot_path = Path(snapshot_download(**kwargs))
        progress(progress_path, "snapshot_download_finish", snapshot_path=str(snapshot_path))
        if allow_network:
            local_kwargs = dict(kwargs)
            local_kwargs["local_files_only"] = True
            progress(
                progress_path,
                "snapshot_offline_revalidate_start",
                target_model_id=target_model_id,
                revision=revision,
                cache_dir=str(cache_dir or ""),
            )
            snapshot_path = Path(snapshot_download(**local_kwargs))
            progress(progress_path, "snapshot_offline_revalidate_finish", snapshot_path=str(snapshot_path))
    except Exception as exc:
        download_error = {"type": type(exc).__name__, "message": str(exc)}
        progress(progress_path, "snapshot_download_failed", error=download_error)

    if snapshot_path is not None:
        snapshot = snapshot_file_report(snapshot_path, target_model_id=target_model_id, local_files_only=True)
    else:
        snapshot = {
            "target_model_id": target_model_id,
            "cache_available": False,
            "snapshot_path": "",
            "local_files_only": local_files_only,
            "required_files": {"config_json": False, "tokenizer": False, "weights": False},
            "missing_required_files": ["config_json", "tokenizer", "weights"],
            "blocker": "model_snapshot_not_cached",
        }
        if download_error is not None:
            snapshot["error"] = download_error
    snapshot = enforce_cache_root(snapshot, cache_dir=cache_dir)
    progress(
        progress_path,
        "snapshot_verified",
        cache_available=snapshot["cache_available"],
        missing_required_files=snapshot.get("missing_required_files", []),
        blocker=snapshot.get("blocker", ""),
    )

    failed_checks: list[str] = []
    if snapshot["cache_available"] is not True:
        failed_checks.append(str(snapshot.get("blocker") or "model_cache_unavailable"))
    cache_status = "ready" if not failed_checks else "blocked"
    manifest = {
        "schema": "qwen_controller_model_cache_manifest.v1",
        "manifest_type": "qwen_controller_model_cache",
        "cache_status": cache_status,
        "failed_checks": sorted(set(failed_checks)),
        "target_model_id": target_model_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "download_attempted": download_attempted,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "qwen_model_cache_not_for_listen_pack",
        "release_status": "model_cache_artifact_only",
        "release_ready": False,
        "snapshot": snapshot,
        "network": {
            "allow_network": allow_network,
            "local_files_only": local_files_only,
            "hf_hub_offline_before": hf_hub_offline_before,
            "hf_hub_offline_during": hf_hub_offline_during,
            "hf_hub_disable_progress_bars_before": hf_hub_disable_progress_bars_before,
            "hf_hub_disable_progress_bars_during": hf_hub_disable_progress_bars_during,
        },
        "execution_contract": {
            "docker_runtime_required": True,
            "network_requires_explicit_allow_network": True,
            "writes_progress_jsonl": True,
            "validates_snapshot_required_files": True,
            "does_not_start_training": True,
            "does_not_edit_model_registry": True,
        },
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "progress_jsonl": str(progress_path),
        },
        "safety": {
            "metadata_only": True,
            "not_release_artifact": True,
            "not_listen_artifact": True,
            "short_artifact_names": True,
            "blocks_incomplete_snapshot": True,
        },
        "recommended_next_step": "run_qwen_controller_qlora_preflight" if cache_status == "ready" else "rerun_with_allow_network_or_fix_hf_access",
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(progress_path, "finish", cache_status=cache_status, failed_checks=manifest["failed_checks"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache and validate the Qwen controller model snapshot inside Docker.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-model-id", default=TARGET_MODEL_ID)
    parser.add_argument("--revision", default="")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--heartbeat-interval-sec", type=float, default=30.0)
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="cache_qwen_controller_model.py",
        entrypoint_path="/app/tools/cache_qwen_controller_model.py",
    )
    args = parse_args(argv)
    manifest = run_cache_qwen_controller_model(
        output_dir=args.output_dir,
        allow_network=bool(args.allow_network),
        target_model_id=str(args.target_model_id),
        revision=str(args.revision or ""),
        cache_dir=args.cache_dir,
        max_workers=int(args.max_workers),
        heartbeat_interval_sec=float(args.heartbeat_interval_sec),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["outputs"]["manifest_json"],
                "cache_status": manifest["cache_status"],
                "failed_checks": manifest["failed_checks"],
                "snapshot_path": manifest["snapshot"].get("snapshot_path", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["cache_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
