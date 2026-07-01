"""Run Mac MelBand MPS export and pitch-robust leak guard as one diagnostic profile."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from tools.mac_melband_pitch_guard import run_pitch_guard_on_existing_melband_run


ROUTE_ID = "melband_roformer_vocals"
PROFILE = "melband_pitch_guard_v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, ensure_ascii=False) + "\n")
        handle.flush()


def submit_mac_mps_job(*, api_url: str, source_audio_path: str, output_subdir: str) -> dict[str, Any]:
    payload = {
        "schema": "mac_mps_separator_api_request.v1",
        "route_id": ROUTE_ID,
        "model_name": ROUTE_ID,
        "source_audio_path": str(source_audio_path),
        "output_subdir": str(output_subdir),
    }
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/jobs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return json.loads(response.read().decode("utf-8"))


def poll_mac_mps_job(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_sec)
    last_manifest: dict[str, Any] = {}
    while time.time() <= deadline:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/jobs/{job_id}", timeout=30.0) as response:
            last_manifest = json.loads(response.read().decode("utf-8"))
        status = str(last_manifest.get("status") or "").lower()
        if status in {"ok", "failed"}:
            return last_manifest
        time.sleep(max(0.1, float(poll_sec)))
    last_manifest.setdefault("job_id", job_id)
    last_manifest["status"] = "failed"
    last_manifest["error"] = "mac_mps_job_poll_timeout"
    return last_manifest


def _resolve_run_dir(job_manifest: Mapping[str, Any]) -> Path:
    run_dir = Path(str(job_manifest.get("run_dir") or ""))
    if run_dir.exists():
        return run_dir
    output_dir = Path(str(job_manifest.get("output_dir") or ""))
    candidate = output_dir / "run"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"mac_mps_run_dir_missing:{run_dir or candidate}")


def run_pipeline(
    *,
    mac_api_url: str,
    source_audio_path: str,
    output_dir: Path,
    api_output_subdir: str,
    poll_sec: float = 2.0,
    timeout_sec: float = 7200.0,
    guard_config: Mapping[str, Any] | None = None,
    submit_job: Callable[..., dict[str, Any]] = submit_mac_mps_job,
    poll_job: Callable[..., dict[str, Any]] = poll_mac_mps_job,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(progress_path, "mac_api_submit", "start", route_id=ROUTE_ID)
    queued_manifest = submit_job(
        api_url=str(mac_api_url),
        source_audio_path=str(source_audio_path),
        output_subdir=str(api_output_subdir),
    )
    job_id = str(queued_manifest.get("job_id") or "")
    if not job_id:
        raise RuntimeError("mac_mps_api_missing_job_id")
    _write_json(output_dir / "mac_job_queued.json", queued_manifest)
    _append_progress(progress_path, "mac_api_submit", "end", job_id=job_id)

    _append_progress(progress_path, "mac_api_poll", "start", job_id=job_id)
    job_manifest = poll_job(api_url=str(mac_api_url), job_id=job_id, poll_sec=float(poll_sec), timeout_sec=float(timeout_sec))
    _write_json(output_dir / "mac_job_final.json", job_manifest)
    _append_progress(progress_path, "mac_api_poll", "end", job_status=job_manifest.get("status"), job_id=job_id)
    if str(job_manifest.get("status") or "").lower() != "ok":
        manifest = {
            "schema": "mac_melband_pitch_guard_pipeline.v1",
            "profile": PROFILE,
            "route_id": ROUTE_ID,
            "source_audio_path": str(source_audio_path),
            "mac_api_url": str(mac_api_url),
            "api_output_subdir": str(api_output_subdir),
            "mac_job": job_manifest,
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "diagnostic_melband_pitch_guard_only",
            "blocker_reasons": ["mac_mps_export_failed"],
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest

    _append_progress(progress_path, "pitch_guard", "start", job_id=job_id)
    guard_manifest = run_pitch_guard_on_existing_melband_run(
        melband_run_dir=_resolve_run_dir(job_manifest),
        output_dir=output_dir / "guard",
        config=guard_config,
    )
    _append_progress(progress_path, "pitch_guard", "end", guard_verdict=guard_manifest.get("delivery_status"))

    blocker_reasons = list(guard_manifest.get("blocker_reasons") or [])
    manifest = {
        "schema": "mac_melband_pitch_guard_pipeline.v1",
        "profile": PROFILE,
        "route_id": ROUTE_ID,
        "source_audio_path": str(source_audio_path),
        "mac_api_url": str(mac_api_url),
        "api_output_subdir": str(api_output_subdir),
        "mac_job": job_manifest,
        "guard_manifest": guard_manifest,
        "delivery_status": "blocked" if blocker_reasons else "diagnostic_passed",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "diagnostic_melband_pitch_guard_only",
        "blocker_reasons": sorted(set(blocker_reasons)),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mac MelBand pitch guard diagnostic profile.")
    parser.add_argument("--mac-api-url", required=True)
    parser.add_argument("--source-audio-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-output-subdir", required=True)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--guard-config-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    guard_config = json.loads(args.guard_config_json) if args.guard_config_json else None
    manifest = run_pipeline(
        mac_api_url=args.mac_api_url,
        source_audio_path=args.source_audio_path,
        output_dir=args.out,
        api_output_subdir=args.api_output_subdir,
        poll_sec=args.poll_sec,
        timeout_sec=args.timeout_sec,
        guard_config=guard_config,
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "delivery_allowed": manifest["delivery_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
