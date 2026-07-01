#!/usr/bin/env python3
"""Run a staged Mac/MPS track pipeline with separate vocal/minus and stem lanes.

This runner is diagnostic-only. It coordinates existing Mac API routes, writes
one manifest, and blocks if either lane is missing required evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mac_audio_damage_audit import run_damage_audit
from tools.mac_melband_pitch_guard import run_pitch_guard_on_existing_melband_run


PROFILE = "mac_unified_track_pipeline_v1"
MELBAND_ROUTE_ID = "melband_roformer_vocals"
STEM_ROUTE_ID = "htdemucs_6s"
REQUIRED_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, ensure_ascii=False) + "\n")
        handle.flush()


def submit_mac_mps_job(*, api_url: str, source_audio_path: str, output_subdir: str, route_id: str) -> dict[str, Any]:
    payload = {
        "schema": "mac_mps_separator_api_request.v1",
        "route_id": str(route_id),
        "model_name": str(route_id),
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


def _blocked_payload(schema: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": schema,
        "verdict": "blocked_unknown",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "listen_pack_created": False,
        "blocker_reasons": [str(reason)],
        **extra,
    }


def _run_route(
    *,
    route_id: str,
    mac_api_url: str,
    source_audio_path: str,
    output_subdir: str,
    output_dir: Path,
    progress_path: Path,
    poll_sec: float,
    timeout_sec: float,
    submit_job: Callable[..., dict[str, Any]],
    poll_job: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    lane_dir = output_dir / ("mel" if route_id == MELBAND_ROUTE_ID else "stm")
    _append_progress(progress_path, f"{route_id}:submit", "start", output_subdir=output_subdir)
    queued = submit_job(
        api_url=str(mac_api_url),
        source_audio_path=str(source_audio_path),
        output_subdir=str(output_subdir),
        route_id=str(route_id),
    )
    _write_json(lane_dir / "queued.json", queued)
    job_id = str(queued.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"mac_api_missing_job_id:{route_id}")
    _append_progress(progress_path, f"{route_id}:submit", "end", job_id=job_id)
    _append_progress(progress_path, f"{route_id}:poll", "start", job_id=job_id)
    final = poll_job(api_url=str(mac_api_url), job_id=job_id, poll_sec=float(poll_sec), timeout_sec=float(timeout_sec))
    _write_json(lane_dir / "final.json", final)
    _append_progress(progress_path, f"{route_id}:poll", "end", job_id=job_id, job_status=final.get("status"))
    return final


def _summarize_stem_lane(job_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if str(job_manifest.get("status") or "").lower() != "ok":
        return {
            "route_id": STEM_ROUTE_ID,
            "job_status": job_manifest.get("status"),
            "required_stems_present": False,
            "available_stems": [],
            "missing_required_stems": list(REQUIRED_STEMS),
            "blocker_reasons": ["mac_stem_export_failed"],
        }
    try:
        run_dir = _resolve_run_dir(job_manifest)
    except FileNotFoundError as exc:
        return {
            "route_id": STEM_ROUTE_ID,
            "job_status": job_manifest.get("status"),
            "required_stems_present": False,
            "available_stems": [],
            "missing_required_stems": list(REQUIRED_STEMS),
            "blocker_reasons": ["mac_stem_run_dir_missing"],
            "error": str(exc),
        }
    stem_manifest_path = run_dir / "stem_manifest.json"
    if not stem_manifest_path.is_file():
        return {
            "route_id": STEM_ROUTE_ID,
            "job_status": job_manifest.get("status"),
            "run_dir": str(run_dir),
            "stem_manifest": str(stem_manifest_path),
            "required_stems_present": False,
            "available_stems": [],
            "missing_required_stems": list(REQUIRED_STEMS),
            "blocker_reasons": ["blocked_missing_stem_manifest"],
        }
    stem_manifest = json.loads(stem_manifest_path.read_text(encoding="utf-8"))
    stems = dict(stem_manifest.get("stems") or {})
    available = sorted(str(name) for name in stems)
    missing = [stem for stem in REQUIRED_STEMS if stem not in stems]
    blocker_reasons = ["blocked_missing_required_stems"] if missing else []
    return {
        "route_id": STEM_ROUTE_ID,
        "job_status": job_manifest.get("status"),
        "run_dir": str(run_dir),
        "stem_manifest": str(stem_manifest_path),
        "required_stems": list(REQUIRED_STEMS),
        "available_stems": available,
        "missing_required_stems": missing,
        "required_stems_present": not missing,
        "stems": stems,
        "blocker_reasons": blocker_reasons,
    }


def _run_melband_damage_audit_if_available(*, guard_manifest: Mapping[str, Any], output_dir: Path) -> dict[str, Any] | None:
    repair_manifest = guard_manifest.get("repair")
    if not isinstance(repair_manifest, Mapping):
        has_guard_failure = bool(guard_manifest.get("blocker_reasons")) or str(guard_manifest.get("delivery_status") or "") == "blocked"
        if has_guard_failure:
            return _blocked_payload(
                "mac_instrumental_damage_audit.v1",
                "damage_audit_missing_repair_manifest",
                presentation_status="damage_audit_diagnostic_only",
            )
        return {
            "schema": "mac_instrumental_damage_audit.v1",
            "verdict": "pass",
            "blocker_reasons": [],
            "skipped": True,
            "skip_reason": "no_repair_candidate_needed",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "damage_audit_diagnostic_only",
        }
    audio_map = guard_manifest.get("audio")
    if not isinstance(audio_map, Mapping):
        return _blocked_payload(
            "mac_instrumental_damage_audit.v1",
            "damage_audit_missing_audio_manifest",
            presentation_status="damage_audit_diagnostic_only",
        )
    reference_instrumental = audio_map.get("instrumental")
    candidate_instrumental = repair_manifest.get("repaired_instrumental_path") or repair_manifest.get("instrumental")
    if not reference_instrumental or not candidate_instrumental:
        return _blocked_payload(
            "mac_instrumental_damage_audit.v1",
            "damage_audit_missing_instrumental_paths",
            presentation_status="damage_audit_diagnostic_only",
        )
    score_json = output_dir / "mel" / "guard" / "score" / "pitch.json"
    if not score_json.is_file():
        score_payload = guard_manifest.get("score") if isinstance(guard_manifest.get("score"), Mapping) else {}
        score_json.parent.mkdir(parents=True, exist_ok=True)
        score_json.write_text(json.dumps(dict(score_payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    try:
        return run_damage_audit(
            reference_instrumental_path=Path(str(reference_instrumental)),
            candidate_instrumental_path=Path(str(candidate_instrumental)),
            score_json_path=score_json,
            output_dir=output_dir / "mel" / "damage",
        )
    except Exception as exc:
        return _blocked_payload(
            "mac_instrumental_damage_audit.v1",
            "damage_audit_failed",
            error=str(exc),
            presentation_status="damage_audit_diagnostic_only",
        )


def run_unified_pipeline(
    *,
    mac_api_url: str,
    source_audio_path: str,
    output_dir: Path,
    mel_api_output_subdir: str,
    stem_api_output_subdir: str,
    poll_sec: float = 2.0,
    timeout_sec: float = 7200.0,
    guard_config: Mapping[str, Any] | None = None,
    submit_job: Callable[..., dict[str, Any]] = submit_mac_mps_job,
    poll_job: Callable[..., dict[str, Any]] = poll_mac_mps_job,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    _append_progress(
        progress_path,
        "preflight",
        "start",
        profile=PROFILE,
        source_audio_path=str(source_audio_path),
        routes=[MELBAND_ROUTE_ID, STEM_ROUTE_ID],
    )

    blocker_reasons: list[str] = []
    mel_job = _run_route(
        route_id=MELBAND_ROUTE_ID,
        mac_api_url=mac_api_url,
        source_audio_path=source_audio_path,
        output_subdir=mel_api_output_subdir,
        output_dir=output_dir,
        progress_path=progress_path,
        poll_sec=poll_sec,
        timeout_sec=timeout_sec,
        submit_job=submit_job,
        poll_job=poll_job,
    )
    if str(mel_job.get("status") or "").lower() == "ok":
        _append_progress(progress_path, "melband_pitch_guard", "start")
        try:
            mel_run_dir = _resolve_run_dir(mel_job)
        except FileNotFoundError as exc:
            mel_guard = {
                "schema": "mac_melband_pitch_guard_run.v1",
                "delivery_status": "blocked",
                "delivery_allowed": False,
                "listen_pack_created": False,
                "presentation_status": "diagnostic_melband_pitch_guard_only",
                "blocker_reasons": ["mac_melband_run_dir_missing"],
                "error": str(exc),
            }
            damage_audit = _blocked_payload(
                "mac_instrumental_damage_audit.v1",
                "damage_audit_missing_melband_run_dir",
                presentation_status="damage_audit_diagnostic_only",
            )
        else:
            mel_guard = run_pitch_guard_on_existing_melband_run(
                melband_run_dir=mel_run_dir,
                output_dir=output_dir / "mel" / "guard",
                config=guard_config,
            )
            damage_audit = _run_melband_damage_audit_if_available(guard_manifest=mel_guard, output_dir=output_dir)
        _append_progress(progress_path, "melband_pitch_guard", "end", delivery_status=mel_guard.get("delivery_status"))
        blocker_reasons.extend(str(reason) for reason in mel_guard.get("blocker_reasons") or [])
        if damage_audit is not None:
            blocker_reasons.extend(str(reason) for reason in damage_audit.get("blocker_reasons") or [])
    else:
        mel_guard = {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "blocker_reasons": ["mac_melband_export_failed"],
        }
        damage_audit = None
        blocker_reasons.append("mac_melband_export_failed")

    stem_job = _run_route(
        route_id=STEM_ROUTE_ID,
        mac_api_url=mac_api_url,
        source_audio_path=source_audio_path,
        output_subdir=stem_api_output_subdir,
        output_dir=output_dir,
        progress_path=progress_path,
        poll_sec=poll_sec,
        timeout_sec=timeout_sec,
        submit_job=submit_job,
        poll_job=poll_job,
    )
    stem_lane = _summarize_stem_lane(stem_job)
    blocker_reasons.extend(str(reason) for reason in stem_lane.get("blocker_reasons") or [])

    blocker_reasons = sorted(set(blocker_reasons))
    status = "blocked" if blocker_reasons else "diagnostic_passed"
    manifest = {
        "schema": "mac_unified_track_pipeline.v1",
        "profile": PROFILE,
        "source_audio_path": str(source_audio_path),
        "mac_api_url": str(mac_api_url),
        "routes": [MELBAND_ROUTE_ID, STEM_ROUTE_ID],
        "lane_contract": {
            "independent_lane_winners": True,
            "vocal_minus_lane": MELBAND_ROUTE_ID,
            "stem_lane": STEM_ROUTE_ID,
            "do_not_regenerate_passed_vocal_minus_for_stem_failures": True,
            "do_not_replace_passed_lane_with_failed_sibling_lane": True,
        },
        "melband_lane": {
            "route_id": MELBAND_ROUTE_ID,
            "job": mel_job,
            "guard_manifest": mel_guard,
            "damage_audit": damage_audit,
        },
        "stem_lane": stem_lane,
        "blocker_reasons": blocker_reasons,
        "delivery_status": status,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "diagnostic_unified_mac_pipeline_only",
        "contract": {
            "diagnostic_only": True,
            "no_listen_delivery": True,
            "melband_lane_required": True,
            "stem_lane_required": True,
            "required_stems": list(REQUIRED_STEMS),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", delivery_status=status, blocker_reasons=blocker_reasons)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-api-url", required=True)
    parser.add_argument("--source-audio-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mel-api-output-subdir", required=True)
    parser.add_argument("--stem-api-output-subdir", required=True)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--guard-config-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    guard_config = json.loads(args.guard_config_json) if args.guard_config_json else None
    manifest = run_unified_pipeline(
        mac_api_url=str(args.mac_api_url),
        source_audio_path=str(args.source_audio_path),
        output_dir=Path(args.out),
        mel_api_output_subdir=str(args.mel_api_output_subdir),
        stem_api_output_subdir=str(args.stem_api_output_subdir),
        poll_sec=float(args.poll_sec),
        timeout_sec=float(args.timeout_sec),
        guard_config=guard_config,
    )
    print(json.dumps({"manifest": str(Path(args.out) / "manifest.json"), "delivery_status": manifest["delivery_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
