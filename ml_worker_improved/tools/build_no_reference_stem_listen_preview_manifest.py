from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_status_contract import sha256_file  # noqa: E402
from tools.path_prefix_mapping import parse_path_prefix_map, resolve_audio_path  # noqa: E402


ALLOWED_STEMS = {"melody", "drums", "bass", "other"}
UNSUPPORTED_STEMS = {
    "both",
    "instrumental",
    "minus",
    "pair",
    "vocal",
    "vocal+minus",
    "vocal_minus",
    "vocal_minus_pair",
    "vocals",
}
CPU_DSP_ACCELERATION_PREFIX = "cpu_dsp"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "release_ready": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_stem(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def safety_from_source(source_manifest: dict[str, Any]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for source in (source_manifest, nested_dict(source_manifest, "safety"), nested_dict(source_manifest, "safety_flags")):
        for key in safety:
            safety[key] = bool(safety[key] or source.get(key) is True)
    return safety


def source_ref(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
    }


def source_claims_delivery(source_manifest: dict[str, Any]) -> bool:
    return (
        source_manifest.get("delivery_allowed") is True
        or source_manifest.get("release_ready") is True
        or source_manifest.get("can_mark_release_ready") is True
        or source_manifest.get("can_publish_as_release") is True
        or str(source_manifest.get("release_status") or "") == "release_ready"
    )


def source_contract_violation(source_manifest: dict[str, Any]) -> str:
    contract = nested_dict(source_manifest, "contract")
    if not contract:
        return "source_contract_missing"
    if contract.get("fallback_used") is True:
        return "source_contract_fallback_used"
    if contract.get("hidden_inference") is True:
        return "source_contract_hidden_inference"
    if contract.get("per_track_hardcode_allowed") is True:
        return "source_contract_per_track_hardcode"
    if contract.get("not_timestamp_hardcoded") is False:
        return "source_contract_timestamp_hardcode"
    return ""


def source_safety_violation(safety: dict[str, bool]) -> str:
    violations = [key for key, value in safety.items() if value is True]
    return violations[0] if violations else ""


def manifest_target_stem(source_manifest: dict[str, Any]) -> str:
    artifact = nested_dict(source_manifest, "target_stem_artifact")
    return normalize_stem(
        source_manifest.get("target_stem")
        or artifact.get("target_stem")
        or source_manifest.get("stem_type")
        or source_manifest.get("stem")
    )


def target_audio_path(source_manifest: dict[str, Any], target_stem: str) -> str:
    artifact = nested_dict(source_manifest, "target_stem_artifact")
    artifact_stem = normalize_stem(artifact.get("target_stem"))
    if artifact_stem == target_stem and str(artifact.get("path") or "").strip():
        return str(artifact.get("path")).strip()
    stems = nested_dict(source_manifest, "stems")
    stem_payload = stems.get(target_stem)
    if isinstance(stem_payload, dict) and str(stem_payload.get("path") or "").strip():
        return str(stem_payload.get("path")).strip()
    return ""


def gpu_provenance_gate(source_manifest: dict[str, Any], *, require_gpu_proof: bool) -> tuple[str, dict[str, Any]]:
    execution_contract = nested_dict(source_manifest, "execution_contract")
    cuda_preflight = nested_dict(execution_contract, "cuda_preflight")
    expected_acceleration_mode = str(execution_contract.get("expected_acceleration_mode") or "")
    gpu_inference_required = execution_contract.get("gpu_inference_required") is True
    gpu_inference_skipped = execution_contract.get("gpu_inference_skipped") is True
    cpu_dsp_only_route = (
        expected_acceleration_mode.startswith(CPU_DSP_ACCELERATION_PREFIX)
        and execution_contract.get("gpu_inference_required") is False
        and gpu_inference_skipped
        and str(execution_contract.get("gpu_inference_skip_reason") or "") == "cpu_only_dsp_route"
    )
    gate = {
        "status": "passed",
        "require_gpu_proof": require_gpu_proof,
        "expected_acceleration_mode": expected_acceleration_mode,
        "gpu_inference_required": gpu_inference_required,
        "gpu_inference_skipped": gpu_inference_skipped,
        "gpu_inference_skip_reason": str(execution_contract.get("gpu_inference_skip_reason") or ""),
        "cpu_dsp_only_route": cpu_dsp_only_route,
        "cuda_available": cuda_preflight.get("cuda_available") is True,
        "requested_device": str(cuda_preflight.get("requested_device") or ""),
        "resolved_device": str(cuda_preflight.get("resolved_device") or ""),
        "device_name": str(cuda_preflight.get("device_name") or ""),
    }
    if not require_gpu_proof:
        gate["status"] = "not_required"
        return "", gate
    if cpu_dsp_only_route:
        gate["reason"] = "cpu_dsp_route_gpu_not_required"
        return "", gate
    if (
        not gpu_inference_required
        or gpu_inference_skipped
        or cuda_preflight.get("cuda_available") is not True
        or str(cuda_preflight.get("resolved_device") or "").lower() != "cuda"
    ):
        gate["status"] = "blocked"
        gate["reason"] = "gpu_proof_missing"
        return "gpu_proof_missing", gate
    return "", gate


def audio_integrity_gate(
    audio_path: str,
    *,
    min_duration_sec: float,
    min_rms: float,
    min_peak: float,
    max_peak: float,
) -> tuple[str, dict[str, Any]]:
    gate = {
        "status": "blocked",
        "path": audio_path,
        "min_duration_sec": min_duration_sec,
        "min_rms": min_rms,
        "min_peak": min_peak,
        "max_peak": max_peak,
    }
    if not audio_path:
        gate["reason"] = "missing_audio_path"
        return "missing_audio_path", gate
    path = Path(audio_path)
    if not path.exists():
        gate["reason"] = "audio_path_missing"
        return "audio_path_missing", gate
    try:
        with sf.SoundFile(path) as audio_file:
            sample_rate = int(audio_file.samplerate or 0)
            frames = int(audio_file.frames or 0)
            channels = int(audio_file.channels or 0)
            duration_sec = float(frames / sample_rate) if sample_rate else 0.0
            total_square = 0.0
            sample_count = 0
            peak = 0.0
            for block in audio_file.blocks(blocksize=65_536, dtype="float32", always_2d=True):
                if block.size == 0:
                    continue
                total_square += float(np.sum(np.asarray(block, dtype=np.float64) ** 2))
                sample_count += int(block.size)
                peak = max(peak, float(np.max(np.abs(block))))
    except Exception as exc:  # pragma: no cover - exercised by corrupt external artifacts.
        gate["reason"] = "audio_read_failed"
        gate["error"] = str(exc)
        return "audio_read_failed", gate

    rms = math.sqrt(total_square / sample_count) if sample_count else 0.0
    gate.update(
        {
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "duration_sec": round(duration_sec, 6),
            "rms": round(rms, 10),
            "peak": round(peak, 10),
        }
    )
    if sample_rate <= 0 or frames <= 0 or channels <= 0:
        gate["reason"] = "audio_empty_or_invalid"
        return "audio_empty_or_invalid", gate
    if duration_sec < min_duration_sec:
        gate["reason"] = "audio_duration_below_floor"
        return "audio_duration_below_floor", gate
    if rms < min_rms:
        gate["reason"] = "audio_rms_below_floor"
        return "audio_rms_below_floor", gate
    if peak < min_peak:
        gate["reason"] = "audio_peak_below_floor"
        return "audio_peak_below_floor", gate
    if peak > max_peak:
        gate["reason"] = "audio_peak_above_ceiling"
        return "audio_peak_above_ceiling", gate
    gate["status"] = "passed"
    return "", gate


def source_contract_gate(source_manifest: dict[str, Any], safety: dict[str, bool]) -> tuple[str, dict[str, Any]]:
    gate = {
        "status": "passed",
        "source_delivery_allowed": source_manifest.get("delivery_allowed") is True,
        "source_release_status": str(source_manifest.get("release_status") or ""),
        "source_release_ready": source_manifest.get("release_ready") is True,
    }
    if source_claims_delivery(source_manifest):
        gate["status"] = "blocked"
        gate["reason"] = "source_manifest_claims_delivery"
        return "source_manifest_claims_delivery", gate
    contract_reason = source_contract_violation(source_manifest)
    if contract_reason:
        gate["status"] = "blocked"
        gate["reason"] = contract_reason
        return contract_reason, gate
    safety_reason = source_safety_violation(safety)
    if safety_reason:
        gate["status"] = "blocked"
        gate["reason"] = "source_safety_violation"
        gate["violating_flag"] = safety_reason
        return "source_safety_violation", gate
    return "", gate


def blocked_manifest(
    *,
    reason: str,
    stem_manifest_json: Path,
    source_manifest: dict[str, Any],
    target_stem: str,
    selected_audio_path: str,
    track_id: str,
    audio_gate: dict[str, Any],
    gpu_gate: dict[str, Any],
    contract_gate: dict[str, Any],
    path_resolution: dict[str, Any],
    safety: dict[str, bool],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": reason,
        "stem_type": target_stem,
        "track_id": track_id,
        "model_name": str(source_manifest.get("model_name") or ""),
        "source_audio_path": str(source_manifest.get("audio_path") or ""),
        "selected_audio_path": selected_audio_path,
        "selected_stem_path": selected_audio_path,
        "source_selected_audio_path": str(path_resolution.get("original_path") or selected_audio_path),
        "selected_audio_path_resolution": path_resolution,
        "audio_integrity_gate": audio_gate,
        "gpu_provenance_gate": gpu_gate,
        "source_contract_gate": contract_gate,
        "semantic_quality_verified": False,
        "semantic_quality_status": "not_verified_no_reference_audio_integrity_only",
        "accepted_count": 0,
        "skipped_count": 1,
        "skipped_rows": [
            {
                "stem_type": target_stem,
                "selected_audio_path": selected_audio_path,
                "skip_reason": reason,
            }
        ],
        "source_stem_manifest": source_ref(stem_manifest_json),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": safety,
    }


def accepted_manifest(
    *,
    stem_manifest_json: Path,
    source_manifest: dict[str, Any],
    target_stem: str,
    selected_audio_path: str,
    track_id: str,
    audio_gate: dict[str, Any],
    gpu_gate: dict[str, Any],
    contract_gate: dict[str, Any],
    path_resolution: dict[str, Any],
    safety: dict[str, bool],
) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": target_stem,
        "track_id": track_id,
        "model_name": str(source_manifest.get("model_name") or ""),
        "source_audio_path": str(source_manifest.get("audio_path") or ""),
        "selected_audio_path": selected_audio_path,
        "selected_stem_path": selected_audio_path,
        "source_selected_audio_path": str(path_resolution.get("original_path") or selected_audio_path),
        "selected_audio_path_resolution": path_resolution,
        "audio_integrity_gate": audio_gate,
        "gpu_provenance_gate": gpu_gate,
        "source_contract_gate": contract_gate,
        "selected_artifacts": {
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "listen_stem_flac": selected_audio_path,
            f"listen_{target_stem}_flac": selected_audio_path,
        },
        "semantic_quality_verified": False,
        "semantic_quality_status": "not_verified_no_reference_audio_integrity_only",
        "accepted_count": 1,
        "skipped_count": 0,
        "accepted_listen_preview_rows": [
            {
                "track_id": track_id,
                "stem_type": target_stem,
                "stem_path": selected_audio_path,
                "selected_audio_path": selected_audio_path,
                "source_selected_audio_path": str(path_resolution.get("original_path") or selected_audio_path),
                "selected_audio_path_resolution": path_resolution,
                "audio_integrity_gate": audio_gate,
                "gpu_provenance_gate": gpu_gate,
                "semantic_quality_verified": False,
            }
        ],
        "skipped_rows": [],
        "source_stem_manifest": source_ref(stem_manifest_json),
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": safety,
    }


def build_manifest(
    *,
    stem_manifest_json: Path,
    target_stem: str | None = None,
    track_id: str | None = None,
    min_duration_sec: float = 1.0,
    min_rms: float = 1e-4,
    min_peak: float = 1e-4,
    max_peak: float = 1.01,
    path_prefix_map: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    source_manifest = read_json(stem_manifest_json)
    manifest_stem = manifest_target_stem(source_manifest)
    selected_stem = normalize_stem(target_stem or manifest_stem)
    source_selected_audio_path = target_audio_path(source_manifest, selected_stem)
    selected_audio_path, path_resolution = resolve_audio_path(source_selected_audio_path, path_prefix_map)
    safety = safety_from_source(source_manifest)
    selected_track_id = str(track_id or source_manifest.get("track_id") or "")

    audio_reason, audio_gate = audio_integrity_gate(
        selected_audio_path,
        min_duration_sec=min_duration_sec,
        min_rms=min_rms,
        min_peak=min_peak,
        max_peak=max_peak,
    )
    gpu_reason, gpu_gate = gpu_provenance_gate(source_manifest, require_gpu_proof=True)
    contract_reason, contract_gate = source_contract_gate(source_manifest, safety)

    reason = ""
    if selected_stem not in ALLOWED_STEMS or selected_stem in UNSUPPORTED_STEMS:
        reason = "unsupported_stem_type"
    elif contract_reason:
        reason = contract_reason
    elif audio_reason:
        reason = audio_reason
    elif gpu_reason:
        reason = gpu_reason

    if reason:
        return blocked_manifest(
            reason=reason,
            stem_manifest_json=stem_manifest_json,
            source_manifest=source_manifest,
            target_stem=selected_stem,
            selected_audio_path=selected_audio_path,
            track_id=selected_track_id,
            audio_gate=audio_gate,
            gpu_gate=gpu_gate,
            contract_gate=contract_gate,
            path_resolution=path_resolution,
            safety=safety,
        )
    return accepted_manifest(
        stem_manifest_json=stem_manifest_json,
        source_manifest=source_manifest,
        target_stem=selected_stem,
        selected_audio_path=selected_audio_path,
        track_id=selected_track_id,
        audio_gate=audio_gate,
        gpu_gate=gpu_gate,
        contract_gate=contract_gate,
        path_resolution=path_resolution,
        safety=safety,
    )


def build_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Stem Listen Preview Summary",
        "",
        f"- delivery_status: {manifest.get('delivery_status')}",
        f"- delivery_allowed: {str(manifest.get('delivery_allowed') is True).lower()}",
        f"- presentation_status: {manifest.get('presentation_status')}",
        f"- release_status: {manifest.get('release_status')}",
        f"- stem_type: {manifest.get('stem_type')}",
        f"- accepted_count: {manifest.get('accepted_count')}",
        f"- skipped_count: {manifest.get('skipped_count')}",
        f"- selected_audio_path: {manifest.get('selected_audio_path')}",
        f"- dominant_failure_reason: {manifest.get('dominant_failure_reason')}",
        f"- semantic_quality_verified: {str(manifest.get('semantic_quality_verified') is True).lower()}",
        "",
        "## Gates",
        "",
        f"- audio_integrity_gate: {nested_dict(manifest, 'audio_integrity_gate').get('status')}",
        f"- gpu_provenance_gate: {nested_dict(manifest, 'gpu_provenance_gate').get('status')}",
        f"- source_contract_gate: {nested_dict(manifest, 'source_contract_gate').get('status')}",
        "",
        "## Safety",
        "",
    ]
    safety = nested_dict(manifest, "safety")
    for key in sorted(SAFETY_FLAGS):
        lines.append(f"- {key}: {str(safety.get(key) is True).lower()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a contract-safe no-reference non-vocal stem listen-preview manifest.")
    parser.add_argument("--stem-manifest-json", type=Path, required=True)
    parser.add_argument("--target-stem", default="")
    parser.add_argument("--track-id", default="")
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=1e-4)
    parser.add_argument("--min-peak", type=float, default=1e-4)
    parser.add_argument("--max-peak", type=float, default=1.01)
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Map source manifest audio paths into the local artifact mirror before gating.",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(
        stem_manifest_json=args.stem_manifest_json,
        target_stem=args.target_stem or None,
        track_id=args.track_id or None,
        min_duration_sec=args.min_duration_sec,
        min_rms=args.min_rms,
        min_peak=args.min_peak,
        max_peak=args.max_peak,
        path_prefix_map=parse_path_prefix_map(args.path_prefix_map),
    )
    write_json(args.output_manifest_json, manifest)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "delivery_status",
                    "delivery_allowed",
                    "presentation_status",
                    "release_status",
                    "stem_type",
                    "accepted_count",
                    "skipped_count",
                    "dominant_failure_reason",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
