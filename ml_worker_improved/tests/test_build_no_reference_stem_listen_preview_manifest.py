from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.artifact_status_contract import status_view_from_manifest
from tools.build_no_reference_stem_listen_preview_manifest import build_manifest, main


def write_audio(path: Path, *, amplitude: float = 0.05, duration_sec: float = 1.25) -> Path:
    sample_rate = 16_000
    frame_count = int(sample_rate * duration_sec)
    if amplitude:
        t = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
        mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32) * amplitude
    else:
        mono = np.zeros(frame_count, dtype=np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, sample_rate)
    return path


def write_source_manifest(
    path: Path,
    *,
    target_stem: str,
    audio_path: Path,
    gpu_proof: bool = True,
    include_contract: bool = True,
    expected_acceleration_mode: str = "gpu_inference_then_cpu_dsp_scoring",
    gpu_inference_required: bool = True,
    gpu_inference_skipped: bool | None = None,
    gpu_skip_reason: str = "",
) -> Path:
    if gpu_inference_skipped is None:
        gpu_inference_skipped = not gpu_proof
    payload = {
        "audio_path": "/host_testing_stuff_music/source.mp3",
        "model_name": "htdemucs_6s",
        "sample_rate": 44100,
        "execution_contract": {
            "docker_runtime_required": True,
            "expected_acceleration_mode": expected_acceleration_mode,
            "gpu_inference_required": gpu_inference_required,
            "gpu_inference_skipped": gpu_inference_skipped,
            "gpu_inference_skip_reason": gpu_skip_reason,
            "cuda_preflight": {
                "cuda_available": gpu_proof,
                "requested_device": "cuda",
                "resolved_device": "cuda" if gpu_proof else "cpu",
                "device_count": 1 if gpu_proof else 0,
                "device_name": "NVIDIA GeForce RTX 4080 Laptop GPU" if gpu_proof else "",
            },
        },
        "stems": {
            target_stem: {
                "path": str(audio_path),
                "source": "existing_manifest_stem",
                "source_stems": [target_stem],
            }
        },
        "target_stem": target_stem,
        "target_stem_artifact": {
            "target_stem": target_stem,
            "path": str(audio_path),
            "source": "existing_manifest_stem",
            "source_stems": [target_stem],
        },
        "delivery_allowed": False,
        "presentation_status": "diagnostic_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
    }
    if include_contract:
        payload["contract"] = {
            "application_reference_free": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "route_name": "full_res_stem_export",
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stem_listen_preview_accepts_gpu_generated_non_vocal_stem(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "melody.flac")
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="melody", audio_path=audio_path)

    manifest = build_manifest(
        stem_manifest_json=source_manifest,
        target_stem="melody",
        track_id="track-a",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "melody"
    assert manifest["selected_audio_path"] == str(audio_path)
    assert manifest["semantic_quality_verified"] is False
    assert manifest["release_ready"] is False
    assert manifest["safety"]["training_started"] is False
    assert manifest["audio_integrity_gate"]["status"] == "passed"
    assert manifest["gpu_provenance_gate"]["status"] == "passed"


def test_multistem_manifest_can_select_existing_non_manifest_target_stem(tmp_path: Path) -> None:
    bass_path = write_audio(tmp_path / "bass.flac")
    drums_path = write_audio(tmp_path / "drums.flac", amplitude=0.06)
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="bass", audio_path=bass_path)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["stems"]["drums"] = {
        "path": str(drums_path),
        "source": "existing_manifest_stem",
        "source_stems": ["drums"],
    }
    source_manifest.write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_manifest(
        stem_manifest_json=source_manifest,
        target_stem="drums",
        track_id="track-a",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["stem_type"] == "drums"
    assert manifest["selected_audio_path"] == str(drums_path)
    assert manifest["dominant_failure_reason"] == ""
    assert manifest["audio_integrity_gate"]["status"] == "passed"


def test_path_prefix_map_resolves_mac_local_stem_path_to_artifact_mirror(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "artifacts" / "mac_api" / "altm002" / "bci"
    mirror_audio_path = write_audio(mirror_dir / "bass.flac")
    mac_source_dir = "/Users/andrejviluha/aimusic_local_artifacts/mac_api/altm002/bci"
    mac_audio_path = Path(f"{mac_source_dir}/bass.flac")
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="bass", audio_path=mac_audio_path)

    blocked = build_manifest(stem_manifest_json=source_manifest, target_stem="bass")
    manifest = build_manifest(
        stem_manifest_json=source_manifest,
        target_stem="bass",
        path_prefix_map=[(mac_source_dir, str(mirror_dir))],
    )

    assert blocked["delivery_status"] == "blocked"
    assert blocked["dominant_failure_reason"] == "audio_path_missing"
    assert manifest["delivery_status"] == "accepted"
    assert manifest["selected_audio_path"] == str(mirror_audio_path)
    assert manifest["source_selected_audio_path"] == str(mac_audio_path)
    assert manifest["selected_audio_path_resolution"]["mapped"] is True
    assert manifest["selected_audio_path_resolution"]["original_path"] == str(mac_audio_path)
    assert manifest["selected_audio_path_resolution"]["resolved_path"] == str(mirror_audio_path)
    assert manifest["audio_integrity_gate"]["path"] == str(mirror_audio_path)


def test_blocks_unsupported_vocal_minus_or_pair_stems(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "vocals.flac")
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="vocals", audio_path=audio_path)

    manifest = build_manifest(stem_manifest_json=source_manifest, target_stem="vocals")

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "unsupported_stem_type"


def test_blocks_silent_audio(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "bass.flac", amplitude=0.0)
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="bass", audio_path=audio_path)

    manifest = build_manifest(stem_manifest_json=source_manifest, target_stem="bass")

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "audio_rms_below_floor"
    assert manifest["audio_integrity_gate"]["status"] == "blocked"


def test_blocks_missing_gpu_proof(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "drums.flac")
    source_manifest = write_source_manifest(
        tmp_path / "stem_manifest.json",
        target_stem="drums",
        audio_path=audio_path,
        gpu_proof=False,
    )

    manifest = build_manifest(stem_manifest_json=source_manifest, target_stem="drums")

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "gpu_proof_missing"
    assert manifest["gpu_provenance_gate"]["status"] == "blocked"


def test_accepts_explicit_cpu_dsp_only_candidate_without_fake_gpu_proof(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "melody.flac")
    source_manifest = write_source_manifest(
        tmp_path / "stem_manifest.json",
        target_stem="melody",
        audio_path=audio_path,
        gpu_proof=False,
        expected_acceleration_mode="cpu_dsp_candidate_generation",
        gpu_inference_required=False,
        gpu_inference_skipped=True,
        gpu_skip_reason="cpu_only_dsp_route",
    )

    manifest = build_manifest(stem_manifest_json=source_manifest, target_stem="melody")

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["gpu_provenance_gate"]["status"] == "passed"
    assert manifest["gpu_provenance_gate"]["cpu_dsp_only_route"] is True
    assert manifest["gpu_provenance_gate"]["gpu_inference_required"] is False
    assert manifest["gpu_provenance_gate"]["gpu_inference_skipped"] is True


def test_blocks_missing_source_contract(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "melody.flac")
    source_manifest = write_source_manifest(
        tmp_path / "stem_manifest.json",
        target_stem="melody",
        audio_path=audio_path,
        include_contract=False,
    )

    manifest = build_manifest(stem_manifest_json=source_manifest, target_stem="melody")

    assert manifest["delivery_status"] == "blocked"
    assert manifest["dominant_failure_reason"] == "source_contract_missing"
    assert manifest["source_contract_gate"]["status"] == "blocked"


def test_cli_writes_manifest_and_summary(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "other.flac")
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="other", audio_path=audio_path)
    output_manifest = tmp_path / "out" / "manifest.json"
    summary_md = tmp_path / "out" / "summary.md"

    exit_code = main(
        [
            "--stem-manifest-json",
            str(source_manifest),
            "--target-stem",
            "other",
            "--track-id",
            "track-a",
            "--output-manifest-json",
            str(output_manifest),
            "--summary-md",
            str(summary_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["delivery_status"] == "accepted"
    assert "- stem_type: other" in summary_md.read_text(encoding="utf-8")
    view = status_view_from_manifest(output_manifest)
    assert view.listenable
    assert view.stem_type == "other"


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    audio_path = write_audio(tmp_path / "bass.flac")
    source_manifest = write_source_manifest(tmp_path / "stem_manifest.json", target_stem="bass", audio_path=audio_path)
    output_manifest = tmp_path / "out" / "manifest.json"
    summary_md = tmp_path / "out" / "summary.md"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_no_reference_stem_listen_preview_manifest.py",
            "--stem-manifest-json",
            str(source_manifest),
            "--target-stem",
            "bass",
            "--output-manifest-json",
            str(output_manifest),
            "--summary-md",
            str(summary_md),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_manifest.read_text(encoding="utf-8"))["delivery_status"] == "accepted"
