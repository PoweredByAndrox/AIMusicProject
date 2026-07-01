from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_no_reference_stem_candidate_score_matrix import build_stem_candidate_score_matrix, main


def write_audio(path: Path, *, amplitude: float = 0.05) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32) * amplitude
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, sample_rate)
    return path


def write_listen_manifest(
    path: Path,
    *,
    stem_type: str,
    audio_path: Path,
    delivery_status: str = "pending_semantic_judge",
    delivery_allowed: bool = False,
    presentation_status: str = "semantic_judge_input_not_for_listen_pack",
    release_status: str = "semantic_judge_input_only",
    semantic_judge_input: bool = True,
    audio_gate_status: str = "passed",
    gpu_gate_status: str = "passed",
    source_gate_status: str = "passed",
    expected_acceleration_mode: str = "gpu_inference_then_cpu_dsp_scoring",
    gpu_inference_required: bool = True,
    gpu_inference_skipped: bool = False,
    cuda_available: bool = True,
    resolved_device: str = "cuda",
    cpu_dsp_only_route: bool = False,
    rms: float = 0.05,
    peak: float = 0.5,
    safety_overrides: dict[str, bool] | None = None,
) -> Path:
    safety = {
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "cleanup_routes_added": False,
        "thresholds_relaxed": False,
        "release_ready": False,
    }
    if safety_overrides:
        safety.update(safety_overrides)
    payload = {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": delivery_status,
        "delivery_allowed": delivery_allowed,
        "presentation_status": presentation_status,
        "release_status": release_status,
        "stem_type": stem_type,
        "track_id": "track-a",
        "model_name": "htdemucs_6s",
        "selected_audio_path": str(audio_path),
        "audio_integrity_gate": {
            "status": audio_gate_status,
            "duration_sec": 120.0,
            "rms": rms,
            "peak": peak,
            "min_rms": 0.0001,
            "min_peak": 0.0001,
            "max_peak": 1.01,
        },
        "gpu_provenance_gate": {
            "status": gpu_gate_status,
            "expected_acceleration_mode": expected_acceleration_mode,
            "gpu_inference_required": gpu_inference_required,
            "gpu_inference_skipped": gpu_inference_skipped,
            "cuda_available": cuda_available,
            "resolved_device": resolved_device,
            "cpu_dsp_only_route": cpu_dsp_only_route,
        },
        "source_contract_gate": {"status": source_gate_status},
        "semantic_quality_verified": False,
        "semantic_judge_input": semantic_judge_input,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": safety,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_semantic_judge_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_stem_matrix_keeps_presemantic_preview_blocked_without_judge_pass(tmp_path: Path) -> None:
    presemantic_audio = write_audio(tmp_path / "melody.flac", amplitude=0.08)
    blocked_audio = write_audio(tmp_path / "other.flac", amplitude=0.001)
    presemantic = write_listen_manifest(
        tmp_path / "presemantic.json",
        stem_type="melody",
        audio_path=presemantic_audio,
        rms=0.08,
        peak=0.7,
    )
    blocked = write_listen_manifest(
        tmp_path / "blocked.json",
        stem_type="other",
        audio_path=blocked_audio,
        delivery_status="blocked",
        delivery_allowed=False,
        presentation_status="blocked_from_listen_pack",
        release_status="listen_preview_only",
        semantic_judge_input=False,
        audio_gate_status="blocked",
        rms=0.00001,
        peak=0.001,
    )

    matrix = build_stem_candidate_score_matrix([presemantic, blocked])
    rows = matrix["ranked_candidates"]
    by_stem = {str(row["stem_type"]): row for row in rows}

    assert by_stem["melody"]["next_action"] == "do_not_release_generate_better_stem_candidate"
    assert by_stem["melody"]["delivery_allowed"] is False
    assert by_stem["melody"]["release_ready"] is False
    assert by_stem["melody"]["semantic_quality_verified"] is False
    assert by_stem["other"]["next_action"] == "do_not_release_missing_audio_gpu_or_source_gate"
    assert matrix["summary"]["accepted_listen_preview_count"] == 0
    assert matrix["summary"]["release_ready"] is False


def test_stem_matrix_blocks_unsupported_vocal_minus_pair_manifest(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "vocals.flac")
    vocal_manifest = write_listen_manifest(tmp_path / "vocal.json", stem_type="vocal", audio_path=audio)

    matrix = build_stem_candidate_score_matrix([vocal_manifest])
    row = matrix["ranked_candidates"][0]

    assert row["stem_type"] == "vocal"
    assert row["delivery_allowed"] is False
    assert row["next_action"] == "do_not_release_unsupported_stem_type"
    assert matrix["summary"]["unsupported_stem_count"] == 1


def test_stem_matrix_requires_audio_gpu_and_source_gates(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "drums.flac")
    manifest = write_listen_manifest(
        tmp_path / "drums.json",
        stem_type="drums",
        audio_path=audio,
        gpu_gate_status="blocked",
    )

    matrix = build_stem_candidate_score_matrix([manifest])
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is False
    assert row["has_gpu_proof"] is False
    assert row["next_action"] == "do_not_release_missing_audio_gpu_or_source_gate"


def test_stem_matrix_uses_semantic_judge_to_block_candidate(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac")
    manifest = write_listen_manifest(tmp_path / "melody.json", stem_type="melody", audio_path=audio)
    semantic_judge = write_semantic_judge_jsonl(
        tmp_path / "semantic.jsonl",
        [
            {
                "stem_type": "melody",
                "selected_audio_path": str(audio),
                "decision": "blocked_hard_case",
                "stem_semantic_score": 0.2,
                "stem_semantic_blockers": ["melody_low_pitch_confidence"],
                "semantic_quality_verified": False,
            }
        ],
    )

    matrix = build_stem_candidate_score_matrix([manifest], semantic_judge_jsonl=semantic_judge)
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is False
    assert row["semantic_quality_verified"] is False
    assert row["stem_semantic_score"] == 0.2
    assert row["stem_semantic_blockers"] == ["melody_low_pitch_confidence"]
    assert row["next_action"] == "do_not_release_semantic_judge_blocked"
    assert matrix["summary"]["semantic_judge_blocked_count"] == 1


def test_stem_matrix_treats_accept_decision_with_blockers_as_semantic_blocked(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac")
    manifest = write_listen_manifest(tmp_path / "melody.json", stem_type="melody", audio_path=audio)
    semantic_judge = write_semantic_judge_jsonl(
        tmp_path / "semantic.jsonl",
        [
            {
                "stem_type": "melody",
                "selected_audio_path": str(audio),
                "decision": "accept_candidate",
                "stem_semantic_score": 0.82,
                "stem_semantic_blockers": ["melody_low_pitch_confidence"],
                "semantic_quality_verified": True,
            }
        ],
    )

    matrix = build_stem_candidate_score_matrix([manifest], semantic_judge_jsonl=semantic_judge)
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is False
    assert row["next_action"] == "do_not_release_semantic_judge_blocked"
    assert row["semantic_quality_verified"] is True
    assert row["stem_semantic_blockers"] == ["melody_low_pitch_confidence"]


def test_stem_matrix_accepts_candidate_with_semantic_judge_pass(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "bass.flac")
    manifest = write_listen_manifest(tmp_path / "bass.json", stem_type="bass", audio_path=audio)
    semantic_judge = write_semantic_judge_jsonl(
        tmp_path / "semantic.jsonl",
        [
            {
                "stem_type": "bass",
                "selected_audio_path": str(audio),
                "decision": "accept_candidate",
                "stem_semantic_score": 0.88,
                "stem_semantic_blockers": [],
                "semantic_quality_verified": True,
            }
        ],
    )

    matrix = build_stem_candidate_score_matrix([manifest], semantic_judge_jsonl=semantic_judge)
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is True
    assert row["delivery_status"] == "accepted"
    assert row["presentation_status"] == "accepted_for_listen_pack"
    assert row["release_status"] == "listen_preview_only"
    assert row["semantic_quality_verified"] is True
    assert row["stem_semantic_score"] == 0.88
    assert row["next_action"] == "accepted_for_listen_pack_not_release"


def test_stem_matrix_accepts_explicit_cpu_dsp_provenance_with_semantic_pass(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac")
    manifest = write_listen_manifest(
        tmp_path / "melody.json",
        stem_type="melody",
        audio_path=audio,
        expected_acceleration_mode="cpu_dsp_candidate_generation",
        gpu_inference_required=False,
        gpu_inference_skipped=True,
        cuda_available=False,
        resolved_device="cpu",
        cpu_dsp_only_route=True,
    )
    semantic_judge = write_semantic_judge_jsonl(
        tmp_path / "semantic.jsonl",
        [
            {
                "stem_type": "melody",
                "selected_audio_path": str(audio),
                "decision": "accept_candidate",
                "stem_semantic_score": 0.79,
                "stem_semantic_blockers": [],
                "semantic_quality_verified": True,
            }
        ],
    )

    matrix = build_stem_candidate_score_matrix([manifest], semantic_judge_jsonl=semantic_judge)
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is True
    assert row["has_gpu_proof"] is False
    assert row["has_acceleration_provenance"] is True
    assert row["semantic_quality_verified"] is True
    assert row["next_action"] == "accepted_for_listen_pack_not_release"


def test_stem_matrix_aggregates_source_safety_violations(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac")
    manifest = write_listen_manifest(
        tmp_path / "melody.json",
        stem_type="melody",
        audio_path=audio,
        safety_overrides={"training_started": True},
    )

    matrix = build_stem_candidate_score_matrix([manifest])
    row = matrix["ranked_candidates"][0]

    assert row["delivery_allowed"] is False
    assert row["source_safety_violation"] == "training_started"
    assert row["next_action"] == "do_not_release_source_safety_violation"
    assert matrix["summary"]["safety"]["training_started"] is True


def test_cli_writes_matrix_json_jsonl_summary_without_audio_export(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "bass.flac")
    listen_manifest = write_listen_manifest(tmp_path / "bass.json", stem_type="bass", audio_path=audio)
    semantic_judge = write_semantic_judge_jsonl(
        tmp_path / "semantic.jsonl",
        [
            {
                "stem_type": "bass",
                "selected_audio_path": str(audio),
                "decision": "accept_candidate",
                "stem_semantic_score": 0.9,
                "stem_semantic_blockers": [],
                "semantic_quality_verified": True,
            }
        ],
    )
    output_root = tmp_path / "matrix"

    exit_code = main(
        [
            "--listen-manifest-json",
            str(listen_manifest),
            "--semantic-judge-jsonl",
            str(semantic_judge),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    matrix_json = output_root / "no_reference_stem_candidate_score_matrix.json"
    matrix_jsonl = output_root / "no_reference_stem_candidate_score_matrix.jsonl"
    summary_md = output_root / "no_reference_stem_candidate_score_matrix_summary.md"
    assert matrix_json.exists()
    assert matrix_jsonl.exists()
    assert summary_md.exists()
    assert json.loads(matrix_json.read_text(encoding="utf-8"))["summary"]["release_ready"] is False
    assert not list(output_root.rglob("*.wav"))
    assert not list(output_root.rglob("*.flac"))


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac")
    listen_manifest = write_listen_manifest(tmp_path / "melody.json", stem_type="melody", audio_path=audio)
    output_root = tmp_path / "matrix"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_no_reference_stem_candidate_score_matrix.py",
            "--listen-manifest-json",
            str(listen_manifest),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "no_reference_stem_candidate_score_matrix.json").exists()
