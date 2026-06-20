from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.export_instrumental_resurrection_minus import parse_args as parse_ir_args
from tools.run_aris_paired_gpu_route import run_aris_paired_gpu_route


def _write_bytes(path: Path, payload: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fake_command_runner(command: list[str], *, stage: str, track_alias: str) -> None:
    del track_alias
    output_dir = Path(command[3])
    if stage == "scnet":
        _write_bytes(output_dir / "vocals.flac", b"vocal")
        _write_bytes(output_dir / "00_raw_ref.flac", b"mix")
        (output_dir / "stem_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "stems": {"vocals": {"path": str(output_dir / "vocals.flac")}},
                    "raw_reference": {"path": str(output_dir / "00_raw_ref.flac")},
                }
            ),
            encoding="utf-8",
        )
        return
    if stage == "minus":
        _write_bytes(output_dir / "minus.flac", b"minus")
        _write_bytes(output_dir / "raw_ref.flac", b"mix")
        _write_bytes(output_dir / "prog.jsonl", b"{}")
        (output_dir / "stem_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "target_stem": "minus",
                    "target_audio_path": str(output_dir / "minus.flac"),
                    "stems": {"minus": {"path": str(output_dir / "minus.flac")}},
                }
            ),
            encoding="utf-8",
        )
        return
    raise AssertionError(f"unexpected stage: {stage}")


def _accepting_judge(row: dict, *, score_mode: str) -> dict:
    base = {
        "track_id": row["track_id"],
        "candidate_id": row["candidate_id"],
        "reasons": [],
        "failed_checks": [],
    }
    if score_mode == "vocal":
        return {**base, "score_mode": "vocal", "vocal_decision": "accept_candidate", "vocal_blockers": []}
    if score_mode == "minus":
        return {
            **base,
            "score_mode": "minus",
            "minus_decision": "accept_candidate",
            "minus_blockers": [],
            "minus_musical_detail_preservation_gate": {"status": "proxy_pass", "blockers": []},
        }
    if score_mode == "both":
        return {**base, "score_mode": "both", "pair_decision": "pair_compatible", "pair_blockers": []}
    raise AssertionError(score_mode)


def _blocking_judge(row: dict, *, score_mode: str) -> dict:
    base = {
        "track_id": row["track_id"],
        "candidate_id": row["candidate_id"],
        "reasons": ["severe_music_damage"],
        "failed_checks": [],
    }
    if score_mode == "vocal":
        return {**base, "score_mode": "vocal", "vocal_decision": "accept_candidate", "vocal_blockers": []}
    if score_mode == "minus":
        return {
            **base,
            "score_mode": "minus",
            "minus_decision": "blocked_hard_case",
            "minus_blockers": ["severe_music_damage"],
            "minus_musical_detail_preservation_gate": {
                "status": "proxy_blocked",
                "blockers": [{"label": "mixture_consistency_score_below_detail_gate"}],
            },
        }
    if score_mode == "both":
        return {
            **base,
            "score_mode": "both",
            "pair_decision": "blocked_pair_incompatible",
            "pair_blockers": ["severe_mixture_reconstruction_drift"],
        }
    raise AssertionError(score_mode)


def test_rejects_output_outside_artifact_rag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be inside artifact_rag_root"):
        run_aris_paired_gpu_route(
            artifact_rag_root=tmp_path / "artifact_rag",
            corpus_dir=tmp_path / "corpus",
            output_dir=tmp_path / "outside",
            limit=1,
        )


def test_route_generates_short_rag_audio_and_passed_gate_rows(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    corpus = tmp_path / "corpus"
    _write_bytes(corpus / "raw track title.mp3", b"source")

    manifest = run_aris_paired_gpu_route(
        artifact_rag_root=rag_root,
        corpus_dir=corpus,
        output_dir=rag_root / "gpu" / "r001",
        limit=1,
        command_runner=_fake_command_runner,
        judge_runner=_accepting_judge,
    )

    assert manifest["status"] == "completed"
    assert manifest["accepted_pair_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_only_not_listen_pack"
    assert (rag_root / "gpu" / "r001" / "audio" / "t001" / "mix.flac").exists()
    assert (rag_root / "gpu" / "r001" / "audio" / "t001" / "vocal.flac").exists()
    assert (rag_root / "gpu" / "r001" / "audio" / "t001" / "minus.flac").exists()
    assert not list((rag_root / "gpu" / "r001").rglob("instrumental_resurrection_minus.flac"))
    assert all(len(part) <= 32 for path in (rag_root / "gpu" / "r001").rglob("*") for part in path.relative_to(rag_root).parts)

    wave = _read_json(Path(manifest["wave_manifest_json"]))
    jobs = _read_jsonl(Path(wave["jobs_jsonl"]))
    assert [row["stem_type"] for row in jobs] == ["vocal", "minus"]
    assert all(row["candidate_status"] == "passed" for row in jobs)
    assert all(row["no_reference_gate"] == "passed" for row in jobs)
    assert all(row["reference_gate"] == "not_available" for row in jobs)
    assert all(row["pair_consistency_gate"] == "passed" for row in jobs)
    assert all(row["binary_audio_available"] is True for row in jobs)
    assert all(row["delivery_allowed"] is False for row in jobs)

    progress = _read_jsonl(Path(manifest["progress_jsonl"]))
    gate_events = [row for row in progress if row["event"] in {"gate_start", "gate_finish"}]
    assert [row["event"] for row in gate_events] == ["gate_start", "gate_finish"]
    assert all(row["expected_acceleration_mode"] == "cpu_dsp_scoring" for row in gate_events)
    assert all(row["gpu_inference_required"] is False for row in gate_events)
    assert all(row["gpu_inference_skipped"] is True for row in gate_events)


def test_failed_no_reference_gate_never_grants_candidate_status_passed(tmp_path: Path) -> None:
    rag_root = tmp_path / "artifact_rag"
    corpus = tmp_path / "corpus"
    _write_bytes(corpus / "song.mp3", b"source")

    manifest = run_aris_paired_gpu_route(
        artifact_rag_root=rag_root,
        corpus_dir=corpus,
        output_dir=rag_root / "gpu" / "r001",
        limit=1,
        command_runner=_fake_command_runner,
        judge_runner=_blocking_judge,
    )

    assert manifest["status"] == "blocked"
    assert manifest["accepted_pair_count"] == 0
    assert manifest["delivery_allowed"] is False
    jobs = _read_jsonl(Path(_read_json(Path(manifest["wave_manifest_json"]))["jobs_jsonl"]))
    assert all(row["candidate_status"] == "failed" for row in jobs)
    assert all(row["no_reference_gate"] == "failed" for row in jobs)
    assert all(row["pair_consistency_gate"] == "failed" for row in jobs)
    assert {"severe_music_damage", "severe_mixture_reconstruction_drift"}.issubset(set(jobs[0]["failed_checks"]))


def test_instrumental_resurrection_minus_parse_args_accepts_short_rag_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_instrumental_resurrection_minus.py",
            "/app/in.mp3",
            "/app/artifact_rag/gpu/r001/work/t001/irm",
            "--minus-output-name",
            "minus.flac",
            "--progress-name",
            "prog.jsonl",
        ],
    )

    args = parse_ir_args()

    assert args.minus_output_name == "minus.flac"
    assert args.progress_name == "prog.jsonl"
