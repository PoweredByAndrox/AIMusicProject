from __future__ import annotations

import json
from pathlib import Path

from tools.plan_generic_controller_selector_candidate_wave import (
    CANDIDATE_WAVE_JOBS_NAME,
    CANDIDATE_WAVE_MANIFEST_NAME,
    CANDIDATE_WAVE_PROGRESS_NAME,
    plan_candidate_wave,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_shadow_and_gap_plan(tmp_path: Path) -> Path:
    source = tmp_path / "song.wav"
    source.write_bytes(b"fake wav bytes")
    single_dir = tmp_path / "shadow" / "single_cases" / "case-a"
    selector_jsonl = write_jsonl(
        single_dir / "generic_finetuning_controller_selector_decisions.jsonl",
        [
            {
                "track_id": "track-a",
                "selected_candidate_id": "instrumental_resurrection_minus",
                "selected_candidate": {
                    "candidate_id": "instrumental_resurrection_minus",
                    "mix_path": str(source),
                    "source_audio_path": str(source),
                },
            }
        ],
    )
    single_manifest = write_json(
        single_dir / "generic_finetuning_controller_selector_integration_manifest.json",
        {
            "outputs": {
                "selector_decisions_jsonl": str(selector_jsonl),
                "integration_decisions_jsonl": str(
                    single_dir / "generic_finetuning_controller_selector_integration_decisions.jsonl"
                ),
            }
        },
    )
    items_jsonl = write_jsonl(
        tmp_path / "shadow" / "generic_controller_selector_shadow_batch_items.jsonl",
        [{"judge_jsonl": "/app/fake/judge.jsonl", "single_manifest_json": str(single_manifest)}],
    )
    shadow_manifest = write_json(
        tmp_path / "shadow" / "generic_controller_selector_shadow_batch_manifest.json",
        {"outputs": {"items_jsonl": str(items_jsonl)}},
    )
    route_gaps = write_jsonl(
        tmp_path / "gap" / "generic_controller_selector_route_gaps.jsonl",
        [
            {
                "stem_type": "minus",
                "route_id": "htdemucs_6s",
                "recommended_action": "generate_new_candidate_wave",
                "priority": "P0",
                "next_wave_required": True,
                "blocked_count": 6,
                "missing_count": 0,
                "allowed_count": 0,
            },
            {
                "stem_type": "minus",
                "route_id": "instrumental_resurrection_minus",
                "recommended_action": "generate_new_candidate_wave",
                "priority": "P0",
                "next_wave_required": True,
                "blocked_count": 12,
                "missing_count": 0,
                "allowed_count": 0,
            },
            {
                "stem_type": "minus",
                "route_id": "mdx23c_vocals",
                "recommended_action": "collect_controller_prior_evidence",
                "priority": "P1",
                "next_wave_required": False,
                "blocked_count": 0,
                "missing_count": 2,
                "allowed_count": 0,
            },
        ],
    )
    return write_json(
        tmp_path / "gap" / "generic_controller_selector_route_gap_plan_manifest.json",
        {
            "summary": {"shadow_batch_manifest_json": str(shadow_manifest)},
            "outputs": {"route_gaps_jsonl": str(route_gaps)},
        },
    )


def test_plans_candidate_wave_jobs_from_route_gap_plan(tmp_path: Path) -> None:
    gap_manifest = build_shadow_and_gap_plan(tmp_path)

    manifest = plan_candidate_wave(
        route_gap_plan_manifest_json=gap_manifest,
        output_dir=tmp_path / "candidate_wave",
        device="cuda",
        resurrection_device="cuda:0",
        max_sources_per_route=1,
    )

    jobs = read_jsonl(tmp_path / "candidate_wave" / CANDIDATE_WAVE_JOBS_NAME)
    by_route = {row["route_id"]: row for row in jobs}

    assert by_route["htdemucs_6s"]["execution_backend"] == "registered_model_candidate"
    assert by_route["htdemucs_6s"]["job_status"] == "planned"
    assert by_route["htdemucs_6s"]["target_stem"] == "minus"
    assert by_route["htdemucs_6s"]["command"][:2] == ["python", "/app/tools/run_registered_model_candidate.py"]
    assert "--model-id" in by_route["htdemucs_6s"]["command"]
    assert "htdemucs_6s" in by_route["htdemucs_6s"]["command"]
    assert "--device" in by_route["htdemucs_6s"]["command"]
    assert "cuda" in by_route["htdemucs_6s"]["command"]

    assert by_route["instrumental_resurrection_minus"]["execution_backend"] == "direct_instrumental_resurrection_export"
    assert by_route["instrumental_resurrection_minus"]["job_status"] == "planned"
    assert by_route["instrumental_resurrection_minus"]["command"][:2] == [
        "python",
        "/app/tools/export_instrumental_resurrection_minus.py",
    ]
    assert "cuda:0" in by_route["instrumental_resurrection_minus"]["command"]

    assert by_route["mdx23c_vocals"]["execution_backend"] == "registered_model_candidate"
    assert by_route["mdx23c_vocals"]["wave_intent"] == "collect_controller_prior_evidence"
    assert by_route["mdx23c_vocals"]["job_status"] == "planned"

    assert manifest["summary"]["planned_job_count"] == 3
    assert manifest["summary"]["blocked_job_count"] == 0
    assert manifest["promotion"]["status"] == "candidate_wave_plan_ready"
    assert manifest["promotion"]["can_start_training"] is False
    assert manifest["promotion"]["can_execute_audio_inference"] is False
    assert manifest["release_ready"] is False

    disk_manifest = read_json(tmp_path / "candidate_wave" / CANDIDATE_WAVE_MANIFEST_NAME)
    assert disk_manifest["fingerprint"] == manifest["fingerprint"]
    assert [row["event"] for row in read_jsonl(tmp_path / "candidate_wave" / CANDIDATE_WAVE_PROGRESS_NAME)] == [
        "start",
        "write_outputs",
        "finish",
    ]
