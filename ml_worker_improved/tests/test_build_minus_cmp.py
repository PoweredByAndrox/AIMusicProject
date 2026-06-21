from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_minus_cmp import build_minus_cmp


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_audio(path: Path, *, freq: float = 220.0, duration_sec: float = 2.0) -> Path:
    sample_rate = 16_000
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * freq * t) * 0.03).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _mid_pack(tmp_path: Path) -> Path:
    root = tmp_path / "mid"
    _write_audio(root / "audio" / "old" / "w001.flac")
    return _write_json(
        root / "pack.json",
        {
            "schema": "minus_mid_audition_pack.v1",
            "pack_status": "ready_for_diagnostic_listen",
            "release_ready": False,
            "can_publish_as_result": False,
            "failed_checks": [],
            "items": [
                {
                    "audio_path": "audio/old/w001.flac",
                    "source_alias": "old",
                    "source_audio_path": str(root / "audio" / "old" / "w001.flac"),
                    "stem_type": "minus",
                    "window_id": "w001",
                    "window_start_sec": 0.0,
                    "window_duration_sec": 1.0,
                    "window_reason": "middle_bleed",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )


def _selection_pack(tmp_path: Path) -> Path:
    root = tmp_path / "rs"
    candidate = _write_audio(root / "audio" / "min" / "m001.flac", freq=330.0)
    return _write_json(
        root / "pack.json",
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "selected_count": 1,
            "failed_checks": [],
            "items": [
                {
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(candidate),
                    "source_candidate_id": "q001_c001",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {},
                }
            ],
        },
    )


def _selection_pack_listen_audio_files(tmp_path: Path) -> Path:
    root = tmp_path / "rs"
    candidate = _write_audio(root / "audio" / "min" / "m001.flac", freq=330.0)
    return _write_json(
        root / "pack.json",
        {
            "schema": "short_listen_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "selected_count": 1,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(candidate),
                    "source_candidate_id": "q001_c001",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {},
                }
            ],
        },
    )


def _candidate_wave_pack(tmp_path: Path) -> Path:
    root = tmp_path / "cw"
    candidate = _write_audio(root / "audio" / "min" / "m001.flac", freq=330.0)
    return _write_json(
        root / "pack.json",
        {
            "schema": "short_candidate_wave_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "selected_count": 1,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(candidate),
                    "source_candidate_id": "cw001",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {},
                }
            ],
        },
    )


def _phase_repair_pack(tmp_path: Path) -> Path:
    root = tmp_path / "mpr"
    candidate = _write_audio(root / "audio" / "min" / "m001.flac", freq=330.0)
    return _write_json(
        root / "pack.json",
        {
            "schema": "short_minus_phase_repair_selection_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "release_ready": False,
            "can_publish_as_result": False,
            "selected_count": 1,
            "failed_checks": [],
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/min/m001.flac",
                    "listen_audio_path": str(candidate),
                    "source_candidate_id": "mpr001",
                    "stem_type": "minus",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_row": {},
                }
            ],
        },
    )


def test_build_minus_cmp_writes_diagnostic_comparison(monkeypatch, tmp_path: Path) -> None:
    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)
    calls: list[tuple[str, float, float]] = []

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        calls.append((Path(minus_path).name, start_sec, duration_sec))
        leak = 0.30 if "old" in minus_path else 0.10
        return {
            "minus_score": 0.70 if "old" in minus_path else 0.88,
            "vocal_leakage_in_minus_score": leak,
            "mixture_consistency_score": 0.74 if "old" in minus_path else 0.82,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=_mid_pack(tmp_path),
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
        command_args=["--example"],
    )

    assert manifest["schema"] == "minus_middle_comparison_report.v1"
    assert manifest["comparison_status"] == "completed"
    assert manifest["row_count"] == 1
    assert manifest["improved_count"] == 1
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["failed_checks"] == []
    assert calls == [("w001.flac", 0.0, 1.0), ("m001.flac", 0.0, 1.0)]

    rows = [json.loads(line) for line in (tmp_path / "cmp" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["baseline_audio_path"] == "audio/old/w001.flac"
    assert rows[0]["candidate_audio_path"] == "audio/min/m001.flac"
    assert rows[0]["vocal_leakage_delta"] == -0.2
    assert rows[0]["minus_score_delta"] == 0.18
    assert rows[0]["comparison_decision"] == "candidate_proxy_improved"
    assert all(len(path.name) <= 32 for path in (tmp_path / "cmp").iterdir() if path.is_file())


def test_build_minus_cmp_reads_real_selection_schema_and_source_audio(monkeypatch, tmp_path: Path) -> None:
    mid_pack = _mid_pack(tmp_path)
    payload = json.loads(mid_pack.read_text(encoding="utf-8"))
    source_audio = _write_audio(tmp_path / "old_full.flac", freq=210.0)
    payload["items"][0]["source_audio_path"] = str(source_audio)
    payload["items"][0]["window_start_sec"] = 48.0
    mid_pack.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)
    calls: list[tuple[str, float, float]] = []

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        calls.append((Path(minus_path).name, start_sec, duration_sec))
        return {
            "minus_score": 0.70 if "old_full" in minus_path else 0.88,
            "vocal_leakage_in_minus_score": 0.30 if "old_full" in minus_path else 0.10,
            "mixture_consistency_score": 0.74 if "old_full" in minus_path else 0.82,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=mid_pack,
        selection_pack_json=_selection_pack_listen_audio_files(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "completed"
    assert calls == [("old_full.flac", 48.0, 1.0), ("m001.flac", 48.0, 1.0)]


def test_build_minus_cmp_accepts_candidate_wave_selection_schema(monkeypatch, tmp_path: Path) -> None:
    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        return {
            "minus_score": 0.70 if "old" in minus_path else 0.88,
            "vocal_leakage_in_minus_score": 0.30 if "old" in minus_path else 0.10,
            "mixture_consistency_score": 0.74 if "old" in minus_path else 0.82,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=_mid_pack(tmp_path),
        selection_pack_json=_candidate_wave_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "completed"
    assert manifest["failed_checks"] == []
    rows = [json.loads(line) for line in (tmp_path / "cmp" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["candidate_id"] == "cw001"


def test_build_minus_cmp_accepts_phase_repair_selection_schema(monkeypatch, tmp_path: Path) -> None:
    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        return {
            "minus_score": 0.70 if "old" in minus_path else 0.88,
            "vocal_leakage_in_minus_score": 0.30 if "old" in minus_path else 0.10,
            "mixture_consistency_score": 0.74 if "old" in minus_path else 0.82,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=_mid_pack(tmp_path),
        selection_pack_json=_phase_repair_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "completed"
    assert manifest["failed_checks"] == []
    rows = [json.loads(line) for line in (tmp_path / "cmp" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["candidate_id"] == "mpr001"


def test_build_minus_cmp_counts_detail_regressed_rows(monkeypatch, tmp_path: Path) -> None:
    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        return {
            "minus_score": 0.80 if "old" in minus_path else 0.83,
            "vocal_leakage_in_minus_score": 0.30 if "old" in minus_path else 0.29,
            "mixture_consistency_score": 0.86 if "old" in minus_path else 0.70,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=_mid_pack(tmp_path),
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "completed"
    assert manifest["improved_count"] == 0
    assert manifest["regressed_count"] == 1
    assert manifest["inconclusive_count"] == 0
    rows = [json.loads(line) for line in (tmp_path / "cmp" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["comparison_decision"] == "candidate_proxy_detail_regressed"


def test_build_minus_cmp_uses_overall_no_reference_score(monkeypatch, tmp_path: Path) -> None:
    mix = _write_audio(tmp_path / "mix.flac", freq=440.0)
    vocal = _write_audio(tmp_path / "vocal.flac", freq=550.0)

    def fake_window_metrics(*, mix_path: str, vocal_path: str, minus_path: str, start_sec: float, duration_sec: float) -> dict:
        return {
            "overall_no_reference_score": 0.70 if "old" in minus_path else 0.88,
            "vocal_leakage_in_minus_score": 0.30 if "old" in minus_path else 0.10,
            "mixture_consistency_score": 0.74 if "old" in minus_path else 0.82,
            "minus_blockers": [],
            "reasons": ["no_reference_proxy_passed"],
        }

    monkeypatch.setattr("tools.build_minus_cmp.window_metrics", fake_window_metrics)

    manifest = build_minus_cmp(
        mid_pack_json=_mid_pack(tmp_path),
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=mix,
        vocal_path=vocal,
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "completed"
    assert manifest["improved_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "cmp" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["minus_score_delta"] == 0.18


def test_build_minus_cmp_blocks_release_ready_mid_pack(tmp_path: Path) -> None:
    mid_pack = _mid_pack(tmp_path)
    payload = json.loads(mid_pack.read_text(encoding="utf-8"))
    payload["release_ready"] = True
    mid_pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = build_minus_cmp(
        mid_pack_json=mid_pack,
        selection_pack_json=_selection_pack(tmp_path),
        output_dir=tmp_path / "cmp",
        mix_path=tmp_path / "mix.flac",
        vocal_path=tmp_path / "vocal.flac",
        baseline_alias="old",
    )

    assert manifest["comparison_status"] == "blocked"
    assert "mid_pack_release_ready_true" in manifest["failed_checks"]
