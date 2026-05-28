from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "drum_audit_guided_residual_debleed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("drum_audit_guided_residual_debleed", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_audit_guided_debleed_reduces_reference_melody_residual(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    frame_count = sample_rate * 2
    time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    drums = np.zeros(frame_count, dtype=np.float32)
    transient = np.hanning(128).astype(np.float32)
    transient_start = int(0.900 * sample_rate)
    drums[transient_start : transient_start + len(transient)] = 0.8 * transient
    melody = np.zeros(frame_count, dtype=np.float32)
    leak_start_sec = 0.920
    leak_end_sec = 1.080
    leak_start = int(leak_start_sec * sample_rate)
    leak_end = int(leak_end_sec * sample_rate)
    melody[leak_start:leak_end] = (
        0.35 * np.sin(2.0 * np.pi * 660.0 * time[: leak_end - leak_start])
    ).astype(np.float32)
    candidate = drums + (0.25 * melody)
    audit_report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "worst_windows_by_time_bucket": {
                        "0.000-10.000": [
                            {
                                "start_sec": leak_start_sec,
                                "end_sec": leak_end_sec,
                                "issue_labels": ["micro_residual_reference_stem_leakage"],
                                "metrics": {
                                    "max_residual_leakage_source": "melody",
                                    "max_residual_leakage_strength": 0.90,
                                    "max_leakage_source": "melody",
                                    "max_leakage_strength": 0.80,
                                },
                            }
                        ]
                    }
                },
            }
        ]
    }
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    report_path = tmp_path / "report.json"
    _write_audio(candidate_path, _stereo(candidate), sample_rate)
    _write_audio(drums_path, _stereo(drums), sample_rate)
    _write_audio(melody_path, _stereo(melody), sample_rate)
    report_path.write_text(json.dumps(audit_report), encoding="utf-8")

    manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path},
        output_dir=tmp_path / "out",
        listen_dir=tmp_path / "listen",
        output_label="02_out.flac",
        input_label="01_in.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            repair_strength=1.0,
            max_projection_gain=0.9,
            min_residual_leakage_strength=0.01,
            min_candidate_leakage_strength=0.01,
            context_ms=0.0,
        ),
    )

    output_audio, _ = module.read_audio(Path(manifest["stems"]["drums"]["path"]))
    input_residual = _stereo(candidate - drums)
    output_residual = output_audio - _stereo(drums)
    assert float(np.sqrt(np.mean(np.square(output_residual)))) < 0.25 * float(
        np.sqrt(np.mean(np.square(input_residual)))
    )
    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["application_reference_free"] is False
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert manifest["diagnostics"]["applied_window_count"] == 1


def test_audit_guided_debleed_ignores_below_floor_windows(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "worst_windows": [
                        {
                            "start_sec": 0.0,
                            "end_sec": 0.1,
                            "issue_labels": ["micro_residual_reference_stem_leakage"],
                            "metrics": {
                                "max_residual_leakage_source": "melody",
                                "max_residual_leakage_strength": 0.001,
                            },
                        }
                    ]
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(min_residual_leakage_strength=0.01),
    )

    assert windows == []


def test_audit_guided_debleed_selects_phrase_drop_melody_windows(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows": [
                        {
                            "start_sec": 1.0,
                            "end_sec": 3.0,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "melody",
                                "phrase_leakage_strength": 0.12,
                            },
                        }
                    ]
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(min_phrase_leakage_strength=0.05),
    )

    assert len(windows) == 1
    assert windows[0]["repair_source"] == "melody"
    assert windows[0]["repair_source_reason"] == "phrase_reference_projection"


def test_audit_guided_debleed_selects_phrase_windows_from_time_buckets(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows_by_time_bucket": {
                        "80.000-90.000": [
                            {
                                "start_sec": 82.5,
                                "end_sec": 83.5,
                                "issue_labels": ["bar_spanning_drop_melody_leakage"],
                                "metrics": {
                                    "phrase_leakage_source": "bass",
                                    "phrase_leakage_strength": 0.14,
                                },
                            }
                        ]
                    }
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(min_phrase_leakage_strength=0.05),
    )

    assert len(windows) == 1
    assert windows[0]["start_sec"] == 82.5
    assert windows[0]["repair_source"] == "bass"


def test_audit_guided_debleed_can_filter_to_phrase_repairs(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "worst_windows": [
                        {
                            "start_sec": 1.0,
                            "end_sec": 1.5,
                            "issue_labels": ["micro_residual_reference_stem_leakage"],
                            "metrics": {
                                "max_residual_leakage_source": "vocals",
                                "max_residual_leakage_strength": 0.25,
                            },
                        }
                    ],
                    "phrase_level_windows": [
                        {
                            "start_sec": 2.0,
                            "end_sec": 4.0,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "melody",
                                "phrase_leakage_strength": 0.12,
                            },
                        }
                    ],
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            min_phrase_leakage_strength=0.05,
            allowed_repair_reasons=("phrase_reference_projection",),
        ),
    )

    assert len(windows) == 1
    assert windows[0]["repair_source"] == "melody"
    assert windows[0]["repair_source_reason"] == "phrase_reference_projection"


def test_audit_guided_debleed_can_filter_repair_sources(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows": [
                        {
                            "start_sec": 1.0,
                            "end_sec": 3.0,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "melody",
                                "phrase_leakage_strength": 0.12,
                            },
                        },
                        {
                            "start_sec": 3.0,
                            "end_sec": 5.0,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "bass",
                                "phrase_leakage_strength": 0.18,
                            },
                        },
                    ]
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            min_phrase_leakage_strength=0.05,
            allowed_repair_sources=("melody",),
        ),
    )

    assert len(windows) == 1
    assert windows[0]["repair_source"] == "melody"


def test_audit_guided_debleed_rejects_target_stem_as_leakage_source(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    report_path = tmp_path / "report.json"
    _write_audio(candidate_path, audio, sample_rate)
    _write_audio(drums_path, audio, sample_rate)
    report_path.write_text(
        json.dumps(
            {
                "stems": [
                    {
                        "stem": "drums",
                        "micro_quality_windows": {"phrase_level_windows": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not include target stem"):
        module.build_audit_guided_residual_debleed(
            candidate_drum_path=candidate_path,
            target_reference_path=drums_path,
            audit_report_path=report_path,
            leakage_source_paths={"drums": drums_path},
            output_dir=tmp_path / "out",
            listen_dir=None,
            output_label="out.flac",
            input_label="input.flac",
            stem_name="drums",
            config=module.AuditGuidedDebleedConfig(),
        )


def test_audit_guided_debleed_uses_phrase_projection_gain_multiplier(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    frame_count = sample_rate * 3
    time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    drums = np.zeros(frame_count, dtype=np.float32)
    melody = (0.25 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    candidate = drums + (0.05 * melody)
    audit_report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows": [
                        {
                            "start_sec": 0.5,
                            "end_sec": 2.5,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "melody",
                                "phrase_leakage_strength": 0.20,
                            },
                        }
                    ]
                },
            }
        ]
    }
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    report_path = tmp_path / "report.json"
    _write_audio(candidate_path, _stereo(candidate), sample_rate)
    _write_audio(drums_path, _stereo(drums), sample_rate)
    _write_audio(melody_path, _stereo(melody), sample_rate)
    report_path.write_text(json.dumps(audit_report), encoding="utf-8")

    base_manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path},
        output_dir=tmp_path / "base",
        listen_dir=None,
        output_label="base.flac",
        input_label="input.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            repair_strength=0.5,
            max_projection_gain=0.9,
            min_phrase_leakage_strength=0.05,
            context_ms=0.0,
            phrase_projection_gain_multiplier=1.0,
        ),
    )
    boosted_manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path},
        output_dir=tmp_path / "boosted",
        listen_dir=None,
        output_label="boosted.flac",
        input_label="input.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            repair_strength=0.5,
            max_projection_gain=0.9,
            min_phrase_leakage_strength=0.05,
            context_ms=0.0,
            phrase_projection_gain_multiplier=2.0,
        ),
    )

    base_audio, _ = module.read_audio(Path(base_manifest["stems"]["drums"]["path"]))
    boosted_audio, _ = module.read_audio(Path(boosted_manifest["stems"]["drums"]["path"]))
    base_residual = base_audio - _stereo(drums)
    boosted_residual = boosted_audio - _stereo(drums)
    assert float(np.sqrt(np.mean(np.square(boosted_residual)))) < float(
        np.sqrt(np.mean(np.square(base_residual)))
    )


def test_audit_guided_debleed_multi_source_phrase_projection_reduces_combined_residual(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    frame_count = sample_rate * 3
    time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    drums = np.zeros(frame_count, dtype=np.float32)
    melody = (0.25 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    bass = (0.30 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
    candidate = drums + (0.05 * melody) + (0.06 * bass)
    audit_report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows": [
                        {
                            "start_sec": 0.5,
                            "end_sec": 2.5,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "melody",
                                "phrase_leakage_strength": 0.20,
                            },
                        }
                    ]
                },
            }
        ]
    }
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    bass_path = tmp_path / "bass.flac"
    report_path = tmp_path / "report.json"
    _write_audio(candidate_path, _stereo(candidate), sample_rate)
    _write_audio(drums_path, _stereo(drums), sample_rate)
    _write_audio(melody_path, _stereo(melody), sample_rate)
    _write_audio(bass_path, _stereo(bass), sample_rate)
    report_path.write_text(json.dumps(audit_report), encoding="utf-8")

    single_source_manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path, "bass": bass_path},
        output_dir=tmp_path / "single",
        listen_dir=None,
        output_label="single.flac",
        input_label="input.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            repair_strength=1.0,
            max_projection_gain=0.9,
            min_phrase_leakage_strength=0.05,
            context_ms=0.0,
            allowed_repair_reasons=("phrase_reference_projection",),
            allowed_repair_sources=("melody", "bass"),
        ),
    )
    multi_source_manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path, "bass": bass_path},
        output_dir=tmp_path / "multi",
        listen_dir=None,
        output_label="multi.flac",
        input_label="input.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(
            repair_strength=1.0,
            max_projection_gain=0.9,
            min_phrase_leakage_strength=0.05,
            context_ms=0.0,
            phrase_multi_source_projection=True,
            allowed_repair_reasons=("phrase_reference_projection",),
            allowed_repair_sources=("melody", "bass"),
        ),
    )

    single_audio, _ = module.read_audio(Path(single_source_manifest["stems"]["drums"]["path"]))
    multi_audio, _ = module.read_audio(Path(multi_source_manifest["stems"]["drums"]["path"]))
    single_residual = single_audio - _stereo(drums)
    multi_residual = multi_audio - _stereo(drums)
    assert float(np.sqrt(np.mean(np.square(multi_residual)))) < float(
        np.sqrt(np.mean(np.square(single_residual)))
    )
    first_applied_window = multi_source_manifest["diagnostics"]["applied_windows"][0]
    assert set(first_applied_window["multi_source_projection_gains"]) == {"melody", "bass"}


def test_multi_source_projection_correction_limits_total_energy() -> None:
    module = _load_module()
    frame_count = 1024
    source = np.ones((frame_count, 2), dtype=np.float64)
    residual = source

    correction, gains = module.multi_source_projection_correction(
        residual_segment=residual,
        source_segments={
            "melody": source,
            "bass": source,
            "fx": source,
        },
        config=module.AuditGuidedDebleedConfig(
            repair_strength=1.0,
            max_projection_gain=1.0,
            max_multi_source_correction_ratio=0.75,
            phrase_projection_gain_multiplier=1.0,
            phrase_multi_source_projection=True,
        ),
    )

    residual_energy = float(np.linalg.norm(residual))
    correction_energy = float(np.linalg.norm(correction))
    assert correction_energy <= residual_energy * 0.75 + 1.0e-9
    assert set(gains) == {"melody", "bass", "fx"}


def test_audit_guided_debleed_manifest_records_provenance(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8_000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    report_path = tmp_path / "report.json"
    _write_audio(candidate_path, audio, sample_rate)
    _write_audio(drums_path, audio, sample_rate)
    _write_audio(melody_path, audio, sample_rate)
    report_path.write_text(
        json.dumps(
            {
                "stems": [
                    {
                        "stem": "drums",
                        "micro_quality_windows": {"phrase_level_windows": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = module.build_audit_guided_residual_debleed(
        candidate_drum_path=candidate_path,
        target_reference_path=drums_path,
        audit_report_path=report_path,
        leakage_source_paths={"melody": melody_path},
        output_dir=tmp_path / "out",
        listen_dir=None,
        output_label="out.flac",
        input_label="input.flac",
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(),
        command_args=["--allowed-repair-source", "melody"],
    )

    assert manifest["provenance"]["tool_path"].endswith("drum_audit_guided_residual_debleed.py")
    assert manifest["provenance"]["command_args"] == ["--allowed-repair-source", "melody"]


def test_parse_args_accepts_phrase_multi_source_filters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    candidate_path = tmp_path / "candidate.flac"
    output_dir = tmp_path / "out"
    target_path = tmp_path / "drums.flac"
    report_path = tmp_path / "report.json"
    melody_path = tmp_path / "melody.flac"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drum_audit_guided_residual_debleed.py",
            str(candidate_path),
            str(output_dir),
            "--target-reference-path",
            str(target_path),
            "--audit-report-path",
            str(report_path),
            "--leakage-source",
            f"melody={melody_path}",
            "--phrase-multi-source-projection",
            "--max-multi-source-correction-ratio",
            "0.75",
            "--allowed-repair-reason",
            "phrase_reference_projection",
            "--allowed-repair-source",
            "melody",
        ],
    )

    args = module.parse_args()

    assert args.phrase_multi_source_projection is True
    assert args.max_multi_source_correction_ratio == 0.75
    assert args.allowed_repair_reason == ["phrase_reference_projection"]
    assert args.allowed_repair_source == ["melody"]


def test_cli_smoke_runs_phrase_multi_source_route(tmp_path: Path) -> None:
    sample_rate = 8_000
    frame_count = sample_rate
    time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    drums = np.zeros(frame_count, dtype=np.float32)
    melody = (0.2 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    candidate = drums + (0.05 * melody)
    candidate_path = tmp_path / "candidate.flac"
    drums_path = tmp_path / "drums.flac"
    melody_path = tmp_path / "melody.flac"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "out"
    _write_audio(candidate_path, _stereo(candidate), sample_rate)
    _write_audio(drums_path, _stereo(drums), sample_rate)
    _write_audio(melody_path, _stereo(melody), sample_rate)
    report_path.write_text(
        json.dumps(
            {
                "stems": [
                    {
                        "stem": "drums",
                        "micro_quality_windows": {
                            "phrase_level_windows": [
                                {
                                    "start_sec": 0.0,
                                    "end_sec": 1.0,
                                    "issue_labels": ["bar_spanning_drop_melody_leakage"],
                                    "metrics": {
                                        "phrase_leakage_source": "melody",
                                        "phrase_leakage_strength": 0.20,
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            str(candidate_path),
            str(output_dir),
            "--target-reference-path",
            str(drums_path),
            "--audit-report-path",
            str(report_path),
            "--leakage-source",
            f"melody={melody_path}",
            "--output-label",
            "out.flac",
            "--phrase-multi-source-projection",
            "--max-multi-source-correction-ratio",
            "0.75",
            "--allowed-repair-reason",
            "phrase_reference_projection",
            "--allowed-repair-source",
            "melody",
            "--context-ms",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_payload = json.loads(completed.stdout)
    manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

    assert Path(stdout_payload["output"]["path"]).exists()
    assert manifest["diagnostics"]["applied_window_count"] == 1
    assert manifest["provenance"]["tool_path"].endswith("drum_audit_guided_residual_debleed.py")
    assert "--phrase-multi-source-projection" in manifest["provenance"]["command_args"]
    assert manifest["config"]["max_multi_source_correction_ratio"] == 0.75


def test_audit_guided_debleed_ignores_unavailable_unattributed_phrase_source(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "stems": [
            {
                "stem": "drums",
                "micro_quality_windows": {
                    "phrase_level_windows": [
                        {
                            "start_sec": 1.0,
                            "end_sec": 3.0,
                            "issue_labels": ["bar_spanning_drop_melody_leakage"],
                            "metrics": {
                                "phrase_leakage_source": "unattributed_harmonic_residual",
                                "phrase_leakage_strength": 0.50,
                            },
                        }
                    ]
                },
            }
        ]
    }

    windows = module.iter_audit_windows(
        report,
        stem_name="drums",
        config=module.AuditGuidedDebleedConfig(min_phrase_leakage_strength=0.05),
    )

    assert windows == []
