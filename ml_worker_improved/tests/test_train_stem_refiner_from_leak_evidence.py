from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.stem_region_repair_model import StemRegionRepairTrainConfig
from tools.stem_contamination_model import band_limited_audio
from tools import train_stem_refiner_from_leak_evidence as leak_train
from tools.train_stem_refiner_from_leak_evidence import train_from_leak_evidence


def _write(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate)
    return path


def test_leak_evidence_training_changes_weights_without_ground_truth(tmp_path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 2
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clean = np.stack(
        [
            0.04 * np.sin(2.0 * np.pi * 110.0 * t),
            0.035 * np.sin(2.0 * np.pi * 113.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    leak = np.stack(
        [
            0.03 * np.sin(2.0 * np.pi * 830.0 * t),
            0.03 * np.sin(2.0 * np.pi * 850.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad = clean + leak
    window_dir = tmp_path / "w000"
    candidate_dir = window_dir / "c101"
    _write(window_dir / "mix.flac", bad)
    _write(candidate_dir / "stems" / "bass.flac", bad)
    _write(candidate_dir / "score" / "leaks" / "bass_vocals.flac", leak)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "c101",
                "stem_paths": {"bass": str(candidate_dir / "stems" / "bass.flac")},
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "score.json").write_text(
        json.dumps(
            {
                "contamination_score": {
                    "leak_results": [
                        {
                            "tested_stem": "bass",
                            "foreign_source": "vocals",
                            "verdict": "fail",
                            "leak_audio": "leaks/bass_vocals.flac",
                            "leak_dbfs": -30.0,
                            "regions": [{"start": 0.0, "end": frames / sample_rate}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "train",
        stem="bass",
        issue_type="vocal_leak",
        foreign_sources=["vocals"],
        leak_gain=1.0,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            device="cpu",
        ),
    )

    assert manifest["is_ground_truth"] is False
    assert manifest["label_type"] == "pseudo_teacher"
    assert manifest["delivery_allowed"] is False
    assert manifest["usable_for_promotion"] is False
    assert manifest["weights_changed"] is True
    assert manifest["target_issue_improved"] is True
    assert manifest["target_region_mse_after"] < manifest["target_region_mse_before"]
    assert Path(manifest["checkpoint"]).is_file()


def test_leak_evidence_batch_training_uses_multiple_candidates(tmp_path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 2
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)

    candidate_dirs: list[Path] = []
    for index, leak_hz in enumerate((830.0, 960.0)):
        clean = np.stack(
            [
                0.04 * np.sin(2.0 * np.pi * 110.0 * t),
                0.035 * np.sin(2.0 * np.pi * 113.0 * t),
            ],
            axis=1,
        ).astype(np.float32)
        leak = np.stack(
            [
                0.03 * np.sin(2.0 * np.pi * leak_hz * t),
                0.03 * np.sin(2.0 * np.pi * (leak_hz + 20.0) * t),
            ],
            axis=1,
        ).astype(np.float32)
        bad = clean + leak
        window_dir = tmp_path / f"w{index:03d}"
        candidate_dir = window_dir / "c101"
        candidate_dirs.append(candidate_dir)
        _write(window_dir / "mix.flac", bad)
        _write(candidate_dir / "stems" / "bass.flac", bad)
        _write(candidate_dir / "score" / "leaks" / "bass_vocals.flac", leak)
        (candidate_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": "c101",
                    "stem_paths": {"bass": str(candidate_dir / "stems" / "bass.flac")},
                }
            ),
            encoding="utf-8",
        )
        (candidate_dir / "score.json").write_text(
            json.dumps(
                {
                    "contamination_score": {
                        "leak_results": [
                            {
                                "tested_stem": "bass",
                                "foreign_source": "vocals",
                                "verdict": "fail",
                                "leak_audio": "leaks/bass_vocals.flac",
                                "leak_dbfs": -30.0,
                                "regions": [{"start": 0.0, "end": frames / sample_rate}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dirs[0],
        candidate_dirs=[candidate_dirs[1]],
        output_dir=tmp_path / "train_batch",
        stem="bass",
        issue_type="vocal_leak",
        foreign_sources=["vocals"],
        leak_gain=1.0,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            device="cpu",
        ),
    )

    assert manifest["is_ground_truth"] is False
    assert manifest["usable_for_promotion"] is False
    assert manifest["training_example_count"] == 2
    assert len(manifest["candidate_dirs"]) == 2
    assert manifest["weights_changed"] is True
    assert manifest["target_issue_improved"] is True
    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))
    assert history["training_example_count"] == 2
    assert [row["row_id"] for row in history["training_rows"]] == ["leak_ev_000001", "leak_ev_000002"]


def test_leak_evidence_training_records_hard_to_soft_curriculum(tmp_path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 2
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clean = np.stack(
        [
            0.04 * np.sin(2.0 * np.pi * 110.0 * t),
            0.035 * np.sin(2.0 * np.pi * 113.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    leak = np.stack(
        [
            0.03 * np.sin(2.0 * np.pi * 830.0 * t),
            0.03 * np.sin(2.0 * np.pi * 850.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad = clean + leak
    window_dir = tmp_path / "w000"
    candidate_dir = window_dir / "c101"
    _write(window_dir / "mix.flac", bad)
    _write(candidate_dir / "stems" / "bass.flac", bad)
    _write(candidate_dir / "score" / "leaks" / "bass_vocals.flac", leak)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "c101",
                "stem_paths": {"bass": str(candidate_dir / "stems" / "bass.flac")},
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "score.json").write_text(
        json.dumps(
            {
                "contamination_score": {
                    "leak_results": [
                        {
                            "tested_stem": "bass",
                            "foreign_source": "vocals",
                            "verdict": "fail",
                            "leak_audio": "leaks/bass_vocals.flac",
                            "leak_dbfs": -30.0,
                            "regions": [{"start": 0.0, "end": frames / sample_rate}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "train_curriculum",
        stem="bass",
        issue_type="vocal_leak",
        foreign_sources=["vocals"],
        leak_gain=1.0,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            curriculum_target_boost=6.0,
            curriculum_hard_fraction=0.75,
            curriculum_hard_loss_weight=5.0,
            max_delta=0.9,
            device="cpu",
        ),
    )

    assert manifest["weights_changed"] is True
    assert manifest["target_issue_improved"] is True
    assert manifest["train_config"]["curriculum_target_boost"] == 6.0
    assert manifest["train_config"]["curriculum_hard_fraction"] == 0.75
    assert manifest["train_config"]["curriculum_hard_loss_weight"] == 5.0
    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))
    assert history["config"]["curriculum_target_boost"] == 6.0
    assert history["config"]["curriculum_hard_fraction"] == 0.75
    assert history["config"]["curriculum_hard_loss_weight"] == 5.0


def test_non_vocal_leak_training_does_not_auto_compensate_into_foreign_stem(tmp_path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 2
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    bass = np.stack(
        [
            0.04 * np.sin(2.0 * np.pi * 110.0 * t),
            0.035 * np.sin(2.0 * np.pi * 113.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    vocal_leak = np.stack(
        [
            0.03 * np.sin(2.0 * np.pi * 830.0 * t),
            0.03 * np.sin(2.0 * np.pi * 850.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    vocals = np.stack(
        [
            0.05 * np.sin(2.0 * np.pi * 720.0 * t),
            0.045 * np.sin(2.0 * np.pi * 725.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_bass = bass + vocal_leak
    window_dir = tmp_path / "w000"
    candidate_dir = window_dir / "c101"
    _write(window_dir / "mix.flac", bad_bass + vocals)
    _write(candidate_dir / "stems" / "bass.flac", bad_bass)
    _write(candidate_dir / "stems" / "vocals.flac", vocals)
    _write(candidate_dir / "score" / "leaks" / "bass_vocals.flac", vocal_leak)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "c101",
                "stem_paths": {
                    "bass": str(candidate_dir / "stems" / "bass.flac"),
                    "vocals": str(candidate_dir / "stems" / "vocals.flac"),
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "score.json").write_text(
        json.dumps(
            {
                "contamination_score": {
                    "leak_results": [
                        {
                            "tested_stem": "bass",
                            "foreign_source": "vocals",
                            "verdict": "fail",
                            "leak_audio": "leaks/bass_vocals.flac",
                            "leak_dbfs": -30.0,
                            "regions": [{"start": 0.0, "end": frames / sample_rate}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "train",
        stem="bass",
        issue_type="vocal_leak",
        foreign_sources=["vocals"],
        leak_gain=1.0,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )

    assert manifest["target_issue_improved"] is True
    assert manifest["mix_preserve_companion_stem"] == ""
    assert manifest["forensic_source_conditioning"] is True
    assert manifest["forensic_source_audio_path"].endswith("vocals.flac")
    assert manifest["diagnostics"]["source_conditioned"] is True
    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))
    assert "mix_preserve_companion_stem" not in history
    assert history["forensic_source_condition_stem"] == "vocals"

    transfer_manifest = train_from_leak_evidence(
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "train_transfer",
        stem="bass",
        issue_type="vocal_leak",
        foreign_sources=["vocals"],
        leak_gain=1.0,
        transfer_leak_to_foreign_source=True,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )

    assert transfer_manifest["target_issue_improved"] is True
    assert transfer_manifest["mix_preserve_companion_stem"] == "vocals"
    assert transfer_manifest["leak_transfer_to_foreign_source"] is True
    transfer_history = json.loads(Path(transfer_manifest["history_json"]).read_text(encoding="utf-8"))
    assert transfer_history["mix_preserve_companion_stem"] == "vocals"
    assert transfer_history["leak_transfer_to_foreign_source"] is True


def test_forensic_proxy_training_records_companion_without_ground_truth(tmp_path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 2
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    clean_vocal = np.stack(
        [
            0.05 * np.sin(2.0 * np.pi * 720.0 * t),
            0.045 * np.sin(2.0 * np.pi * 725.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    drum_imprint = np.stack(
        [
            0.025 * np.sin(2.0 * np.pi * 2200.0 * t),
            0.025 * np.sin(2.0 * np.pi * 2210.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    bad_vocal = clean_vocal + drum_imprint
    minus = drum_imprint.copy()
    window_dir = tmp_path / "w000"
    candidate_dir = window_dir / "c103"
    _write(window_dir / "mix.flac", clean_vocal + minus)
    _write(candidate_dir / "stems" / "vocals.flac", bad_vocal)
    _write(candidate_dir / "stems" / "drums.flac", drum_imprint)
    _write(candidate_dir / "stems" / "minus.flac", minus)
    proxy_path = candidate_dir / "score" / "forensic" / "pairs" / "proxy" / "vocals_drums.flac"
    _write(proxy_path, drum_imprint)
    pair_json = candidate_dir / "score" / "forensic" / "pairs" / "vocals_drums.json"
    pair_json.parent.mkdir(parents=True, exist_ok=True)
    pair_json.write_text(
        json.dumps(
            {
                "schema": "stem_pair_forensic.v1",
                "tested_stem": "vocals",
                "source_stem": "drums",
                "source_path": str(candidate_dir / "stems" / "drums.flac"),
                "verdict": "fail",
                "forensic_proxy_audio": "proxy/vocals_drums.flac",
                "regions": [{"start": 0.0, "end": frames / sample_rate}],
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "c103",
                "stem_paths": {
                    "vocals": str(candidate_dir / "stems" / "vocals.flac"),
                    "drums": str(candidate_dir / "stems" / "drums.flac"),
                    "minus": str(candidate_dir / "stems" / "minus.flac"),
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "score.json").write_text(json.dumps({"contamination_score": {"leak_results": []}}), encoding="utf-8")

    manifest = train_from_leak_evidence(
        candidate_dir=candidate_dir,
        output_dir=tmp_path / "train",
        stem="vocals",
        issue_type="instrument_leak",
        foreign_sources=["drums"],
        leak_gain=1.0,
        forensic_pair_json=pair_json,
        config=StemRegionRepairTrainConfig(
            epochs=90,
            hidden_channels=12,
            learning_rate=0.003,
            mix_conservation_loss_weight=0.0,
            stereo_preserve_loss_weight=0.0,
            forensic_source_decorrelation_loss_weight=50.0,
            forensic_source_envelope_loss_weight=20.0,
            forensic_envelope_window_ms=12.0,
            forensic_source_conditioning=True,
            device="cpu",
        ),
    )

    assert manifest["is_ground_truth"] is False
    assert manifest["label_type"] == "pseudo_teacher"
    assert manifest["mix_preserve_companion_stem"] == "minus"
    assert manifest["leak_evidence"][0]["evidence_type"] == "cross_stem_forensic_proxy"
    assert manifest["forensic_source_decorrelation_loss_weight"] == 50.0
    assert manifest["forensic_source_envelope_loss_weight"] == 20.0
    assert manifest["forensic_envelope_window_ms"] == 12.0
    assert manifest["forensic_source_conditioning"] is True
    assert manifest["forensic_source_audio_path"].endswith("drums.flac")
    history = json.loads(Path(manifest["history_json"]).read_text(encoding="utf-8"))
    assert history["forensic_source_condition_stem"] == "drums"
    assert history["mix_preserve_companion_stem"] == "minus"
    assert manifest["target_issue_improved"] is True


def test_forensic_envelope_target_reduces_high_band_without_erasing_vocal_body() -> None:
    sample_rate = 16000
    frames = sample_rate
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    vocal_body = np.stack(
        [
            0.06 * np.sin(2.0 * np.pi * 440.0 * t),
            0.055 * np.sin(2.0 * np.pi * 445.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    drum_hf = np.zeros_like(vocal_body)
    burst = ((t >= 0.2) & (t <= 0.32)).astype(np.float32)
    drum_hf[:, 0] = 0.035 * burst * np.sin(2.0 * np.pi * 3200.0 * t)
    drum_hf[:, 1] = 0.035 * burst * np.sin(2.0 * np.pi * 3230.0 * t)
    bad_vocal = vocal_body + drum_hf

    target = leak_train._build_forensic_envelope_target(
        bad_audio=bad_vocal,
        source_audio=drum_hf,
        sample_rate=sample_rate,
        regions=[{"start": 0.15, "end": 0.35}],
        attenuation=0.85,
        low_hz=1200.0,
        high_hz=7000.0,
    )

    bad_hf = band_limited_audio(bad_vocal, sample_rate, 1200.0, 7000.0)
    target_hf = band_limited_audio(target, sample_rate, 1200.0, 7000.0)
    body_before = band_limited_audio(bad_vocal, sample_rate, 80.0, 900.0)
    body_after = band_limited_audio(target, sample_rate, 80.0, 900.0)

    assert np.sqrt(np.mean(np.square(target_hf))) < np.sqrt(np.mean(np.square(bad_hf))) * 0.45
    assert np.max(np.abs(body_after - body_before)) < 1.0e-3
