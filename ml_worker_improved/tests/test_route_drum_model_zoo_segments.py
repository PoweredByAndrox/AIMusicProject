from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import route_drum_model_zoo_segments as drum_router


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _drum_pulses(t: np.ndarray, beat_seconds: float) -> np.ndarray:
    pulses = np.zeros_like(t, dtype=np.float32)
    pulse_indices = (np.arange(0.1, float(t[-1]), beat_seconds) * (len(t) / float(t[-1]))).astype(int)
    pulse_indices = pulse_indices[pulse_indices < len(pulses)]
    pulses[pulse_indices] = 1.0
    return np.convolve(pulses, np.hanning(96).astype(np.float32), mode="same")


def _vocal_phrase(t: np.ndarray, frequency_hz: float = 220.0) -> np.ndarray:
    envelope = np.zeros_like(t, dtype=np.float32)
    envelope[(t >= 0.1) & (t <= 0.85)] = 1.0
    return (0.35 * envelope * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def test_segment_router_switches_between_model_candidates(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 6.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    leakage = 0.12 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    half = len(t) // 2

    candidate_a = drums.copy()
    candidate_a[half:] = 0.45 * drums[half:] + leakage[half:]
    candidate_b = 0.45 * drums + leakage
    candidate_b[half:] = drums[half:]
    reference = np.stack([leakage, leakage], axis=1)

    a_path = tmp_path / "a.flac"
    b_path = tmp_path / "b.flac"
    ref_path = tmp_path / "leakage.flac"
    _write_audio(a_path, np.stack([candidate_a, candidate_a], axis=1), sample_rate)
    _write_audio(b_path, np.stack([candidate_b, candidate_b], axis=1), sample_rate)
    _write_audio(ref_path, reference, sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("candidate_a", a_path, (ref_path,)),
            drum_router.CandidateSpec("candidate_b", b_path, (ref_path,)),
        ],
        tmp_path / "out",
        listen_dir=tmp_path / "listen",
        output_label="selected.flac",
        config=drum_router.RouterConfig(
            segment_seconds=3.0,
            crossfade_seconds=0.02,
            switch_hysteresis=0.0,
            leakage_corr_weight=8.0,
            min_rms_dbfs=-50.0,
        ),
    )

    selected = [decision["selected"] for decision in manifest["decisions"]]
    assert selected == ["candidate_a", "candidate_b"]
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert Path(manifest["output"]["path"]).exists()
    candidate_manifest_path = Path(manifest["candidate_manifest"]["path"])
    assert candidate_manifest_path.exists()
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    assert candidate_manifest["stems"]["drums"]["path"] == manifest["output"]["path"]
    assert candidate_manifest["candidate_contract"]["route"] == "drum_model_zoo_segment_router_v1"
    assert candidate_manifest["candidate_contract"]["quality_gate_required"] is True
    assert candidate_manifest["candidate_contract"]["delivery_status"] == "requires_objective_quality_gate"
    assert candidate_manifest["candidate_contract"]["stem_name"] == "drums"
    assert "leakage" in manifest["decisions"][0]["ranked"][0]["leakage_corr_by_reference"]
    assert manifest["listen_pack"]["selected.flac"].endswith("selected.flac")


def test_segment_router_rejects_too_quiet_candidate_by_score(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 3.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)

    normal_path = tmp_path / "normal.flac"
    quiet_path = tmp_path / "quiet.flac"
    _write_audio(normal_path, np.stack([drums, drums], axis=1), sample_rate)
    _write_audio(quiet_path, np.stack([0.02 * drums, 0.02 * drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("normal", normal_path),
            drum_router.CandidateSpec("too_quiet", quiet_path),
        ],
        tmp_path / "out",
        config=drum_router.RouterConfig(
            segment_seconds=3.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            min_rms_dbfs=-36.0,
            quiet_penalty_weight=1.0,
        ),
    )

    assert manifest["decisions"][0]["selected"] == "normal"
    ranked = manifest["decisions"][0]["ranked"]
    assert ranked[0]["label"] == "normal"
    assert ranked[1]["quiet_penalty"] > ranked[0]["quiet_penalty"]


def test_segment_router_uses_verified_target_reference_when_available(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.18 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    target_path = tmp_path / "reference_drums.flac"
    dirty_path = tmp_path / "dirty.flac"
    cleaner_path = tmp_path / "cleaner.flac"
    _write_audio(target_path, np.stack([drums, drums], axis=1), sample_rate)
    _write_audio(dirty_path, np.stack([drums + tonal_leakage, drums + tonal_leakage], axis=1), sample_rate)
    _write_audio(cleaner_path, np.stack([0.92 * drums, 0.92 * drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("dirty", dirty_path),
            drum_router.CandidateSpec("cleaner", cleaner_path),
        ],
        tmp_path / "target_routed",
        target_reference_path=target_path,
        output_label="target_selected.flac",
        config=drum_router.RouterConfig(
            segment_seconds=2.0,
            crossfade_seconds=0.0,
            target_corr_weight=20.0,
            target_rms_delta_weight=0.5,
            min_rms_dbfs=-50.0,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"cleaner"}
    assert manifest["contract"]["target_reference_used"] is True
    assert manifest["candidate_manifest"]["path"].endswith("candidate_manifest.json")


def test_segment_router_target_reference_is_active_with_default_config(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.18 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    target_path = tmp_path / "reference_drums.flac"
    dirty_path = tmp_path / "dirty.flac"
    cleaner_path = tmp_path / "cleaner.flac"
    _write_audio(target_path, np.stack([drums, drums], axis=1), sample_rate)
    _write_audio(dirty_path, np.stack([drums + tonal_leakage, drums + tonal_leakage], axis=1), sample_rate)
    _write_audio(cleaner_path, np.stack([0.92 * drums, 0.92 * drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("dirty", dirty_path),
            drum_router.CandidateSpec("cleaner", cleaner_path),
        ],
        tmp_path / "target_default",
        target_reference_path=target_path,
        config=drum_router.RouterConfig(segment_seconds=2.0, crossfade_seconds=0.0, min_rms_dbfs=-50.0),
    )
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))

    assert manifest["decisions"][0]["selected"] == "cleaner"
    assert candidate_manifest["candidate_contract"]["route"] == "drum_model_zoo_segment_router_v1"
    assert candidate_manifest["candidate_contract"]["target_reference_used"] is True
    assert candidate_manifest["candidate_contract"]["target_reference_path"] == str(target_path)


def test_segment_router_can_write_a_non_drum_stem_candidate_manifest(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    piano = 0.2 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)

    candidate_path = tmp_path / "piano.flac"
    _write_audio(candidate_path, np.stack([piano, piano], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [drum_router.CandidateSpec("piano_candidate", candidate_path)],
        tmp_path / "piano_route",
        stem_name="piano",
        config=drum_router.RouterConfig(segment_seconds=2.0, crossfade_seconds=0.0, min_rms_dbfs=-80.0),
    )
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))

    assert candidate_manifest["candidate_contract"]["route"] == "stem_model_zoo_segment_router_v1"
    assert candidate_manifest["candidate_contract"]["stem_name"] == "piano"
    assert "piano" in candidate_manifest["stems"]
    assert Path(candidate_manifest["stems"]["piano"]["path"]).name == "06_segment_model_router_piano.flac"


def test_segment_router_treats_drums_plus_percussive_fx_aux_as_drum_profile(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    drums = _drum_pulses(t, 0.5)

    candidate_path = tmp_path / "drums_plus.flac"
    _write_audio(candidate_path, np.stack([drums, drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [drum_router.CandidateSpec("drums_plus", candidate_path)],
        tmp_path / "drums_plus_route",
        stem_name="drums_plus_percussive_fx_aux",
        config=drum_router.RouterConfig(segment_seconds=2.0, crossfade_seconds=0.0, min_rms_dbfs=-80.0),
    )
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))

    assert manifest["contract"]["stem_score_profile"] == "drums"
    assert candidate_manifest["candidate_contract"]["stem_score_profile"] == "drums"
    assert candidate_manifest["candidate_contract"]["route"] == "drum_model_zoo_segment_router_v1"
    assert candidate_manifest["candidate_contract"]["stem_name"] == "drums_plus_percussive_fx_aux"
    assert "drums_plus_percussive_fx_aux" in candidate_manifest["stems"]


def test_segment_router_cli_default_output_label_uses_stem_name(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    piano = 0.2 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)
    candidate_path = tmp_path / "piano.flac"
    output_dir = tmp_path / "cli_out"
    _write_audio(candidate_path, np.stack([piano, piano], axis=1), sample_rate)

    monkeypatch.setattr(
        "sys.argv",
        [
            "route_drum_model_zoo_segments.py",
            str(output_dir),
            "--candidate",
            f"piano_candidate={candidate_path}",
            "--stem-name",
            "piano",
            "--segment-seconds",
            "1.0",
            "--crossfade-seconds",
            "0.0",
            "--min-rms-dbfs",
            "-80.0",
        ],
    )

    drum_router.main()

    assert (output_dir / "06_segment_model_router_piano.flac").exists()
    assert not (output_dir / "None").exists()


def test_vocal_segment_router_prefers_harmonic_body_over_harsh_noise_without_target_reference(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    clean_vocal = _vocal_phrase(t)
    harsh_noise = 0.20 * np.sin(2.0 * np.pi * 6500.0 * t).astype(np.float32)
    thin_vocal = 0.15 * _vocal_phrase(t)
    clean_path = tmp_path / "clean_vocals.flac"
    harsh_path = tmp_path / "harsh_vocals.flac"
    _write_audio(clean_path, np.stack([clean_vocal, clean_vocal], axis=1), sample_rate)
    _write_audio(harsh_path, np.stack([thin_vocal + harsh_noise, thin_vocal + harsh_noise], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("clean", clean_path),
            drum_router.CandidateSpec("harsh", harsh_path),
        ],
        tmp_path / "out",
        stem_name="vocals",
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"clean"}
    ranked = manifest["decisions"][0]["ranked"]
    assert ranked[0]["label"] == "clean"
    assert ranked[0]["stem_profile"] == "vocals"
    assert ranked[1]["high_band_noise_penalty"] > ranked[0]["high_band_noise_penalty"]
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))
    assert candidate_manifest["candidate_contract"]["target_reference_used"] is False
    assert candidate_manifest["candidate_contract"]["delivery_status"] == "requires_objective_quality_gate"


def test_vocal_segment_router_penalizes_named_leakage_without_target_reference(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0.0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    vocal = _vocal_phrase(t)
    leakage = 0.20 * np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)
    clean_path = tmp_path / "clean_vocals.flac"
    leaky_path = tmp_path / "leaky_vocals.flac"
    leakage_path = tmp_path / "melody_leakage.flac"
    _write_audio(clean_path, np.stack([vocal, vocal], axis=1), sample_rate)
    _write_audio(leaky_path, np.stack([vocal + leakage, vocal + leakage], axis=1), sample_rate)
    _write_audio(leakage_path, np.stack([leakage, leakage], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("clean", clean_path, (leakage_path,)),
            drum_router.CandidateSpec("leaky", leaky_path, (leakage_path,)),
        ],
        tmp_path / "out",
        stem_name="vocals",
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            leakage_corr_weight=12.0,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"clean"}
    first_ranked = manifest["decisions"][0]["ranked"]
    assert first_ranked[0]["label"] == "clean"
    assert first_ranked[1]["leakage_corr_max"] > first_ranked[0]["leakage_corr_max"]


def test_reference_free_router_keeps_baseline_when_score_margin_is_weak(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    baseline_drums = _drum_pulses(t, 0.5)
    slightly_louder_drums = 1.03 * baseline_drums
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    _write_audio(baseline_path, np.stack([baseline_drums, baseline_drums], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([slightly_louder_drums, slightly_louder_drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("baseline", baseline_path),
            drum_router.CandidateSpec("challenger", challenger_path),
        ],
        tmp_path / "out",
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            min_rms_dbfs=-60.0,
            baseline_candidate_label="baseline",
            baseline_switch_margin=2.0,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"baseline"}
    first_guard = manifest["decisions"][0]["selection_guard"]
    assert first_guard["applied"] is True
    assert first_guard["reason"] == "baseline_score_margin_not_met"
    assert first_guard["baseline_label"] == "baseline"
    assert first_guard["unguarded_selected"] == "challenger"
    assert first_guard["required_margin"] == 2.0
    assert first_guard["confidence_metric"] == "score_delta_vs_baseline"
    assert first_guard["no_reference_confidence_score"] == first_guard["score_delta"]
    assert all("selection_guard" in decision for decision in manifest["decisions"])
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))
    assert candidate_manifest["candidate_contract"]["reference_free_baseline_guard_enabled"] is True
    assert candidate_manifest["candidate_contract"]["baseline_candidate_label"] == "baseline"
    assert candidate_manifest["candidate_contract"]["baseline_guard_confidence_metric"] == "score_delta_vs_baseline"


def test_reference_free_router_allows_switch_when_score_margin_is_strong(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    clean_drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    leakage_path = tmp_path / "leakage.flac"
    _write_audio(baseline_path, np.stack([0.8 * clean_drums + tonal_leakage, 0.8 * clean_drums + tonal_leakage], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([clean_drums, clean_drums], axis=1), sample_rate)
    _write_audio(leakage_path, np.stack([tonal_leakage, tonal_leakage], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("baseline", baseline_path, (leakage_path,)),
            drum_router.CandidateSpec("challenger", challenger_path, (leakage_path,)),
        ],
        tmp_path / "out",
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            leakage_corr_weight=10.0,
            min_rms_dbfs=-60.0,
            baseline_candidate_label="baseline",
            baseline_switch_margin=0.1,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"challenger"}
    first_guard = manifest["decisions"][0]["selection_guard"]
    assert first_guard["applied"] is False
    assert first_guard["reason"] == "score_margin_met"
    assert first_guard["baseline_label"] == "baseline"
    assert first_guard["unguarded_selected"] == "challenger"
    assert first_guard["confidence_metric"] == "score_delta_vs_baseline"


def test_reference_free_baseline_guard_is_disabled_in_target_reference_mode(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 2.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    clean_drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    target_path = tmp_path / "target.flac"
    _write_audio(baseline_path, np.stack([0.8 * clean_drums + tonal_leakage, 0.8 * clean_drums + tonal_leakage], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([clean_drums, clean_drums], axis=1), sample_rate)
    _write_audio(target_path, np.stack([clean_drums, clean_drums], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("baseline", baseline_path),
            drum_router.CandidateSpec("challenger", challenger_path),
        ],
        tmp_path / "out",
        target_reference_path=target_path,
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            min_rms_dbfs=-60.0,
            baseline_candidate_label="baseline",
            baseline_switch_margin=999.0,
        ),
    )

    assert {decision["selected"] for decision in manifest["decisions"]} == {"challenger"}
    first_guard = manifest["decisions"][0]["selection_guard"]
    assert first_guard["enabled"] is False
    assert first_guard["applied"] is False
    assert first_guard["reason"] == "target_reference_mode"
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))
    assert candidate_manifest["candidate_contract"]["reference_free_baseline_guard_enabled"] is False
    assert candidate_manifest["candidate_contract"]["target_reference_used"] is True


def test_reference_free_router_limits_switches_by_baseline_budget(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_sec = 4.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    clean_drums = _drum_pulses(t, 0.5)
    tonal_leakage = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    baseline_path = tmp_path / "baseline.flac"
    challenger_path = tmp_path / "challenger.flac"
    leakage_path = tmp_path / "leakage.flac"
    _write_audio(baseline_path, np.stack([0.8 * clean_drums + tonal_leakage, 0.8 * clean_drums + tonal_leakage], axis=1), sample_rate)
    _write_audio(challenger_path, np.stack([clean_drums, clean_drums], axis=1), sample_rate)
    _write_audio(leakage_path, np.stack([tonal_leakage, tonal_leakage], axis=1), sample_rate)

    manifest = drum_router.route_drum_segments(
        [
            drum_router.CandidateSpec("baseline", baseline_path, (leakage_path,)),
            drum_router.CandidateSpec("challenger", challenger_path, (leakage_path,)),
        ],
        tmp_path / "out",
        config=drum_router.RouterConfig(
            segment_seconds=1.0,
            crossfade_seconds=0.0,
            switch_hysteresis=0.0,
            leakage_corr_weight=10.0,
            min_rms_dbfs=-60.0,
            baseline_candidate_label="baseline",
            baseline_switch_margin=0.0,
            baseline_max_switch_fraction=0.25,
        ),
    )

    selected = [decision["selected"] for decision in manifest["decisions"]]
    assert selected.count("challenger") == 1
    assert selected.count("baseline") == 3
    budget_blocked = [decision for decision in manifest["decisions"] if decision["selection_guard"]["reason"] == "baseline_switch_budget_exceeded"]
    assert len(budget_blocked) == 3
    candidate_manifest = json.loads(Path(manifest["candidate_manifest"]["path"]).read_text(encoding="utf-8"))
    assert candidate_manifest["candidate_contract"]["baseline_max_switch_fraction"] == 0.25
