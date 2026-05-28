from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import run_stem_quality_benchmark as benchmark


def _fixture_by_name(manifest: dict, name: str) -> dict:
    for fixture in manifest["fixtures"]:
        if fixture["name"] == name:
            return fixture
    raise AssertionError(f"Missing fixture: {name}")


def test_benchmark_smoke_passes_with_expected_defect_detection(tmp_path: Path) -> None:
    manifest = benchmark.run_benchmark(
        tmp_path / "benchmark",
        config=benchmark.BenchmarkConfig(sample_rate=8000, duration_seconds=3.0),
    )

    assert manifest["summary"]["status"] == "passed"
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["verified_synthetic_truth"] is True
    expected_by_fixture = {
        "drums_with_tonal_melody_leak": {"drum_tonal_leakage", "harmonic_leakage_in_drums"},
        "vocal_dropout_tail_loss": {"vocal_dropout", "vocal_tail_loss"},
        "vocal_harsh_noise": {"vocal_harshness"},
        "minus_with_vocal_residue": {"vocal_residue_in_minus"},
        "long_tail_proxy_copy": {"possible_other_proxy_leakage", "proxy_copy_risk"},
        "near_silent_sparse_stem": {"insufficient_presence", "near_silent_output"},
        "source_conservation_violation": {"source_conservation_error"},
    }
    for fixture_name, expected_labels in expected_by_fixture.items():
        fixture = _fixture_by_name(manifest, fixture_name)
        assert expected_labels.issubset(set(fixture["detected_labels"]))
        assert fixture["missing_expected_labels"] == []


def test_clean_fixture_has_no_p1_false_positive(tmp_path: Path) -> None:
    manifest = benchmark.run_benchmark(
        tmp_path / "benchmark",
        config=benchmark.BenchmarkConfig(sample_rate=8000, duration_seconds=3.0),
    )

    clean_fixture = _fixture_by_name(manifest, "clean_stem_bundle")
    assert clean_fixture["expected_labels"] == []
    assert clean_fixture["unexpected_p1_labels"] == []
    assert clean_fixture["status"] == "passed"


def test_benchmark_writes_manifest_with_hashes_and_thresholds(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    manifest = benchmark.run_benchmark(
        output_dir,
        config=benchmark.BenchmarkConfig(sample_rate=8000, duration_seconds=3.0),
    )
    manifest_path = output_dir / "benchmark_manifest.json"
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert reloaded["summary"] == manifest["summary"]
    assert reloaded["contract"]["no_real_track_claims"] is True
    assert "drum_tonal_narrowband_threshold" in reloaded["config"]
    assert reloaded["fixtures"][0]["source"]["sha256"]
    assert reloaded["fixtures"][0]["stems"]["drums"]["sha256"]


def test_known_defect_fixture_fails_on_unallowed_extra_p1(tmp_path: Path, monkeypatch) -> None:
    original_detect_fixture_labels = benchmark.detect_fixture_labels

    def detect_with_unallowed_extra_label(fixture, clean_reference_stems, config):
        detected_labels, metrics = original_detect_fixture_labels(fixture, clean_reference_stems, config)
        if fixture.name == "drums_with_tonal_melody_leak":
            detected_labels = set(detected_labels)
            detected_labels.add("vocal_dropout")
        return detected_labels, metrics

    monkeypatch.setattr(benchmark, "detect_fixture_labels", detect_with_unallowed_extra_label)

    manifest = benchmark.run_benchmark(
        tmp_path / "benchmark",
        config=benchmark.BenchmarkConfig(sample_rate=8000, duration_seconds=3.0),
    )
    fixture = _fixture_by_name(manifest, "drums_with_tonal_melody_leak")

    assert manifest["summary"]["status"] == "failed"
    assert fixture["status"] == "failed"
    assert "vocal_dropout" in fixture["unexpected_p1_labels"]


def test_cli_fail_on_regression_returns_nonzero(tmp_path: Path, monkeypatch) -> None:
    def failed_benchmark(output_dir: Path, *, suite: str = "smoke", config=None) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"summary": {"status": "failed"}}

    monkeypatch.setattr(benchmark, "run_benchmark", failed_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_stem_quality_benchmark.py", str(tmp_path / "benchmark"), "--fail-on-regression"],
    )

    assert benchmark.main() == 1
