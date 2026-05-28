import importlib.util
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "export_drums_plus_percussive_fx_aux.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_drums_plus_percussive_fx_aux", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_target_manifest_relabels_no_reference_drum_export(tmp_path: Path) -> None:
    module = _load_module()
    base_dir = tmp_path / "base_export"
    base_dir.mkdir()
    drums_path = base_dir / "drums.flac"
    drums_path.write_bytes(b"fake-flac")
    base_manifest = {
        "manifest_path": str(base_dir / "stem_manifest.json"),
        "audio_path": "/app/input/source.flac",
        "model_name": "htdemucs_6s",
        "sample_rate": 44100,
        "export_subtype": "PCM_24",
        "contract": {
            "benchmark_compatible": True,
            "target_reference_assisted": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "fallback_used": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "pipeline": {"mode": "htdemucs_6s"},
        "stems": {
            "drums": {
                "path": str(drums_path),
                "rms_dbfs": -12.0,
                "peak_dbfs": -1.0,
            }
        },
    }

    manifest = module.build_drums_plus_percussive_fx_aux_manifest(
        base_manifest=base_manifest,
        output_dir=tmp_path / "candidate",
        route_name="test_route",
    )

    target_entry = manifest["stems"]["drums_plus_percussive_fx_aux"]
    assert manifest["model_id"] == "test_route"
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert target_entry["source_model_stem"] == "drums"
    assert target_entry["target_definition"] == "drums_plus_percussive_fx_aux"
    assert Path(target_entry["path"]).read_bytes() == b"fake-flac"


def test_build_target_manifest_from_existing_no_reference_drum_candidate(tmp_path: Path) -> None:
    module = _load_module()
    source_dir = tmp_path / "strict_candidate"
    source_dir.mkdir()
    drums_path = source_dir / "strict_drums.flac"
    drums_path.write_bytes(b"strict-drum-candidate")
    base_manifest_path = source_dir / "stem_manifest.json"
    base_manifest_path.write_text(
        """{
  "manifest_path": "will_be_overridden_by_loader",
  "source_audio_path": "/app/input/source.flac",
  "model_id": "drum_tonal_leakage_repair",
  "model_name": "drum_tonal_leakage_repair",
  "contract": {
    "benchmark_compatible": true,
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "pipeline": {
    "route": "drum_tonal_leakage_repair"
  },
  "stems": {
    "drums": {
      "path": "strict_drums.flac",
      "rms_dbfs": -10.0
    }
  }
}""",
        encoding="utf-8",
    )

    manifest = module.build_drums_plus_percussive_fx_aux_from_base_manifest(
        base_manifest_path=base_manifest_path,
        output_dir=tmp_path / "derived_candidate",
        route_name="strict_drum_candidate_to_percussive_aux",
    )

    target_entry = manifest["stems"]["drums_plus_percussive_fx_aux"]
    assert manifest["model_id"] == "strict_drum_candidate_to_percussive_aux"
    assert manifest["model"]["source_candidate_model_id"] == "drum_tonal_leakage_repair"
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["pipeline"]["source_candidate_manifest"] == str(base_manifest_path)
    assert manifest["pipeline"]["source_candidate_model_id"] == "drum_tonal_leakage_repair"
    assert target_entry["source_model_stem"] == "drums"
    assert target_entry["source_candidate_model_id"] == "drum_tonal_leakage_repair"
    assert Path(target_entry["path"]).read_bytes() == b"strict-drum-candidate"
    assert manifest["pipeline"]["provenance_contract"]["no_reference_lineage_verified"] is True
    assert str(base_manifest_path) in manifest["pipeline"]["provenance_contract"]["checked_manifest_paths"]


def test_rejects_base_manifest_with_incomplete_no_reference_contract(tmp_path: Path) -> None:
    module = _load_module()
    source_dir = tmp_path / "incomplete_candidate"
    source_dir.mkdir()
    drums_path = source_dir / "drums.flac"
    drums_path.write_bytes(b"incomplete")
    base_manifest_path = source_dir / "stem_manifest.json"
    base_manifest_path.write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "incomplete_contract_candidate",
  "contract": {
    "target_reference_assisted": false,
    "fallback_used": false
  },
  "stems": {
    "drums": {
      "path": "drums.flac"
    }
  }
}""",
        encoding="utf-8",
    )

    try:
        module.build_drums_plus_percussive_fx_aux_from_base_manifest(
            base_manifest_path=base_manifest_path,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "missing required key" in str(exc)
    else:
        raise AssertionError("incomplete no-reference contract must be rejected")


def test_cli_base_manifest_mode_does_not_run_hidden_inference(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    source_dir = tmp_path / "strict_candidate"
    source_dir.mkdir()
    drums_path = source_dir / "strict_drums.flac"
    drums_path.write_bytes(b"strict-drum-candidate")
    base_manifest_path = source_dir / "stem_manifest.json"
    base_manifest_path.write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "drum_tonal_leakage_repair",
  "contract": {
    "benchmark_compatible": true,
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "stems": {
    "drums": {
      "path": "strict_drums.flac"
    }
  }
}""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "derived_candidate"
    listen_dir = tmp_path / "listen"

    def fail_if_inference_runs(*_args, **_kwargs):
        raise AssertionError("base-manifest mode must not run separation inference")

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "export_track_stems_aggressive_melody_first", fail_if_inference_runs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_drums_plus_percussive_fx_aux.py",
            "--base-manifest",
            str(base_manifest_path),
            "--output-dir",
            str(output_dir),
            "--listen-dir",
            str(listen_dir),
            "--listen-label",
            "candidate.flac",
        ],
    )

    assert module.main() == 0
    assert (output_dir / "stem_manifest.json").exists()
    assert (listen_dir / "candidate.flac").read_bytes() == b"strict-drum-candidate"


def test_route_name_matches_requested_full_mix_model() -> None:
    module = _load_module()

    assert (
        module._route_name_for_model("htdemucs_6s")
        == "htdemucs_6s_drums_plus_percussive_fx_aux"
    )
    assert (
        module._route_name_for_model("htdemucs_ft")
        == "htdemucs_ft_drums_plus_percussive_fx_aux"
    )
    assert (
        module._route_name_for_model("hybrid_htdemucs_ft_bs_roformer_6s")
        == "hybrid_htdemucs_ft_bs_roformer_6s_drums_plus_percussive_fx_aux"
    )


def test_full_mix_hybrid_model_prepares_bs_roformer_runtime_before_inference(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    source_path = tmp_path / "source.flac"
    source_path.write_bytes(b"source-audio")
    output_dir = tmp_path / "candidate"
    calls: list[object] = []

    def fake_ensure_hybrid_bs_runtime() -> None:
        calls.append("ensure_hybrid_bs_runtime")

    def fake_export_track_stems_aggressive_melody_first(
        audio_path: Path,
        base_export_dir: Path,
        *,
        model_name: str,
        device: str,
        sharpen_bleed: bool,
    ) -> dict:
        calls.append(("export", model_name, device, sharpen_bleed))
        base_export_dir.mkdir(parents=True, exist_ok=True)
        drums_path = base_export_dir / "drums.flac"
        drums_path.write_bytes(b"hybrid-drums")
        return {
            "manifest_path": str(base_export_dir / "stem_manifest.json"),
            "audio_path": str(audio_path),
            "model_name": model_name,
            "sample_rate": 44100,
            "export_subtype": "PCM_24",
            "contract": {
                "benchmark_compatible": True,
                "target_reference_assisted": False,
                "target_reference_used_at_application": False,
                "application_reference_free": True,
                "fallback_used": False,
                "per_track_hardcode_allowed": False,
                "not_timestamp_hardcoded": True,
            },
            "pipeline": {"mode": model_name},
            "stems": {"drums": {"path": str(drums_path), "rms_dbfs": -12.0}},
        }

    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(module, "ensure_hybrid_bs_runtime", fake_ensure_hybrid_bs_runtime)
    monkeypatch.setattr(
        module,
        "export_track_stems_aggressive_melody_first",
        fake_export_track_stems_aggressive_melody_first,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_drums_plus_percussive_fx_aux.py",
            str(source_path),
            str(output_dir),
            "--model-name",
            "hybrid_htdemucs_ft_bs_roformer_6s",
            "--device",
            "cuda",
        ],
    )

    assert module.main() == 0
    assert calls[0] == "ensure_hybrid_bs_runtime"
    assert calls[1] == ("export", "hybrid_htdemucs_ft_bs_roformer_6s", "cuda", True)
    assert (output_dir / "stem_manifest.json").exists()


def test_rejects_reference_used_base_manifest_even_when_target_assisted_flag_is_false(tmp_path: Path) -> None:
    module = _load_module()
    source_dir = tmp_path / "unsafe_candidate"
    source_dir.mkdir()
    drums_path = source_dir / "drums.flac"
    drums_path.write_bytes(b"unsafe")
    base_manifest_path = source_dir / "stem_manifest.json"
    base_manifest_path.write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "unsafe_reference_candidate",
  "contract": {
    "target_reference_assisted": false,
    "target_reference_used_at_application": true,
    "application_reference_free": false,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "stems": {
    "drums": {
      "path": "drums.flac"
    }
  }
}""",
        encoding="utf-8",
    )

    try:
        module.build_drums_plus_percussive_fx_aux_from_base_manifest(
            base_manifest_path=base_manifest_path,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "target reference at application time" in str(exc)
    else:
        raise AssertionError("reference-used base manifest must be rejected")


def test_rejects_base_manifest_with_reference_used_ancestor(tmp_path: Path) -> None:
    module = _load_module()
    source_dir = tmp_path / "candidate"
    source_dir.mkdir()
    drums_path = source_dir / "drums.flac"
    drums_path.write_bytes(b"candidate")
    ancestor_path = source_dir / "reference_teacher_manifest.json"
    ancestor_path.write_text(
        """{
  "model_id": "reference_teacher",
  "contract": {
    "target_reference_assisted": false,
    "target_reference_used_at_application": true,
    "application_reference_free": false,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "stems": {
    "drums": {
      "path": "drums.flac"
    }
  }
}""",
        encoding="utf-8",
    )
    base_manifest_path = source_dir / "stem_manifest.json"
    base_manifest_path.write_text(
        """{
  "source_audio_path": "/app/input/source.flac",
  "model_id": "declared_safe_but_tainted",
  "contract": {
    "target_reference_assisted": false,
    "target_reference_used_at_application": false,
    "application_reference_free": true,
    "fallback_used": false,
    "per_track_hardcode_allowed": false,
    "not_timestamp_hardcoded": true
  },
  "pipeline": {
    "base_export_manifest": "reference_teacher_manifest.json"
  },
  "stems": {
    "drums": {
      "path": "drums.flac"
    }
  }
}""",
        encoding="utf-8",
    )

    try:
        module.build_drums_plus_percussive_fx_aux_from_base_manifest(
            base_manifest_path=base_manifest_path,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "target reference at application time" in str(exc)
    else:
        raise AssertionError("base manifest with reference-used ancestor must be rejected")
