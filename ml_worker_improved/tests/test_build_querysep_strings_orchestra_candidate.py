from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_querysep_strings_orchestra_candidate.py"
)
spec = importlib.util.spec_from_file_location("build_querysep_strings_orchestra_candidate", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_querysep_strings_candidate_builds_base_then_long_tail_without_reference(
    tmp_path: Path, monkeypatch
) -> None:
    source_audio = tmp_path / "source_mix.flac"
    source_audio.write_bytes(b"fake source")
    output_dir = tmp_path / "out"

    calls: dict[str, object] = {}

    def fake_export(source_audio_path, base_output_dir, **kwargs):
        base_output_dir = Path(base_output_dir)
        base_output_dir.mkdir(parents=True)
        manifest_path = base_output_dir / "stem_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "contract": {
                        "target_reference_assisted": False,
                        "target_reference_used_at_application": False,
                        "application_reference_free": True,
                        "delivery_allowed_without_reference": True,
                        "fallback_used": False,
                    },
                    "stems": {
                        "drums": {"path": str(base_output_dir / "drums.flac")},
                        "bass": {"path": str(base_output_dir / "bass.flac")},
                        "other": {"path": str(base_output_dir / "other.flac")},
                        "piano": {"path": str(base_output_dir / "piano.flac")},
                    },
                    "manifest_path": str(manifest_path),
                }
            ),
            encoding="utf-8",
        )
        calls["export"] = {
            "source_audio_path": str(source_audio_path),
            "base_output_dir": str(base_output_dir),
            "model_name": kwargs["model_name"],
            "device": kwargs["device"],
        }
        return {"manifest_path": str(manifest_path)}

    def fake_querysep(**kwargs):
        querysep_output_dir = Path(kwargs["output_dir"])
        querysep_output_dir.mkdir(parents=True, exist_ok=True)
        selected_path = querysep_output_dir / "strings_orchestra.wav"
        sf.write(
            selected_path,
            np.zeros((128, 2), dtype=np.float32),
            44_100,
            subtype="PCM_16",
        )
        calls["querysep"] = kwargs
        return {
            "status": "completed",
            "fallback_used": False,
            "backend": "banquet_byoq",
            "manifest_path": str(querysep_output_dir / "querysep_long_tail_manifest.json"),
            "stems": {
                "strings_orchestra": {
                    "status": "accepted",
                    "selected_output_path": str(selected_path),
                    "quality_gate": {"status": "pass"},
                }
            },
        }

    monkeypatch.setattr(module, "export_track_stems_aggressive_melody_first", fake_export)
    monkeypatch.setattr(module, "build_querysep_long_tail_stems", fake_querysep)

    manifest = module.build_querysep_strings_orchestra_candidate(
        source_audio_path=source_audio,
        output_dir=output_dir,
        querysep_worker_url="http://querysep-worker:8000",
        querysep_output_root="/main_lab_artifacts/out",
    )

    assert calls["export"]["source_audio_path"] == str(source_audio)
    assert calls["querysep"]["target_stems"] == ["strings_orchestra"]
    assert calls["querysep"]["backend"] == "banquet_byoq"
    assert calls["querysep"]["dry_run"] is False
    assert calls["querysep"]["source_stem_manifest_path"].endswith("base_stems/stem_manifest.json")
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["stems"]["strings_orchestra"]["path"].endswith("strings_orchestra.flac")
    assert manifest["pipeline"]["querysep_backend"] == "banquet_byoq"


def test_querysep_strings_candidate_lets_long_tail_builder_map_output_root_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    source_audio = tmp_path / "source_mix.flac"
    source_audio.write_bytes(b"fake source")
    output_dir = tmp_path / "out"
    calls: dict[str, object] = {}

    def fake_export(source_audio_path, base_output_dir, **kwargs):
        base_output_dir = Path(base_output_dir)
        base_output_dir.mkdir(parents=True)
        manifest_path = base_output_dir / "stem_manifest.json"
        manifest_path.write_text(
            json.dumps({"contract": {"application_reference_free": True}, "manifest_path": str(manifest_path)}),
            encoding="utf-8",
        )
        return {"manifest_path": str(manifest_path)}

    def fake_querysep(**kwargs):
        querysep_output_dir = Path(kwargs["output_dir"])
        querysep_output_dir.mkdir(parents=True, exist_ok=True)
        selected_path = querysep_output_dir / "strings_orchestra.wav"
        sf.write(selected_path, np.zeros((128, 2), dtype=np.float32), 44_100, subtype="PCM_16")
        calls["querysep"] = kwargs
        return {
            "status": "completed",
            "fallback_used": False,
            "backend": "banquet_byoq",
            "manifest_path": str(querysep_output_dir / "querysep_long_tail_manifest.json"),
            "stems": {"strings_orchestra": {"status": "accepted", "selected_output_path": str(selected_path)}},
        }

    monkeypatch.setattr(module, "export_track_stems_aggressive_melody_first", fake_export)
    monkeypatch.setattr(module, "build_querysep_long_tail_stems", fake_querysep)

    manifest = module.build_querysep_strings_orchestra_candidate(
        source_audio_path=source_audio,
        output_dir=output_dir,
        querysep_worker_url="http://querysep-worker:8000",
    )

    assert calls["querysep"]["querysep_output_root"] is None
    assert manifest["pipeline"]["querysep_output_root"] is None


def test_querysep_strings_candidate_refuses_rejected_long_tail_output(
    tmp_path: Path, monkeypatch
) -> None:
    source_audio = tmp_path / "source_mix.flac"
    source_audio.write_bytes(b"fake source")
    output_dir = tmp_path / "out"

    def fake_export(source_audio_path, base_output_dir, **kwargs):
        base_output_dir = Path(base_output_dir)
        base_output_dir.mkdir(parents=True)
        manifest_path = base_output_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps({"manifest_path": str(manifest_path)}), encoding="utf-8")
        return {"manifest_path": str(manifest_path)}

    def fake_querysep(**kwargs):
        querysep_output_dir = Path(kwargs["output_dir"])
        querysep_output_dir.mkdir(parents=True, exist_ok=True)
        rejected_path = querysep_output_dir / "strings_orchestra.wav"
        sf.write(rejected_path, np.zeros((128, 2), dtype=np.float32), 44_100, subtype="PCM_16")
        return {
            "status": "rejected",
            "failure": {"reason": "long_tail_model_router_rejected_all_candidates"},
            "manifest_path": str(querysep_output_dir / "querysep_long_tail_manifest.json"),
            "stems": {
                "strings_orchestra": {
                    "status": "rejected",
                    "selected_output_path": str(rejected_path),
                    "quality_gate": {"status": "rejected"},
                }
            },
        }

    monkeypatch.setattr(module, "export_track_stems_aggressive_melody_first", fake_export)
    monkeypatch.setattr(module, "build_querysep_long_tail_stems", fake_querysep)

    try:
        module.build_querysep_strings_orchestra_candidate(
            source_audio_path=source_audio,
            output_dir=output_dir,
            querysep_worker_url="http://querysep-worker:8000",
        )
    except RuntimeError as exc:
        assert "QuerySep long-tail route did not accept strings_orchestra" in str(exc)
    else:
        raise AssertionError("rejected QuerySep output must not be wrapped as a candidate manifest")
