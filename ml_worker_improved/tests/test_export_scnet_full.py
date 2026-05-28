from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "export_scnet_full.py"


def _load_module():
    if str(TOOLS_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOLS_ROOT))
    spec = importlib.util.spec_from_file_location("export_scnet_full", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scnet_full_exporter_writes_multistem_no_reference_manifest() -> None:
    module = _load_module()
    sample_rate = 44100
    source_audio = np.zeros((sample_rate, 2), dtype=np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        repo_path = root / "repo"
        config_path = root / "config.yaml"
        model_path = root / "model.ckpt"
        source_path = root / "source.flac"
        output_dir = root / "out"
        repo_path.mkdir()
        (repo_path / "inference.py").write_text("# fake", encoding="utf-8")
        config_path.write_text("inference:\n  num_overlap: 4\n", encoding="utf-8")
        model_path.write_bytes(b"fake checkpoint")
        sf.write(source_path, source_audio, sample_rate)

        executed_commands: list[list[str]] = []

        def fake_run(command, *, cwd, capture_output, text, check):
            executed_commands.append([str(part) for part in command])
            store_dir = Path(command[command.index("--store_dir") + 1])
            stem_dir = store_dir / "full_track"
            stem_dir.mkdir(parents=True)
            for index, stem_name in enumerate(("drums", "bass", "other", "vocals"), start=1):
                stem_audio = np.full((sample_rate, 2), 0.01 * index, dtype=np.float32)
                sf.write(stem_dir / f"{stem_name}.wav", stem_audio, sample_rate)

            class Completed:
                returncode = 0
                stdout = ""
                stderr = ""

            return Completed()

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            exit_code = module.main(
                [
                    str(source_path),
                    str(output_dir),
                    "--repo-path",
                    str(repo_path),
                    "--config-path",
                    str(config_path),
                    "--model-path",
                    str(model_path),
                    "--device-id",
                    "0",
                    "--num-overlap",
                    "2",
                ]
            )

        manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert executed_commands
    command = executed_commands[0]
    assert "inference.py" in command
    assert command[command.index("--model_type") + 1] == "scnet"
    assert command[command.index("--filename_template") + 1] == "{file_name}/{instr}"
    assert manifest["model_id"] == "scnet_xl_ihf_musdb18"
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is True
    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["per_track_hardcode_allowed"] is False
    assert manifest["contract"]["not_timestamp_hardcoded"] is True
    assert set(manifest["stems"]) == {"drums", "bass", "other", "vocals"}
    assert manifest["stems"]["drums"]["role"] == "target"
