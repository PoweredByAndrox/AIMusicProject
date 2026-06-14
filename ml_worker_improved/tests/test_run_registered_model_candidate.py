from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "run_registered_model_candidate.py"


def _load_executor_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("run_registered_model_candidate", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _demo_registry(tmp_path: Path):
    from worker.model_registry import ModelRegistry, ModelRegistryEntry

    return ModelRegistry(
        runtime_root=tmp_path,
        entries=(
            ModelRegistryEntry(
                model_id="demo_model",
                display_name="Demo model",
                family="test",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=("python", "demo.py", "{source_audio_path}", "{output_dir}", "--device", "{device}"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
            ModelRegistryEntry(
                model_id="missing_model",
                display_name="Missing model",
                family="test",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=("python", "missing.py", "{source_audio_path}", "{output_dir}"),
                expected_manifest="{output_dir}/stem_manifest.json",
                required_paths=("models/missing.ckpt",),
            ),
            ModelRegistryEntry(
                model_id="failing_subprocess_model",
                display_name="Failing subprocess model",
                family="test",
                target_stems=("vocals",),
                output_stems=("vocals",),
                command_template=(
                    sys.executable,
                    "-c",
                    "import sys; print('stdout marker'); print('stderr marker', file=sys.stderr); raise SystemExit(7)",
                ),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
            ModelRegistryEntry(
                model_id="demo_multistem_instrumental_model",
                display_name="Demo multi-stem instrumental model",
                family="test",
                target_stems=("instrumental",),
                output_stems=("vocals", "drums", "bass", "other"),
                command_template=("python", "demo_multistem.py", "{source_audio_path}", "{output_dir}"),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
            ModelRegistryEntry(
                model_id="demo_hq_full_stem_exporter",
                display_name="Demo HQ full stem exporter",
                family="test",
                target_stems=("instrumental",),
                output_stems=("vocals", "drums", "bass", "other", "instrumental"),
                command_template=(
                    "python",
                    "/app/tools/export_hq_stems_full.py",
                    "{source_audio_path}",
                    "{output_dir}",
                    "--model-name",
                    "htdemucs_6s",
                    "--device",
                    "{device}",
                ),
                expected_manifest="{output_dir}/stem_manifest.json",
            ),
        ),
    )


def test_executes_registered_candidate_and_writes_execution_manifest() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        executed_commands: list[list[str]] = []

        def fake_runner(command: list[str]):
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "stem_manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")

        manifest = executor.run_registered_model_candidate(
            model_id="demo_model",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=_demo_registry(root),
            command_runner=fake_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert manifest["contract"]["fallback_used"] is False
        assert manifest["model"]["model_id"] == "demo_model"
        assert manifest["candidate_manifest"]["path"].endswith("stem_manifest.json")
        assert executed_commands == [["python", "demo.py", str(source_path), str(root / "candidate"), "--device", "cuda"]]
        assert (root / "candidate" / "registry_candidate_execution_manifest.json").exists()


def test_unavailable_model_reports_missing_paths_without_running_command() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        executed_commands: list[list[str]] = []

        manifest = executor.run_registered_model_candidate(
            model_id="missing_model",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=_demo_registry(root),
            command_runner=lambda command: executed_commands.append(command),
            fail_on_unavailable=False,
        )

        assert manifest["status"] == "unavailable"
        assert manifest["reason"] == "missing_required_paths"
        assert manifest["availability"]["missing_paths"] == ["models/missing.ckpt"]
        assert manifest["contract"]["fallback_used"] is False
        assert executed_commands == []


def test_unavailable_model_reports_missing_python_modules_without_running_command() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, ModelRegistryEntry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        executed_commands: list[list[str]] = []
        registry = ModelRegistry(
            runtime_root=root,
            entries=(
                ModelRegistryEntry(
                    model_id="missing_python_module_model",
                    display_name="Missing Python module model",
                    family="test",
                    target_stems=("vocals",),
                    output_stems=("vocals",),
                    command_template=("python", "missing_module.py", "{source_audio_path}", "{output_dir}"),
                    expected_manifest="{output_dir}/stem_manifest.json",
                    required_python_modules=("definitely_missing_audio_dependency_for_test",),
                ),
            ),
        )

        manifest = executor.run_registered_model_candidate(
            model_id="missing_python_module_model",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=lambda command: executed_commands.append(command),
            fail_on_unavailable=False,
        )

        assert manifest["status"] == "unavailable"
        assert manifest["reason"] == "missing_required_python_modules"
        assert manifest["availability"]["missing_paths"] == []
        assert manifest["availability"]["missing_python_modules"] == [
            "definitely_missing_audio_dependency_for_test"
        ]
        assert manifest["contract"]["fallback_used"] is False
        assert executed_commands == []


def test_missing_expected_manifest_fails_after_command_without_fallback() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")

        with pytest.raises(RuntimeError, match="expected candidate manifest was not created"):
            executor.run_registered_model_candidate(
                model_id="demo_model",
                source_audio_path=source_path,
                output_dir=root / "candidate",
                registry=_demo_registry(root),
                command_runner=lambda command: None,
            )

        failure_manifest = root / "candidate" / "registry_candidate_execution_manifest.json"
        assert failure_manifest.exists()
        assert "expected_manifest_missing" in failure_manifest.read_text(encoding="utf-8")


def test_registered_multistem_instrumental_job_materializes_minus_target_manifest() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        sample_rate = 8000
        source_path = root / "source.wav"
        sf.write(source_path, np.zeros((8, 2), dtype=np.float32), sample_rate, subtype="FLOAT")

        drums = np.full((8, 2), 0.10, dtype=np.float32)
        bass = np.full((8, 2), 0.20, dtype=np.float32)
        other = np.full((8, 2), -0.05, dtype=np.float32)
        vocals = np.full((8, 2), 0.40, dtype=np.float32)

        def fake_runner(command: list[str]) -> None:
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            stem_arrays = {
                "vocals": vocals,
                "drums": drums,
                "bass": bass,
                "other": other,
            }
            stems = {}
            for stem, audio in stem_arrays.items():
                stem_path = output_dir / f"{stem}.wav"
                sf.write(stem_path, audio, sample_rate, subtype="FLOAT")
                stems[stem] = {"path": str(stem_path)}
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "demo_multistem_instrumental_model",
                        "contract": {
                            "full_track": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": stems,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="demo_multistem_instrumental_model",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=_demo_registry(root),
            command_runner=fake_runner,
            target_stem="minus",
        )

        assert manifest["status"] == "completed"
        assert manifest["candidate_manifest"]["target_stem"] == "minus"
        target_audio_path = Path(manifest["candidate_manifest"]["target_audio_path"])
        assert target_audio_path.exists()

        target_audio, observed_sample_rate = sf.read(target_audio_path, always_2d=True)
        assert observed_sample_rate == sample_rate
        np.testing.assert_allclose(target_audio, drums + bass + other, atol=1e-4)

        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["target_stem"] == "minus"
        assert candidate_manifest["stems"]["minus"]["path"] == str(target_audio_path)
        assert candidate_manifest["stems"]["instrumental"]["path"] == str(target_audio_path)
        assert candidate_manifest["target_stem_artifact"]["source"] == "sum_non_vocal_stems"
        assert candidate_manifest["delivery_allowed"] is False
        assert candidate_manifest["release_ready"] is False


def test_hq_full_stem_minus_target_skips_melody_export_step() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        executed_commands: list[list[str]] = []

        def fake_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            stems = {}
            for stem_name, value in {
                "vocals": 0.4,
                "drums": 0.1,
                "bass": 0.2,
                "other": 0.3,
            }.items():
                path = output_dir / f"{stem_name}.flac"
                sf.write(path, np.full((8, 2), value, dtype=np.float32), 44100)
                stems[stem_name] = {"path": str(path)}
            (output_dir / "stem_manifest.json").write_text(
                json.dumps({"model_name": "htdemucs_6s", "stems": stems}),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="demo_hq_full_stem_exporter",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=_demo_registry(root),
            command_runner=fake_runner,
            device="cuda",
            target_stem="minus",
        )

        assert manifest["status"] == "completed"
        assert "--no-melody-stem" in executed_commands[0]
        assert manifest["candidate_manifest"]["target_stem"] == "minus"
        assert manifest["candidate_manifest"]["target_source"] == "sum_non_vocal_stems"


def test_existing_instrumental_minus_target_preserves_original_stem_metadata() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, ModelRegistryEntry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(
                ModelRegistryEntry(
                    model_id="demo_instrumental_model",
                    display_name="Demo instrumental model",
                    family="test",
                    target_stems=("instrumental",),
                    output_stems=("vocals", "instrumental"),
                    command_template=("python", "demo_instrumental.py", "{source_audio_path}", "{output_dir}"),
                    expected_manifest="{output_dir}/stem_manifest.json",
                ),
            ),
        )

        def fake_runner(command: list[str]) -> None:
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path = output_dir / "vocals.flac"
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            vocals_path.write_text("stub vocals", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "demo_instrumental_model",
                        "stems": {
                            "vocals": {"path": str(vocals_path), "source": "residual_from_raw_minus_instrumental"},
                            "instrumental": {
                                "path": str(instrumental_path),
                                "source": "model_output",
                                "sample_rate": 48000,
                                "sha256": "demo-hash",
                            },
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        executor.run_registered_model_candidate(
            model_id="demo_instrumental_model",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_runner,
            target_stem="minus",
        )

        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["stems"]["minus"]["source"] == "existing_manifest_stem"
        assert candidate_manifest["stems"]["minus"]["source_stems"] == ["instrumental"]
        assert candidate_manifest["stems"]["instrumental"]["source"] == "model_output"
        assert candidate_manifest["stems"]["instrumental"]["sample_rate"] == 48000
        assert candidate_manifest["stems"]["instrumental"]["sha256"] == "demo-hash"


def test_command_runner_error_writes_execution_failed_manifest() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")

        def failing_runner(command: list[str]) -> None:
            raise RuntimeError("runner exploded")

        with pytest.raises(RuntimeError, match="runner exploded"):
            executor.run_registered_model_candidate(
                model_id="demo_model",
                source_audio_path=source_path,
                output_dir=root / "candidate",
                registry=_demo_registry(root),
                command_runner=failing_runner,
            )

        failure_manifest = root / "candidate" / "registry_candidate_execution_manifest.json"
        assert failure_manifest.exists()
        payload = json.loads(failure_manifest.read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["reason"] == "execution_failed"
        assert payload["candidate_manifest"]["exists"] is False
        assert payload["candidate_manifest"]["path"].endswith("stem_manifest.json")
        assert payload["error"]["type"] == "RuntimeError"
        assert payload["error"]["message"] == "runner exploded"
        assert payload["contract"]["fallback_used"] is False


def test_default_subprocess_runner_error_writes_stdout_and_stderr_logs() -> None:
    executor = _load_executor_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")

        with pytest.raises(Exception):
            executor.run_registered_model_candidate(
                model_id="failing_subprocess_model",
                source_audio_path=source_path,
                output_dir=root / "candidate",
                registry=_demo_registry(root),
            )

        failure_manifest = root / "candidate" / "registry_candidate_execution_manifest.json"
        payload = json.loads(failure_manifest.read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["reason"] == "execution_failed"
        assert payload["logs"]["stdout_path"].endswith("registered_candidate_stdout.log")
        assert payload["logs"]["stderr_path"].endswith("registered_candidate_stderr.log")
        assert "stdout marker" in Path(payload["logs"]["stdout_path"]).read_text(encoding="utf-8")
        assert "stderr marker" in Path(payload["logs"]["stderr_path"]).read_text(encoding="utf-8")


def test_melband_registry_executor_contract_writes_full_track_candidate_manifest() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("melband_roformer_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_melband_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "melband_roformer_vocals",
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="melband_roformer_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_melband_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_melband_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert "preview_window_sec" not in candidate_manifest
        assert candidate_manifest["model_id"] == "melband_roformer_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["stems"]["vocals"]["path"].endswith("vocals.flac")
        assert candidate_manifest["stems"]["instrumental"]["path"].endswith("instrumental.flac")


def test_becruily_karaoke_registry_executor_contract_preserves_checkpoint_metadata() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("becruily_karaoke_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_becruily_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "becruily_karaoke_vocals",
                        "model": {
                            "model_id": "becruily_karaoke_vocals",
                            "name": "Becruily MelBand RoFormer karaoke checkpoint",
                        },
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="becruily_karaoke_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_becruily_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_melband_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
                "--model-path",
                "/app/vendor/becruily_karaoke/mel_band_roformer_karaoke_becruily.ckpt",
                "--config-path",
                "/app/vendor/becruily_karaoke/config_karaoke_becruily.yaml",
                "--manifest-model-id",
                "becruily_karaoke_vocals",
                "--manifest-model-display-name",
                "Becruily MelBand RoFormer karaoke checkpoint",
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["model_id"] == "becruily_karaoke_vocals"
        assert candidate_manifest["model"]["model_id"] == "becruily_karaoke_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["contract"]["fallback_used"] is False


def test_becruily_instrumental_registry_executor_contract_preserves_checkpoint_metadata() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("becruily_instrumental_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_becruily_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "becruily_instrumental_vocals",
                        "model": {
                            "model_id": "becruily_instrumental_vocals",
                            "name": "Becruily MelBand RoFormer instrumental checkpoint",
                        },
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="becruily_instrumental_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_becruily_runner,
            device="cuda",
            target_stem="minus",
        )

        assert manifest["status"] == "completed"
        assert manifest["candidate_manifest"]["target_stem"] == "minus"
        assert manifest["candidate_manifest"]["target_source"] == "existing_manifest_stem"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_melband_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
                "--model-path",
                "/app/vendor/becruily_instrumental/mel_band_roformer_instrumental_becruily.ckpt",
                "--config-path",
                "/app/vendor/becruily_instrumental/config_instrumental_becruily.yaml",
                "--manifest-model-id",
                "becruily_instrumental_vocals",
                "--manifest-model-display-name",
                "Becruily MelBand RoFormer instrumental checkpoint",
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["model_id"] == "becruily_instrumental_vocals"
        assert candidate_manifest["model"]["model_id"] == "becruily_instrumental_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["contract"]["fallback_used"] is False
        assert candidate_manifest["target_stem"] == "minus"
        assert candidate_manifest["delivery_allowed"] is False
        assert candidate_manifest["release_ready"] is False


def test_mdx23c_registry_executor_contract_writes_full_track_candidate_manifest() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("mdx23c_vocals")
        registry = ModelRegistry(runtime_root=root, entries=(replace(default_entry, required_paths=()),))
        executed_commands: list[list[str]] = []

        def fake_mdx23c_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "mdx23c_vocals",
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="mdx23c_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_mdx23c_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_mdx23c_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert "preview_window_sec" not in candidate_manifest
        assert candidate_manifest["model_id"] == "mdx23c_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["stems"]["vocals"]["path"].endswith("vocals.flac")
        assert candidate_manifest["stems"]["instrumental"]["path"].endswith("instrumental.flac")


def test_bs_mamba2_registry_executor_contract_writes_full_track_candidate_manifest() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("bs_mamba2_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_bs_mamba2_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "bs_mamba2_vocals",
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="bs_mamba2_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_bs_mamba2_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_bs_mamba2_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert "preview_window_sec" not in candidate_manifest
        assert candidate_manifest["model_id"] == "bs_mamba2_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["stems"]["vocals"]["path"].endswith("vocals.flac")
        assert candidate_manifest["stems"]["instrumental"]["path"].endswith("instrumental.flac")


def test_audio_separator_bs_roformer_registry_executor_contract_writes_full_track_candidate_manifest() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("audio_separator_bs_roformer_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_audio_separator_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "audio_separator_bs_roformer_vocals",
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                            "target_reference_used_at_application": False,
                            "application_reference_free": True,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="audio_separator_bs_roformer_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_audio_separator_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_audio_separator_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
                "--model-filename",
                "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--model-file-dir",
                "/app/models/audio_separator",
                "--audio-separator-python",
                "/opt/audio-separator-venv/bin/python",
                "--manifest-model-id",
                "audio_separator_bs_roformer_vocals",
                "--manifest-model-display-name",
                "Audio Separator BS-RoFormer vocal checkpoint",
                "--require-cuda-provider",
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["model_id"] == "audio_separator_bs_roformer_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["contract"]["application_reference_free"] is True
        assert candidate_manifest["stems"]["vocals"]["path"].endswith("vocals.flac")
        assert candidate_manifest["stems"]["instrumental"]["path"].endswith("instrumental.flac")


def test_audio_separator_bs_roformer_viperx_vendor_config_executor_contract_writes_full_track_candidate_manifest() -> None:
    executor = _load_executor_module()
    from worker.model_registry import ModelRegistry, default_model_registry

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.flac"
        source_path.write_text("stub", encoding="utf-8")
        default_entry = default_model_registry().get("audio_separator_bs_roformer_viperx_1297_vendor_config_vocals")
        registry = ModelRegistry(
            runtime_root=root,
            entries=(replace(default_entry, required_paths=(), required_python_modules=()),),
        )
        executed_commands: list[list[str]] = []

        def fake_audio_separator_runner(command: list[str]) -> None:
            executed_commands.append(command)
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = output_dir / "vocals.flac"
            instrumental_path = output_dir / "instrumental.flac"
            vocals_path.write_text("stub vocals", encoding="utf-8")
            instrumental_path.write_text("stub instrumental", encoding="utf-8")
            (output_dir / "stem_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
                        "contract": {
                            "full_track": True,
                            "benchmark_compatible": True,
                            "fallback_used": False,
                            "hidden_inference": False,
                            "target_reference_used_at_application": False,
                            "application_reference_free": True,
                        },
                        "stems": {
                            "vocals": {"path": str(vocals_path)},
                            "instrumental": {"path": str(instrumental_path)},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        manifest = executor.run_registered_model_candidate(
            model_id="audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
            source_audio_path=source_path,
            output_dir=root / "candidate",
            registry=registry,
            command_runner=fake_audio_separator_runner,
            device="cuda",
        )

        assert manifest["status"] == "completed"
        assert executed_commands == [
            [
                "python",
                "/app/tools/export_audio_separator_vocal_full.py",
                str(source_path),
                str(root / "candidate"),
                "--model-filename",
                "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--model-file-dir",
                "/app/vendor/bs_roformer_viperx_1297",
                "--audio-separator-python",
                "/opt/audio-separator-venv/bin/python",
                "--manifest-model-id",
                "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals",
                "--manifest-model-display-name",
                "Audio Separator BS-RoFormer ViperX-1297 vendor config variant",
                "--require-cuda-provider",
            ]
        ]
        candidate_manifest = json.loads((root / "candidate" / "stem_manifest.json").read_text(encoding="utf-8"))
        assert candidate_manifest["model_id"] == "audio_separator_bs_roformer_viperx_1297_vendor_config_vocals"
        assert candidate_manifest["contract"]["full_track"] is True
        assert candidate_manifest["contract"]["application_reference_free"] is True
        assert candidate_manifest["stems"]["vocals"]["path"].endswith("vocals.flac")
        assert candidate_manifest["stems"]["instrumental"]["path"].endswith("instrumental.flac")
