from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "export_instrumental_resurrection_minus.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_instrumental_resurrection_minus",
        MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_stereo_wav(path: Path, frames: int = 4096, sample_rate: int = 44100) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    left = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    right = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")
    return audio


def _command_value(test_case: unittest.TestCase, command: list[str], option: str) -> str:
    test_case.assertEqual(command.count(option), 1)
    option_index = command.index(option)
    test_case.assertLess(option_index + 1, len(command))
    return command[option_index + 1]


def _assert_flag_once(test_case: unittest.TestCase, command: list[str], option: str) -> None:
    test_case.assertEqual(command.count(option), 1)


def _assert_common_inference_command(
    test_case: unittest.TestCase,
    command: list[str],
    *,
    cwd: Path,
    runtime_repo: Path,
    config_path: Path,
    checkpoint_path: Path,
    input_wav_name: str,
) -> Path:
    test_case.assertEqual(cwd, runtime_repo.resolve())
    test_case.assertEqual(command[0], sys.executable)
    test_case.assertEqual(command[1], "inference.py")
    test_case.assertEqual(_command_value(test_case, command, "--model_type"), "bs_roformer")
    test_case.assertEqual(_command_value(test_case, command, "--config_path"), str(config_path))
    test_case.assertEqual(_command_value(test_case, command, "--start_check_point"), str(checkpoint_path))
    test_case.assertEqual(_command_value(test_case, command, "--filename_template"), "{instr}")
    test_case.assertEqual(_command_value(test_case, command, "--pcm_type"), "PCM_24")
    _assert_flag_once(test_case, command, "--flac_file")
    _assert_flag_once(test_case, command, "--disable_detailed_pbar")

    input_folder = Path(_command_value(test_case, command, "--input_folder"))
    test_case.assertTrue(input_folder.is_dir())
    test_case.assertTrue(input_folder.name.startswith("irs_input_"))
    test_case.assertTrue((input_folder / input_wav_name).exists())

    store_dir = Path(_command_value(test_case, command, "--store_dir"))
    test_case.assertTrue(store_dir.is_dir())
    test_case.assertTrue(store_dir.name.startswith("irs_output_"))
    return store_dir


class TestExportInstrumentalResurrectionMinus(unittest.TestCase):
    def test_writes_artifact_contract(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            listen_dir = (temp_path / "listen").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text("training:\n  target_instrument: other\n", encoding="utf-8")
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            source_audio = _write_stereo_wav(source_path)

            captured: dict[str, object] = {}

            def fake_run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                captured["cwd"] = cwd
                store_dir = _assert_common_inference_command(
                    self,
                    command,
                    cwd=cwd,
                    runtime_repo=runtime_repo,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    input_wav_name="source.wav",
                )
                store_dir.mkdir(parents=True, exist_ok=True)
                predicted_other = (source_audio * 0.5).astype(np.float32)
                sf.write(str(store_dir / "other.wav"), predicted_other, 44100, subtype="PCM_24")
                return subprocess.CompletedProcess(command, 0, stdout="fake ok\n", stderr="")

            argv = [
                "export_instrumental_resurrection_minus.py",
                str(source_path),
                str(output_dir),
                "--runtime-repo",
                str(runtime_repo),
                "--config-path",
                str(config_path),
                "--model-path",
                str(checkpoint_path),
                "--device",
                "cuda:0",
                "--use-tta",
                "--listen-dir",
                str(listen_dir),
                "--listen-label",
                "candidate.flac",
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
                module.sys,
                "argv",
                argv,
            ), patch.object(module.os, "fsync") as fsync_mock:
                self.assertEqual(module.main(), 0)

            command = captured["command"]
            self.assertIsInstance(command, list)
            self.assertEqual(_command_value(self, command, "--device_ids"), "0")
            _assert_flag_once(self, command, "--use_tta")
            self.assertNotIn("--force_cpu", command)

            raw_ref_path = output_dir / "raw_ref.flac"
            minus_path = output_dir / "instrumental_resurrection_minus.flac"
            manifest_path = output_dir / "stem_manifest.json"
            progress_log_path = output_dir / "instrumental_resurrection_progress.jsonl"
            self.assertTrue(raw_ref_path.exists())
            self.assertTrue(minus_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(progress_log_path.exists())
            self.assertGreaterEqual(fsync_mock.call_count, 5)

            progress_events = [
                json.loads(line)
                for line in progress_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(progress_events), 5)
            self.assertEqual(progress_events[0]["type"], "instrumental_resurrection_stage")
            self.assertEqual(progress_events[0]["stage"], "execution_contract")
            self.assertEqual(progress_events[0]["metadata"]["route_name"], "export_instrumental_resurrection_minus")
            self.assertEqual(progress_events[0]["metadata"]["docker_runtime_required"], True)
            self.assertEqual(progress_events[0]["metadata"]["expected_acceleration_mode"], "gpu_inference")
            self.assertIn(
                "model_inference",
                {event["stage"] for event in progress_events},
            )

            minus_audio, minus_sample_rate = sf.read(str(minus_path), always_2d=True)
            self.assertEqual(minus_sample_rate, 44100)
            np.testing.assert_allclose(minus_audio, source_audio * 0.5, atol=2e-4)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_audio_path"], str(source_path.resolve()))
            self.assertEqual(manifest["target_stem"], "minus")
            self.assertEqual(manifest["target_audio_path"], str(minus_path))
            self.assertEqual(
                manifest["target_stem_artifact"],
                {
                    "target_stem": "minus",
                    "path": str(minus_path),
                    "source": "instrumental_resurrection_model_output",
                    "source_stems": ["other"],
                },
            )
            self.assertEqual(manifest["stems"]["minus"]["path"], str(minus_path))
            self.assertEqual(manifest["stems"]["minus"]["source"], "instrumental_resurrection_model_output")
            self.assertEqual(manifest["instrumental_resurrection_output_path"], str(minus_path))
            self.assertEqual(manifest["instrumental_resurrection_model_output_name"], "other.wav")
            self.assertEqual(manifest["artifacts"]["raw_ref_flac"], str(raw_ref_path))
            self.assertEqual(manifest["artifacts"]["instrumental_resurrection_minus_flac"], str(minus_path))
            self.assertEqual(manifest["model"]["config_path"], str(config_path))
            self.assertEqual(manifest["model"]["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(manifest["model"]["target_instrument"], "other")
            self.assertEqual(manifest["model"]["device"], "cuda:0")
            self.assertEqual(manifest["audio"]["sample_rate"], 44100)
            self.assertEqual(manifest["audio"]["frames"], source_audio.shape[0])
            self.assertEqual(
                manifest["execution_contract"]["progress_log_path"],
                str(progress_log_path),
            )
            self.assertEqual(
                manifest["execution_contract"]["expected_acceleration_mode"],
                "gpu_inference",
            )

            self.assertTrue((listen_dir / "00_raw_ref.flac").exists())
            self.assertTrue((listen_dir / "candidate.flac").exists())
            self.assertTrue((listen_dir / "00_README.txt").exists())
            candidate_map = json.loads((listen_dir / "candidate_map.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate_map["00_raw_ref.flac"], str(raw_ref_path))
            self.assertEqual(candidate_map["candidate.flac"], str(minus_path))

    def test_cpu_export_without_listen_dir_uses_force_cpu(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text("training:\n  target_instrument: other\n", encoding="utf-8")
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            source_audio = _write_stereo_wav(source_path)

            captured: dict[str, object] = {}

            def fake_run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                captured["cwd"] = cwd
                store_dir = _assert_common_inference_command(
                    self,
                    command,
                    cwd=cwd,
                    runtime_repo=runtime_repo,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    input_wav_name="source.wav",
                )
                store_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(store_dir / "other.wav"), source_audio * 0.25, 44100, subtype="PCM_24")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            argv = [
                "export_instrumental_resurrection_minus.py",
                str(source_path),
                str(output_dir),
                "--runtime-repo",
                str(runtime_repo),
                "--config-path",
                str(config_path),
                "--model-path",
                str(checkpoint_path),
                "--device",
                "cpu",
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
                module.sys,
                "argv",
                argv,
            ):
                self.assertEqual(module.main(), 0)

            command = captured["command"]
            self.assertIsInstance(command, list)
            _assert_flag_once(self, command, "--force_cpu")
            self.assertNotIn("--device_ids", command)
            self.assertNotIn("--use_tta", command)

            raw_ref_path = output_dir / "raw_ref.flac"
            minus_path = output_dir / "instrumental_resurrection_minus.flac"
            manifest_path = output_dir / "stem_manifest.json"
            progress_log_path = output_dir / "instrumental_resurrection_progress.jsonl"
            self.assertTrue(raw_ref_path.exists())
            self.assertTrue(minus_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(progress_log_path.exists())
            self.assertFalse((output_dir / "candidate_map.json").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["target_stem"], "minus")
            self.assertEqual(manifest["target_audio_path"], str(minus_path))
            self.assertEqual(manifest["target_stem_artifact"]["path"], str(minus_path))
            self.assertEqual(manifest["stems"]["minus"]["path"], str(minus_path))
            self.assertEqual(manifest["instrumental_resurrection_output_path"], str(minus_path))
            self.assertEqual(manifest["instrumental_resurrection_model_output_name"], "other.wav")
            self.assertEqual(manifest["model"]["device"], "cpu")
            self.assertEqual(manifest["audio"]["sample_rate"], 44100)
            self.assertEqual(manifest["audio"]["frames"], source_audio.shape[0])
            self.assertEqual(
                manifest["execution_contract"]["expected_acceleration_mode"],
                "cpu_diagnostic_inference",
            )

    def test_mps_export_lets_vendor_select_mps_without_device_ids(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text("training:\n  target_instrument: other\n", encoding="utf-8")
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            source_audio = _write_stereo_wav(source_path)

            captured: dict[str, object] = {}

            def fake_run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                captured["cwd"] = cwd
                store_dir = _assert_common_inference_command(
                    self,
                    command,
                    cwd=cwd,
                    runtime_repo=runtime_repo,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    input_wav_name="source.wav",
                )
                store_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(store_dir / "other.wav"), source_audio * 0.75, 44100, subtype="PCM_24")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            argv = [
                "export_instrumental_resurrection_minus.py",
                str(source_path),
                str(output_dir),
                "--runtime-repo",
                str(runtime_repo),
                "--config-path",
                str(config_path),
                "--model-path",
                str(checkpoint_path),
                "--device",
                "mps",
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
                module.sys,
                "argv",
                argv,
            ):
                self.assertEqual(module.main(), 0)

            command = captured["command"]
            self.assertIsInstance(command, list)
            self.assertNotIn("--force_cpu", command)
            self.assertNotIn("--device_ids", command)

            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["device"], "mps")
            self.assertEqual(
                manifest["execution_contract"]["expected_acceleration_mode"],
                "mps_inference",
            )

    def test_model_inference_failure_writes_blocked_manifest(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text("training:\n  target_instrument: other\n", encoding="utf-8")
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            _write_stereo_wav(source_path)

            def fake_run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    output="",
                    stderr="RuntimeError: Invalid buffer size",
                )

            argv = [
                "export_instrumental_resurrection_minus.py",
                str(source_path),
                str(output_dir),
                "--runtime-repo",
                str(runtime_repo),
                "--config-path",
                str(config_path),
                "--model-path",
                str(checkpoint_path),
                "--device",
                "mps",
            ]

            with patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
                module.sys,
                "argv",
                argv,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.main()

            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["delivery_allowed"], False)
            self.assertEqual(manifest["delivery_status"], "blocked_model_inference_failed")
            self.assertEqual(manifest["failed_checks"], ["instrumental_resurrection_model_inference_failed"])
            self.assertEqual(manifest["execution_contract"]["expected_acceleration_mode"], "mps_inference")
            self.assertIn("Invalid buffer size", manifest["error"])

    def test_resolve_bs_roformer_device_maps_cpu_and_cuda_values(self):
        module = _load_export_module()

        self.assertEqual(module._resolve_bs_roformer_device("cpu"), (True, []))
        self.assertEqual(module._resolve_bs_roformer_device("auto"), (False, ["0"]))
        self.assertEqual(module._resolve_bs_roformer_device("cuda:1"), (False, ["1"]))
        self.assertEqual(module._resolve_bs_roformer_device("cuda:0,2"), (False, ["0", "2"]))

    def test_resolve_bs_roformer_device_maps_mps_to_vendor_auto_device(self):
        module = _load_export_module()

        self.assertEqual(module._resolve_bs_roformer_device("mps"), (False, []))
