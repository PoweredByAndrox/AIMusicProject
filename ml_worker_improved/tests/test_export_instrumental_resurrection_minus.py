from __future__ import annotations

import importlib.util
import json
import socket
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
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  target_instrument: other\n",
                encoding="utf-8",
            )
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
            self.assertEqual(progress_events[0]["metadata"]["project_approved_runtime_required"], True)
            observed_runtime = progress_events[0]["metadata"]["observed_runtime"]
            self.assertEqual(
                progress_events[0]["metadata"]["docker_runtime_required"],
                observed_runtime["runtime_environment"] == "docker",
            )
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

            self.assertFalse(listen_dir.exists())
            self.assertEqual(manifest["requested_listen_dir"], str(listen_dir))
            self.assertEqual(manifest["listen_pack_status"], "skipped_candidate_not_accepted")
            self.assertIn(
                "skipped",
                {
                    event["status"]
                    for event in progress_events
                    if event["stage"] == "listen_pack"
                },
            )

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
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  target_instrument: other\n",
                encoding="utf-8",
            )
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
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  target_instrument: other\n",
                encoding="utf-8",
            )
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

            with patch.object(
                module,
                "_patch_standard_bs_roformer_mps_runtime",
                return_value=[],
            ), patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
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

    def test_resurrection_route_resamples_model_input_and_restores_source_layout(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  target_instrument: other\n",
                encoding="utf-8",
            )
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"fake checkpoint")
            source_audio = _write_stereo_wav(source_path, frames=4800, sample_rate=48000)

            def fake_run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                input_folder = Path(_command_value(self, command, "--input_folder"))
                model_input, model_sr = sf.read(str(input_folder / "source.wav"), always_2d=True)
                self.assertEqual(model_sr, 44100)
                self.assertEqual(model_input.shape, (4410, 2))
                store_dir = Path(_command_value(self, command, "--store_dir"))
                store_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(store_dir / "other.wav"), model_input * 0.5, model_sr, subtype="PCM_24")
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

            output_audio, output_sr = sf.read(
                str(output_dir / "instrumental_resurrection_minus.flac"),
                always_2d=True,
            )
            self.assertEqual(output_sr, 48000)
            self.assertEqual(output_audio.shape, source_audio.shape)
            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["audio"]["sample_rate_conversion"]["performed"])
            self.assertEqual(manifest["audio"]["model_input_sample_rate"], 44100)
            self.assertEqual(manifest["audio"]["model_input_frames"], 4410)

    def test_custom_architecture_uses_strict_mps_adapter_and_verified_provenance(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "audio:\n  chunk_size: 960000\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ninference:\n  batch_size: 2\n  num_overlap: 4\ntraining:\n  target_instrument: instrument\n  use_amp: true\n",
                encoding="utf-8",
            )
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"strict checkpoint")
            implementation_path = temp_path / "bs_roformer.py"
            implementation_path.write_text("class BSRoformer: pass\n", encoding="utf-8")
            source_audio = _write_stereo_wav(source_path)
            source_metadata_path = temp_path / "source.json"
            source_metadata_path.write_text(
                json.dumps(
                    {
                        "schema": "external_model_source.v1",
                        "status": "downloaded_diagnostic_only",
                        "production_allowed": False,
                        "license_status": "missing_upstream_metadata",
                        "files": [
                            {"file": path.name, "sha256": module._sha256(path)}
                            for path in (config_path, checkpoint_path, implementation_path)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            def fake_run_custom_architecture_in_process(**kwargs) -> None:
                captured.update(kwargs)
                self.assertEqual(kwargs["runtime_repo"], runtime_repo.resolve())
                self.assertEqual(kwargs["model_implementation_path"], implementation_path.resolve())
                runtime_config = Path(kwargs["config_path"])
                config = module.yaml.load(runtime_config.read_text(encoding="utf-8"), Loader=module.yaml.FullLoader)
                self.assertEqual(config["audio"]["chunk_size"], 529200)
                self.assertEqual(config["inference"]["batch_size"], 1)
                self.assertEqual(config["inference"]["num_overlap"], 1)
                store_dir = Path(kwargs["output_dir"])
                store_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(store_dir / "instrument.wav"), source_audio * 0.6, 44100, subtype="PCM_24")

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
                "--model-implementation-path",
                str(implementation_path),
                "--model-source-metadata-path",
                str(source_metadata_path),
                "--model-name",
                "bs_roformer_large_inst",
                "--target-instrument",
                "instrument",
                "--device",
                "mps",
                "--inference-chunk-size",
                "529200",
                "--inference-batch-size",
                "1",
                "--inference-num-overlap",
                "1",
            ]

            with patch.object(
                module,
                "_run_custom_architecture_in_process",
                side_effect=fake_run_custom_architecture_in_process,
            ), patch.object(module.sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            self.assertFalse(captured["use_tta"])
            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["name"], "bs_roformer_large_inst")
            self.assertTrue(manifest["model"]["strict_checkpoint_load"])
            self.assertEqual(manifest["model"]["model_implementation_path"], str(implementation_path.resolve()))
            self.assertEqual(manifest["model"]["inference_overrides"]["chunk_size"], 529200)
            self.assertEqual(manifest["model"]["precision_contract"]["mode"], "mps_autocast_float16")
            self.assertTrue(
                manifest["model"]["precision_contract"][
                    "mps_cuda_autocast_compatibility_requested"
                ]
            )
            self.assertTrue(
                manifest["execution_contract"]["precision_contract"][
                    "mps_cuda_autocast_compatibility_installed"
                ]
            )
            self.assertFalse(
                manifest["execution_contract"]["precision_contract"]["cpu_fallback_allowed"]
            )
            self.assertTrue(
                manifest["execution_contract"]["precision_contract"][
                    "requires_quality_equivalence_gate"
                ]
            )
            self.assertEqual(
                manifest["execution_contract"]["custom_adapter_execution_mode"],
                "in_process_isolated_exporter",
            )
            self.assertEqual(manifest["model_source"]["status"], "downloaded_diagnostic_only")
            self.assertFalse(manifest["model_production_allowed"])
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["delivery_status"], "candidate_only")
            self.assertEqual(manifest["presentation_status"], "blocked_until_judged")

    def test_precision_contract_does_not_claim_installation_before_adapter_success(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  use_amp: true\n",
                encoding="utf-8",
            )
            contract = module._build_precision_contract(
                config_path,
                device="mps",
                model_implementation_path=temp_path / "bs_roformer.py",
            )

        self.assertEqual(contract["mode"], "pending_adapter_initialization")
        self.assertEqual(contract["requested_mode"], "mps_autocast_float16")
        self.assertTrue(contract["mps_cuda_autocast_compatibility_requested"])
        self.assertFalse(contract["mps_cuda_autocast_compatibility_installed"])
        self.assertIsNone(contract["autocast_device_type"])

        module._mark_precision_contract_installed(contract)

        self.assertEqual(contract["mode"], "mps_autocast_float16")
        self.assertTrue(contract["mps_cuda_autocast_compatibility_installed"])
        self.assertEqual(contract["autocast_device_type"], "mps")

    def test_resident_worker_route_reuses_export_contract_without_cold_fallback(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.wav"
            output_dir = (temp_path / "out").resolve()
            runtime_repo = temp_path / "runtime_repo"
            runtime_repo.mkdir()
            (runtime_repo / "inference.py").write_text("# fake vendor entrypoint\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "audio:\n  chunk_size: 529200\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ninference:\n  batch_size: 1\n  num_overlap: 1\ntraining:\n  target_instrument: instrument\n  use_amp: true\n",
                encoding="utf-8",
            )
            checkpoint_path = temp_path / "model.ckpt"
            checkpoint_path.write_bytes(b"strict checkpoint")
            implementation_path = temp_path / "bs_roformer.py"
            implementation_path.write_text("class BSRoformer: pass\n", encoding="utf-8")
            source_audio = _write_stereo_wav(source_path)
            captured: dict[str, object] = {}

            def fake_resident(**kwargs):
                captured.update(kwargs)
                store_dir = Path(kwargs["output_dir"])
                store_dir.mkdir(parents=True, exist_ok=True)
                output_path = store_dir / "instrument.wav"
                sf.write(str(output_path), source_audio * 0.6, 44100, subtype="PCM_24")
                return {
                    "schema": "mac_mps_large_inst_response.v1",
                    "status": "ok",
                    "request_id": "lir001",
                    "device": "mps",
                    "cpu_fallback_allowed": False,
                    "delivery_allowed": False,
                    "model_load_count": 1,
                    "model_load_elapsed_sec": 2.0,
                    "request_count": 4,
                    "elapsed_sec": 7.5,
                    "output_path": str(output_path),
                    "output_sha256": module._sha256(output_path),
                    "output_contract": {"output_scope": "disposable_scratch"},
                }

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
                "--model-implementation-path",
                str(implementation_path),
                "--model-name",
                "bs_roformer_large_inst",
                "--target-instrument",
                "instrument",
                "--device",
                "mps",
                "--resident-worker-socket",
                "/tmp/aimusic_li.sock",
            ]

            with patch.object(
                module,
                "_run_resident_custom_architecture",
                side_effect=fake_resident,
            ), patch.object(
                module,
                "_run_custom_architecture_in_process",
                side_effect=AssertionError("cold custom adapter must not run"),
            ), patch.object(module.sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            self.assertEqual(Path(captured["source_audio_path"]).suffix, ".wav")
            manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
            execution = manifest["execution_contract"]
            self.assertEqual(execution["custom_adapter_execution_mode"], "resident_single_model_worker")
            self.assertTrue(execution["resident_worker_requested"])
            self.assertFalse(execution["resident_worker_fallback_allowed"])
            self.assertEqual(execution["resident_worker_evidence"]["model_load_count"], 1)
            self.assertEqual(execution["resident_worker_evidence"]["request_count"], 4)
            self.assertTrue(execution["resident_worker_evidence"]["scratch_cleanup_verified"])
            self.assertFalse(manifest["delivery_allowed"])
            self.assertEqual(manifest["delivery_status"], "candidate_only")

    def test_resident_helper_verifies_model_fingerprint_and_output_bytes(self):
        module = _load_export_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            socket_path = root / "worker.sock"
            server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_socket.bind(str(socket_path))
            source_path = root / "source.wav"
            source_path.write_bytes(b"source")
            output_dir = root / "out"
            output_dir.mkdir()
            output_path = output_dir / "instrument.wav"
            output_path.write_bytes(b"resident output")
            runtime_config_path = root / "cfg.yaml"
            checkpoint_path = root / "model.ckpt"
            implementation_path = root / "bs_roformer.py"
            runtime_config_path.write_bytes(b"config")
            checkpoint_path.write_bytes(b"checkpoint")
            implementation_path.write_bytes(b"implementation")

            def response_for(payload, *, output_sha256=None, fingerprint=None):
                return {
                    "schema": "mac_mps_large_inst_response.v1",
                    "status": "ok",
                    "request_id": payload["request_id"],
                    "model_id": "bs_roformer_large_inst",
                    "model_fingerprint": fingerprint or payload["expected_model_fingerprint"],
                    "device": "mps",
                    "cpu_fallback_allowed": False,
                    "delivery_allowed": False,
                    "model_load_count": 1,
                    "output_path": str(output_path),
                    "output_sha256": output_sha256 or module._sha256(output_path),
                    "output_contract": {
                        "source_scope": "disposable_scratch",
                        "output_scope": "disposable_scratch",
                    },
                }

            captured: dict[str, object] = {}

            def fake_send(_socket_path, payload, _timeout):
                captured["payload"] = payload
                return response_for(payload)

            try:
                with patch.object(module, "send_request", side_effect=fake_send):
                    result = module._run_resident_custom_architecture(
                        socket_path=socket_path,
                        source_audio_path=source_path,
                        output_dir=output_dir,
                        model_id="bs_roformer_large_inst",
                        runtime_config_path=runtime_config_path,
                        checkpoint_path=checkpoint_path,
                        implementation_path=implementation_path,
                        use_tta=False,
                        timeout_sec=10.0,
                    )
                self.assertEqual(result["status"], "ok")
                payload = captured["payload"]
                self.assertEqual(
                    payload["expected_model_fingerprint"]["config_sha256"],
                    module._sha256(runtime_config_path),
                )

                def wrong_hash(_socket_path, payload, _timeout):
                    return response_for(payload, output_sha256="bad")

                with patch.object(module, "send_request", side_effect=wrong_hash):
                    with self.assertRaisesRegex(subprocess.CalledProcessError, "exit status 76"):
                        module._run_resident_custom_architecture(
                            socket_path=socket_path,
                            source_audio_path=source_path,
                            output_dir=output_dir,
                            model_id="bs_roformer_large_inst",
                            runtime_config_path=runtime_config_path,
                            checkpoint_path=checkpoint_path,
                            implementation_path=implementation_path,
                            use_tta=False,
                            timeout_sec=10.0,
                        )
            finally:
                server_socket.close()

    def test_observed_runtime_distinguishes_native_mps_from_policy_container(self):
        module = _load_export_module()
        policy = {
            "runtime_environment": "native_mac_cpu_dsp",
            "container_runtime": "apple_container",
            "native_runtime_allowed": True,
        }

        with patch.object(module, "cli_runtime_contract", return_value=policy):
            contract = module._observed_runtime_contract("mps")

        self.assertEqual(contract["runtime_environment"], "native_mac_mps_diagnostic")
        self.assertEqual(contract["execution_surface"], "native_macos")
        self.assertIsNone(contract["container_runtime"])
        self.assertEqual(contract["policy_container_runtime"], "apple_container")
        self.assertEqual(contract["accelerator"], "mps")

    def test_in_process_adapter_maps_failure_to_called_process_error(self):
        module = _load_export_module()

        def fail_adapter(_args):
            raise RuntimeError("synthetic adapter failure")

        fake_adapter_module = type(
            "FakeAdapterModule",
            (),
            {"run_custom_architecture": staticmethod(fail_adapter)},
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules,
            {"run_bs_roformer_custom_impl": fake_adapter_module},
        ):
            temp_path = Path(temp_dir)
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                module._run_custom_architecture_in_process(
                    runtime_repo=temp_path,
                    model_implementation_path=temp_path / "bs_roformer.py",
                    config_path=temp_path / "config.yaml",
                    checkpoint_path=temp_path / "model.ckpt",
                    input_dir=temp_path,
                    output_dir=temp_path / "out",
                    use_tta=False,
                )

        self.assertIn("RuntimeError: synthetic adapter failure", str(raised.exception.output))

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
            config_path.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  sample_rate: 44100\ntraining:\n  target_instrument: other\n",
                encoding="utf-8",
            )
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

            with patch.object(
                module,
                "_patch_standard_bs_roformer_mps_runtime",
                return_value=[],
            ), patch.object(module, "_run_command", side_effect=fake_run_command), patch.object(
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

    def test_standard_bs_roformer_mps_patch_uses_real_imag_zero_dc_and_forbids_cpu_istft(self):
        module = _load_export_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_repo = Path(temp_dir)
            implementation_path = runtime_repo / module.BS_ROFORMER_RUNTIME_RELATIVE_PATH
            implementation_path.parent.mkdir(parents=True)
            implementation_path.write_text(
                module.BS_ROFORMER_COMPLEX_ZERO_DC_ORIGINAL
                + module.BS_ROFORMER_ISTFT_CPU_FALLBACK_ORIGINAL,
                encoding="utf-8",
            )

            patches = module._patch_standard_bs_roformer_mps_runtime(
                runtime_repo,
                device="mps",
                model_type="bs_roformer",
                custom_implementation=False,
            )
            patched = implementation_path.read_text(encoding="utf-8")

            self.assertEqual(patches[0]["status"], "applied")
            self.assertIn("stft_repr.real.index_fill", patched)
            self.assertIn("stft_repr.imag.index_fill", patched)
            self.assertIn(module.BS_ROFORMER_ISTFT_NO_FALLBACK, patched)
            self.assertNotIn("stft_repr.cpu() if x_is_mps", patched)
            self.assertFalse(patches[0]["cpu_fallback_allowed"])

            repeat = module._patch_standard_bs_roformer_mps_runtime(
                runtime_repo,
                device="mps",
                model_type="bs_roformer",
                custom_implementation=False,
            )
            self.assertEqual(repeat[0]["status"], "already_present")
            self.assertEqual(repeat[0]["pre_patch_sha256"], repeat[0]["post_patch_sha256"])
