from __future__ import annotations

import json
from pathlib import Path

from tools.build_macos_8gb_runtime_plan import evaluate_macos_runtime_plan, write_macos_runtime_plan


def _result(stdout: str = "", *, available: bool = True) -> dict[str, object]:
    return {
        "command": ["test"],
        "available": available,
        "reason": "" if available else "unavailable",
        "stdout": stdout,
        "stderr": "",
        "returncode": 0 if available else 1,
    }


def _base_checks(*, mps_available: bool, docker_model_runner: bool = False) -> dict[str, dict[str, object]]:
    return {
        "sw_vers": _result("15.5\n"),
        "uname_machine": _result("arm64\n"),
        "memory_bytes": _result(str(8 * 1024**3)),
        "display": _result("Apple M-series GPU\n"),
        "docker_version": _result("Docker version 27\n"),
        "docker_info": _result("Docker Desktop\n"),
        "docker_model_runner": _result("Docker Model Runner\n", available=docker_model_runner),
        "apple_container_cli": _result("container 0.1\n", available=True),
        "python_torch_mps": _result(
            json.dumps(
                {
                    "torch_imported": True,
                    "mps_built": True,
                    "mps_available": mps_available,
                    "torch_version": "2.9.0",
                }
            )
        ),
    }


def test_mps_ready_mac_uses_native_audio_probe_not_docker_gpu() -> None:
    manifest = evaluate_macos_runtime_plan(
        _base_checks(mps_available=True, docker_model_runner=True),
        platform_overrides={"system": "Darwin", "machine": "arm64"},
    )

    assert manifest["launch_status"] == "ready_for_native_mps_audio_probe"
    assert manifest["recommended_runtime"] == "native_mps_audio"
    assert manifest["docker_audio_gpu_supported"] is False
    assert manifest["runtime_modes"]["docker_model_runner"]["recommended_use"] == "optional_text_llm_controller_only"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "diagnostic_macos_runtime_plan_only"


def test_docker_model_runner_without_mps_still_blocks_audio_runtime() -> None:
    manifest = evaluate_macos_runtime_plan(
        _base_checks(mps_available=False, docker_model_runner=True),
        platform_overrides={"system": "Darwin", "machine": "arm64"},
    )

    assert manifest["launch_status"] == "blocked_macos_preflight"
    assert manifest["recommended_runtime"] == "blocked_missing_native_mps"
    assert "pytorch_mps_unavailable" in manifest["blocker_reasons"]
    assert manifest["runtime_modes"]["docker_model_runner"]["audio_separator_gpu_mode"] == "not_a_demucs_mdx_melband_scnet_runtime"
    assert manifest["delivery_allowed"] is False


def test_non_macos_host_blocks_even_if_torch_probe_claims_mps() -> None:
    checks = _base_checks(mps_available=True)
    checks["sw_vers"] = _result("", available=False)
    manifest = evaluate_macos_runtime_plan(
        checks,
        platform_overrides={"system": "Windows", "machine": "AMD64", "memory_bytes": 16 * 1024**3},
    )

    assert manifest["launch_status"] == "blocked_macos_preflight"
    assert "not_macos" in manifest["blocker_reasons"]
    assert "not_apple_silicon_arm64" in manifest["blocker_reasons"]
    assert manifest["delivery_allowed"] is False


def test_write_macos_runtime_plan_uses_short_artifact_names(tmp_path: Path) -> None:
    manifest = evaluate_macos_runtime_plan(
        _base_checks(mps_available=True),
        platform_overrides={"system": "Darwin", "machine": "arm64"},
    )
    written = write_macos_runtime_plan(tmp_path / "m8", manifest)

    assert (tmp_path / "m8" / "manifest.json").is_file()
    assert (tmp_path / "m8" / "summary.md").is_file()
    assert (tmp_path / "m8" / "progress.jsonl").is_file()
    assert Path(str(written["manifest_path"])).name == "manifest.json"
    assert Path(str(written["summary_md"])).name == "summary.md"
    assert all(len(part) <= 32 for part in ("manifest.json", "summary.md", "progress.jsonl"))
