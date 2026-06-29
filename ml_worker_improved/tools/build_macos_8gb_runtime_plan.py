"""Build a fail-closed macOS 8GB audio runtime preflight manifest.

This tool does not run separation and does not claim that Docker exposes the
Apple GPU to the existing CUDA separator routes. It records whether a remote
Mac is viable for a native PyTorch MPS audio probe and keeps Docker/Apple
container runtimes as orchestration-only until proven otherwise on that host.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "macos_8gb_audio_runtime_plan.v1"
PRESENTATION_STATUS = "diagnostic_macos_runtime_plan_only"
MIN_UNIFIED_MEMORY_BYTES = 8 * 1024**3


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_text_command(command: list[str], *, timeout_sec: float = 10.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "reason": f"{command[0]}_cli_unavailable",
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "available": False,
            "reason": f"{command[0]}_failed",
            "stdout": "",
            "stderr": str(exc)[:1000],
            "returncode": None,
        }
    return {
        "command": command,
        "available": completed.returncode == 0,
        "reason": "" if completed.returncode == 0 else f"{command[0]}_failed",
        "stdout": completed.stdout[:8000],
        "stderr": completed.stderr[:1000],
        "returncode": completed.returncode,
    }


def _parse_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _parse_json_stdout(result: Mapping[str, Any]) -> dict[str, Any]:
    if not result.get("available"):
        return {}
    try:
        parsed = json.loads(str(result.get("stdout") or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def collect_macos_runtime_checks() -> dict[str, dict[str, Any]]:
    torch_probe = [
        "python3",
        "-c",
        (
            "import json\n"
            "try:\n"
            " import torch\n"
            " payload={'torch_imported': True,"
            "'mps_built': bool(torch.backends.mps.is_built()),"
            "'mps_available': bool(torch.backends.mps.is_available()),"
            "'torch_version': str(torch.__version__)}\n"
            "except Exception as exc:\n"
            " payload={'torch_imported': False,'mps_built': False,'mps_available': False,'error': str(exc)[:500]}\n"
            "print(json.dumps(payload, sort_keys=True))\n"
        ),
    ]
    return {
        "sw_vers": _run_text_command(["sw_vers", "-productVersion"]),
        "uname_machine": _run_text_command(["uname", "-m"]),
        "memory_bytes": _run_text_command(["sysctl", "-n", "hw.memsize"]),
        "display": _run_text_command(["system_profiler", "SPDisplaysDataType"], timeout_sec=20.0),
        "docker_version": _run_text_command(["docker", "version"], timeout_sec=20.0),
        "docker_info": _run_text_command(["docker", "info"], timeout_sec=20.0),
        "docker_model_runner": _run_text_command(["docker", "model", "version"], timeout_sec=20.0),
        "apple_container_cli": _run_text_command(["container", "--version"], timeout_sec=20.0),
        "python_torch_mps": _run_text_command(torch_probe, timeout_sec=20.0),
    }


def evaluate_macos_runtime_plan(
    checks: Mapping[str, Mapping[str, Any]],
    *,
    platform_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = dict(platform_overrides or {})
    system_name = str(overrides.get("system") or platform.system())
    machine = str(overrides.get("machine") or platform.machine())
    sw_vers = str(checks.get("sw_vers", {}).get("stdout") or "").strip()
    machine_override = overrides.get("machine")
    uname_machine = str(
        overrides.get("uname_machine")
        or (machine_override if machine_override is not None else checks.get("uname_machine", {}).get("stdout"))
        or machine
    ).strip() or machine
    memory_bytes = _parse_int(checks.get("memory_bytes", {}).get("stdout") or overrides.get("memory_bytes") or 0)
    if not memory_bytes:
        memory_bytes = _parse_int(overrides.get("memory_bytes") or 0)

    torch_mps = _parse_json_stdout(checks.get("python_torch_mps", {}))
    torch_imported = bool(torch_mps.get("torch_imported", False))
    mps_built = bool(torch_mps.get("mps_built", False))
    mps_available = bool(torch_mps.get("mps_available", False))

    is_macos = system_name == "Darwin" or bool(sw_vers)
    is_arm64 = uname_machine in {"arm64", "aarch64"} or machine in {"arm64", "aarch64"}
    enough_memory = memory_bytes >= MIN_UNIFIED_MEMORY_BYTES if memory_bytes else False
    docker_available = bool(checks.get("docker_version", {}).get("available", False))
    docker_model_runner_available = bool(checks.get("docker_model_runner", {}).get("available", False))
    apple_container_available = bool(checks.get("apple_container_cli", {}).get("available", False))

    blockers: list[str] = []
    if not is_macos:
        blockers.append("not_macos")
    if not is_arm64:
        blockers.append("not_apple_silicon_arm64")
    if not enough_memory:
        blockers.append("unified_memory_below_8gib_or_unknown")
    if not torch_imported:
        blockers.append("pytorch_not_importable")
    elif not mps_built:
        blockers.append("pytorch_not_built_with_mps")
    elif not mps_available:
        blockers.append("pytorch_mps_unavailable")

    native_mps_audio_ready = is_macos and is_arm64 and enough_memory and mps_available
    recommended_runtime = "native_mps_audio" if native_mps_audio_ready else "blocked_missing_native_mps"
    launch_status = "ready_for_native_mps_audio_probe" if native_mps_audio_ready else "blocked_macos_preflight"

    return {
        "schema": SCHEMA,
        "host": {
            "system": system_name,
            "machine": machine,
            "uname_machine": uname_machine,
            "macos_version": sw_vers,
            "memory_bytes": memory_bytes,
            "memory_gib": round(float(memory_bytes) / float(1024**3), 3) if memory_bytes else 0.0,
        },
        "runtime_checks": {key: dict(value) for key, value in checks.items()},
        "torch_mps": {
            "torch_imported": torch_imported,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "torch_version": str(torch_mps.get("torch_version") or ""),
            "error": str(torch_mps.get("error") or ""),
        },
        "runtime_modes": {
            "docker_desktop": {
                "available": docker_available,
                "audio_separator_gpu_mode": "unsupported_until_host_mps_or_cuda_is_verified",
                "recommended_use": "orchestration_and_cpu_diagnostics_only",
            },
            "docker_model_runner": {
                "available": docker_model_runner_available,
                "audio_separator_gpu_mode": "not_a_demucs_mdx_melband_scnet_runtime",
                "recommended_use": "optional_text_llm_controller_only",
            },
            "apple_container_cli": {
                "available": apple_container_available,
                "audio_separator_gpu_mode": "unsupported_until_pytorch_mps_inside_container_is_verified",
                "recommended_use": "orchestration_experiment_only",
            },
            "native_pytorch_mps": {
                "available": native_mps_audio_ready,
                "audio_separator_gpu_mode": "candidate_for_mac_8gb_probe",
                "recommended_use": "single_model_audio_inference_probe",
            },
        },
        "docker_audio_gpu_supported": False,
        "docker_audio_gpu_reason": "existing_audio_routes_are_cuda_container_routes; macos_audio_gpu_requires_native_mps_or_explicit_container_mps_proof",
        "recommended_runtime": recommended_runtime,
        "launch_status": launch_status,
        "blocker_reasons": blockers,
        "route_constraints": {
            "max_resident_audio_models": 1,
            "route_budget_profile": "fast",
            "disable_text_vllm_for_audio": True,
            "disable_querysep_long_tail": True,
            "no_parallel_separator_workers": True,
            "require_progress_jsonl": True,
            "require_fail_closed_no_cpu_fallback": True,
            "recommended_max_candidates": 3,
        },
        "recommended_next_commands": [
            "python3 ml_worker_improved/tools/build_macos_8gb_runtime_plan.py --out artifacts/stem_eval/macos8_preflight",
            "python3 - <<'PY'\nimport torch\nprint(torch.backends.mps.is_available())\nPY",
        ],
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": PRESENTATION_STATUS,
        "listen_pack": {},
    }


def _summary_markdown(manifest: Mapping[str, Any]) -> str:
    host = dict(manifest.get("host") or {})
    torch_mps = dict(manifest.get("torch_mps") or {})
    blockers = [str(item) for item in manifest.get("blocker_reasons") or []]
    lines = [
        "# macOS 8GB Audio Runtime Plan",
        "",
        f"Launch status: {manifest.get('launch_status', 'unknown')}",
        f"Recommended runtime: {manifest.get('recommended_runtime', 'unknown')}",
        f"Host: {host.get('system', '')} {host.get('macos_version', '')} {host.get('uname_machine', '')}",
        f"Unified memory GiB: {host.get('memory_gib', 0)}",
        f"PyTorch MPS available: {str(bool(torch_mps.get('mps_available'))).lower()}",
        f"Docker audio GPU supported: {str(bool(manifest.get('docker_audio_gpu_supported'))).lower()}",
        f"Blockers: {', '.join(blockers) if blockers else 'none'}",
        f"Delivery allowed: {str(bool(manifest.get('delivery_allowed'))).lower()}",
        f"Presentation status: {manifest.get('presentation_status', '')}",
        "",
        "Route constraints:",
    ]
    for key, value in dict(manifest.get("route_constraints") or {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def write_macos_runtime_plan(output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = dict(manifest)
    manifest_out["output_dir"] = str(output_dir)
    manifest_out["manifest_path"] = str(output_dir / "manifest.json")
    manifest_out["summary_md"] = str(output_dir / "summary.md")
    _write_json(output_dir / "manifest.json", manifest_out)
    (output_dir / "summary.md").write_text(_summary_markdown(manifest_out), encoding="utf-8")
    _append_jsonl(
        output_dir / "progress.jsonl",
        {
            "event": "end",
            "elapsed_sec": 0.0,
            "launch_status": manifest_out["launch_status"],
            "recommended_runtime": manifest_out["recommended_runtime"],
            "gpu_inference_skipped": True,
            "reason": "diagnostic_macos_runtime_plan_only",
        },
    )
    return manifest_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--echo-progress", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "progress.jsonl"
    start_event = {
        "event": "start",
        "elapsed_sec": 0.0,
        "gpu_inference_skipped": True,
        "reason": "diagnostic_macos_runtime_plan_only",
    }
    _append_jsonl(progress_path, start_event)
    if args.echo_progress:
        print(json.dumps(start_event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
    checks = collect_macos_runtime_checks()
    collect_event = {
        "event": "checks_collected",
        "elapsed_sec": round(time.monotonic() - started, 6),
        "gpu_inference_skipped": True,
    }
    _append_jsonl(progress_path, collect_event)
    if args.echo_progress:
        print(json.dumps(collect_event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
    manifest = write_macos_runtime_plan(args.out, evaluate_macos_runtime_plan(checks))
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": manifest["manifest_path"],
                "launch_status": manifest["launch_status"],
                "recommended_runtime": manifest["recommended_runtime"],
                "delivery_allowed": manifest["delivery_allowed"],
                "blocker_reasons": manifest["blocker_reasons"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
