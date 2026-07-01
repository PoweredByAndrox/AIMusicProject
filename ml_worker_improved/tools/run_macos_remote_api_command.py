"""Run a macOS remote command and force its artifacts into the shared folder.

The Mac side writes all output files under the provided mounted artifact root,
for example ``/Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/c001`` which maps to
``D:\\AIMusicProject\\artifacts\\mac_api\\c001`` on the Windows host. The
remote script probes the mount before creating the output directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence

TOOLS_PARENT = Path(__file__).resolve().parents[1]
if str(TOOLS_PARENT) not in sys.path:
    sys.path.insert(0, str(TOOLS_PARENT))

from tools.mac_windows_mount_preflight import (  # noqa: E402
    ALLOWED_PROJECT_ARTIFACT_ROOTS,
    DEFAULT_WINDOWS_PROJECT_ROOT,
    PROBE_DIR_NAME,
    SCHEMA as MOUNT_PREFLIGHT_SCHEMA,
)


SCHEMA = "macos_remote_api_command.v1"
PRESENTATION_STATUS = "diagnostic_macos_remote_api_only"
SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


def validate_artifact_subdir(value: str) -> str:
    raw_value = str(value or "")
    if "\\" in raw_value:
        raise ValueError("unsafe_artifact_path_separator")
    normalized = raw_value.strip("/")
    if not normalized:
        raise ValueError("artifact_subdir_empty")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError("artifact_subdir_empty")
    for part in parts:
        if part in {".", ".."} or not SAFE_SEGMENT_RE.fullmatch(part):
            raise ValueError(f"unsafe_artifact_path_segment:{part}")
    return "/".join(parts)


def _quote_posix(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _windows_join(root: str, parts: Sequence[str]) -> str:
    normalized = str(root).rstrip("\\/")
    suffix = "\\".join(str(part).strip("\\/") for part in parts if str(part).strip("\\/"))
    return f"{normalized}\\{suffix}" if suffix else normalized


def build_remote_mount_preflight(
    *,
    mac_windows_project_mount: str | None,
    mac_output_path: str,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
) -> dict[str, Any]:
    if not mac_windows_project_mount:
        return {
            "schema": MOUNT_PREFLIGHT_SCHEMA,
            "preflight_status": "blocked",
            "blocked_reason": "mac_windows_project_mount_missing",
            "can_write_route_artifacts": False,
            "mac_windows_project_mount": "",
            "mac_output_path": str(mac_output_path),
            "windows_project_root": str(windows_project_root),
            "windows_output_path": "",
            "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
            "probe_status": "not_run",
        }
    mount = PurePosixPath(str(mac_windows_project_mount).rstrip("/"))
    output = PurePosixPath(str(mac_output_path).rstrip("/"))
    try:
        relative = output.relative_to(mount)
    except ValueError:
        return {
            "schema": MOUNT_PREFLIGHT_SCHEMA,
            "preflight_status": "blocked",
            "blocked_reason": "mac_output_path_outside_windows_project_mount",
            "can_write_route_artifacts": False,
            "mac_windows_project_mount": str(mount),
            "mac_output_path": str(output),
            "windows_project_root": str(windows_project_root),
            "windows_output_path": "",
            "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
            "probe_status": "not_run",
        }
    if not relative.parts or relative.parts[0] not in ALLOWED_PROJECT_ARTIFACT_ROOTS:
        return {
            "schema": MOUNT_PREFLIGHT_SCHEMA,
            "preflight_status": "blocked",
            "blocked_reason": "mac_output_path_not_in_allowed_project_artifact_root",
            "can_write_route_artifacts": False,
            "mac_windows_project_mount": str(mount),
            "mac_output_path": str(output),
            "windows_project_root": str(windows_project_root),
            "windows_output_path": "",
            "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
            "relative_output_path": relative.as_posix(),
            "probe_status": "not_run",
        }
    return {
        "schema": MOUNT_PREFLIGHT_SCHEMA,
        "preflight_status": "remote_probe_required",
        "blocked_reason": "",
        "can_write_route_artifacts": False,
        "mac_windows_project_mount": str(mount),
        "mac_output_path": str(output),
        "windows_project_root": str(windows_project_root),
        "windows_output_path": _windows_join(windows_project_root, relative.parts),
        "allowed_project_artifact_roots": list(ALLOWED_PROJECT_ARTIFACT_ROOTS),
        "relative_output_path": relative.as_posix(),
        "probe_status": "remote_write_read_delete_pending",
    }


def build_remote_script(
    *,
    command: str,
    mac_artifact_root: str,
    mac_windows_project_mount: str,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
    artifact_subdir: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    safe_subdir = validate_artifact_subdir(artifact_subdir)
    output_dir = str(PurePosixPath(str(mac_artifact_root).rstrip("/")) / safe_subdir)
    mount_preflight = build_remote_mount_preflight(
        mac_windows_project_mount=mac_windows_project_mount,
        mac_output_path=output_dir,
        windows_project_root=windows_project_root,
    )
    if mount_preflight["preflight_status"] == "blocked":
        raise ValueError(str(mount_preflight["blocked_reason"]))
    command_script_b64 = _b64(str(command) + "\n")
    metadata_b64 = _b64(
        json.dumps({**dict(metadata or {}), "mount_preflight_expected": mount_preflight}, ensure_ascii=False, sort_keys=True)
    )
    return f"""#!/bin/sh
set -u
OUT={_quote_posix(output_dir)}
MOUNT={_quote_posix(str(mac_windows_project_mount).rstrip("/"))}
WIN_OUTPUT={_quote_posix(str(mount_preflight["windows_output_path"]))}
CMD_B64={_quote_posix(command_script_b64)}
META_B64={_quote_posix(metadata_b64)}
PROBE_DIR="$MOUNT/{PROBE_DIR_NAME}"
PROBE="$PROBE_DIR/probe_$$.txt"
PROBE_PAYLOAD="{MOUNT_PREFLIGHT_SCHEMA}:$$:$(date +%s)"
if mkdir -p "$PROBE_DIR" \\
  && printf '%s' "$PROBE_PAYLOAD" > "$PROBE" \\
  && [ "$(cat "$PROBE" 2>/dev/null)" = "$PROBE_PAYLOAD" ]; then
  rm -f "$PROBE"
  rmdir "$PROBE_DIR" 2>/dev/null || true
  MOUNT_PREFLIGHT_STATUS="passed"
  MOUNT_PROBE_STATUS="write_read_delete_passed"
else
  rm -f "$PROBE" 2>/dev/null || true
  printf '%s\\n' '{{"schema":"{MOUNT_PREFLIGHT_SCHEMA}","preflight_status":"blocked","blocked_reason":"mac_windows_project_mount_probe_failed","can_write_route_artifacts":false}}' >&2
  exit 78
fi
mkdir -p "$OUT"
PROGRESS="$OUT/progress.jsonl"
STDOUT="$OUT/stdout.txt"
STDERR="$OUT/stderr.txt"
CMD="$OUT/command.sh"
printf '%s\\n' '{{"event":"start","elapsed_sec":0.0,"gpu_inference_skipped":true,"reason":"macos_remote_api_command"}}' > "$PROGRESS"
if printf '%s' "$CMD_B64" | base64 -D > "$CMD" 2>/dev/null; then
  :
else
  printf '%s' "$CMD_B64" | base64 -d > "$CMD"
fi
chmod 700 "$CMD"
START="$(date +%s)"
printf '%s\\n' '{{"event":"command_start","elapsed_sec":0.0,"gpu_inference_skipped":true}}' >> "$PROGRESS"
sh "$CMD" > "$STDOUT" 2> "$STDERR"
RC="$?"
END="$(date +%s)"
python3 - "$OUT" "$RC" "$START" "$END" "$CMD_B64" "$META_B64" "$MOUNT" "$WIN_OUTPUT" "$MOUNT_PREFLIGHT_STATUS" "$MOUNT_PROBE_STATUS" <<'PY'
import base64
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rc = int(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])
command = base64.b64decode(sys.argv[5]).decode("utf-8", errors="replace").strip()
metadata = json.loads(base64.b64decode(sys.argv[6]).decode("utf-8") or "{{}}")
mount = sys.argv[7]
windows_output_path = sys.argv[8]
mount_preflight_status = sys.argv[9]
mount_probe_status = sys.argv[10]
stdout_path = out / "stdout.txt"
stderr_path = out / "stderr.txt"
stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
mount_preflight = {{
    "schema": "{MOUNT_PREFLIGHT_SCHEMA}",
    "preflight_status": mount_preflight_status,
    "blocked_reason": "",
    "can_write_route_artifacts": mount_preflight_status == "passed",
    "mac_windows_project_mount": mount,
    "mac_output_path": str(out),
    "windows_project_root": "{windows_project_root}",
    "windows_output_path": windows_output_path,
    "probe_status": mount_probe_status,
}}
manifest = {{
    "schema": "{SCHEMA}",
    "output_dir": str(out),
    "mac_windows_project_mount": mount,
    "windows_output_path": windows_output_path,
    "mount_preflight": mount_preflight,
    "command": command,
    "returncode": rc,
    "status": "passed" if rc == 0 else "failed",
    "started_epoch_sec": start,
    "ended_epoch_sec": end,
    "elapsed_sec": max(0, end - start),
    "stdout_path": str(stdout_path),
    "stderr_path": str(stderr_path),
    "stdout_preview": stdout_text[:2000],
    "stderr_preview": stderr_text[:2000],
    "metadata": metadata,
    "delivery_status": "blocked",
    "delivery_allowed": False,
    "presentation_status": "{PRESENTATION_STATUS}",
    "listen_pack": {{}},
}}
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"event": "command_end", "elapsed_sec": manifest["elapsed_sec"], "returncode": rc, "status": manifest["status"], "gpu_inference_skipped": True}}, sort_keys=True) + "\\n")
PY
exit "$RC"
"""


def build_ssh_command(*, ssh_target: str, ssh_key: str, connect_timeout_sec: int) -> list[str]:
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(connect_timeout_sec)}",
        ssh_target,
        "sh -s",
    ]


def _preview_output(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:2000]
    return str(value or "")[:2000]


def run_macos_remote_api_command(
    *,
    ssh_target: str,
    ssh_key: str,
    mac_artifact_root: str,
    mac_windows_project_mount: Optional[str] = None,
    windows_project_root: str = DEFAULT_WINDOWS_PROJECT_ROOT,
    artifact_subdir: str,
    command: str,
    timeout_sec: int = 600,
    connect_timeout_sec: int = 8,
    metadata: Optional[Mapping[str, Any]] = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    safe_subdir = validate_artifact_subdir(artifact_subdir)
    remote_manifest_path = str(PurePosixPath(str(mac_artifact_root).rstrip("/")) / safe_subdir / "manifest.json")
    mac_output_path = str(PurePosixPath(str(mac_artifact_root).rstrip("/")) / safe_subdir)
    mount_preflight = build_remote_mount_preflight(
        mac_windows_project_mount=mac_windows_project_mount,
        mac_output_path=mac_output_path,
        windows_project_root=windows_project_root,
    )
    ssh_command = build_ssh_command(ssh_target=ssh_target, ssh_key=ssh_key, connect_timeout_sec=connect_timeout_sec)
    if mount_preflight.get("preflight_status") == "blocked":
        return {
            "schema": "macos_remote_api_client_result.v1",
            "ssh_command": ssh_command,
            "artifact_subdir": safe_subdir,
            "mac_artifact_root": mac_artifact_root,
            "mac_windows_project_mount": str(mac_windows_project_mount or ""),
            "windows_output_path": str(mount_preflight.get("windows_output_path") or ""),
            "remote_manifest_path": remote_manifest_path,
            "returncode": 78,
            "status": "blocked",
            "blocked_reason": str(mount_preflight.get("blocked_reason") or ""),
            "mount_preflight": mount_preflight,
            "elapsed_sec": 0.0,
            "stdout_preview": "",
            "stderr_preview": "",
            "delivery_allowed": False,
            "presentation_status": PRESENTATION_STATUS,
        }
    script = build_remote_script(
        command=command,
        mac_artifact_root=mac_artifact_root,
        mac_windows_project_mount=str(mac_windows_project_mount),
        windows_project_root=windows_project_root,
        artifact_subdir=artifact_subdir,
        metadata={
            **dict(metadata or {}),
            "ssh_target": ssh_target,
            "mac_artifact_root": mac_artifact_root,
            "artifact_subdir": safe_subdir,
        },
    )
    started = time.monotonic()
    try:
        completed = runner(
            ssh_command,
            input=script.encode("utf-8"),
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "schema": "macos_remote_api_client_result.v1",
            "ssh_command": ssh_command,
            "artifact_subdir": safe_subdir,
            "mac_artifact_root": mac_artifact_root,
            "mac_windows_project_mount": str(mac_windows_project_mount or ""),
            "windows_output_path": str(mount_preflight.get("windows_output_path") or ""),
            "remote_manifest_path": remote_manifest_path,
            "returncode": 127,
            "status": "failed",
            "error": {"type": "ssh_client_unavailable", "message": str(exc)},
            "mount_preflight": mount_preflight,
            "elapsed_sec": round(time.monotonic() - started, 6),
            "stdout_preview": "",
            "stderr_preview": "",
            "delivery_allowed": False,
            "presentation_status": PRESENTATION_STATUS,
        }
    if int(completed.returncode) == 78:
        status = "blocked"
        mount_preflight = {
            **mount_preflight,
            "preflight_status": "blocked",
            "blocked_reason": "mac_windows_project_mount_probe_failed",
            "can_write_route_artifacts": False,
            "probe_status": "failed",
        }
    else:
        status = "passed" if completed.returncode == 0 else "failed"
        mount_preflight = {
            **mount_preflight,
            "preflight_status": "passed",
            "can_write_route_artifacts": True,
            "probe_status": "write_read_delete_passed",
        }
    return {
        "schema": "macos_remote_api_client_result.v1",
        "ssh_command": ssh_command,
        "artifact_subdir": safe_subdir,
        "mac_artifact_root": mac_artifact_root,
        "mac_windows_project_mount": str(mac_windows_project_mount or ""),
        "windows_output_path": str(mount_preflight.get("windows_output_path") or ""),
        "remote_manifest_path": remote_manifest_path,
        "returncode": int(completed.returncode),
        "status": status,
        "blocked_reason": "mac_windows_project_mount_probe_failed" if status == "blocked" else "",
        "mount_preflight": mount_preflight,
        "elapsed_sec": round(time.monotonic() - started, 6),
        "stdout_preview": _preview_output(completed.stdout),
        "stderr_preview": _preview_output(completed.stderr),
        "delivery_allowed": False,
        "presentation_status": PRESENTATION_STATUS,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--mac-artifact-root", default="/Users/andrejviluha/AIMusicProject_win/artifacts")
    parser.add_argument("--mac-windows-project-mount")
    parser.add_argument("--windows-project-root", default=DEFAULT_WINDOWS_PROJECT_ROOT)
    parser.add_argument("--artifact-subdir", required=True)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--command")
    command_group.add_argument("--command-file", type=str)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--connect-timeout-sec", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    command = args.command
    if args.command_file:
        with open(args.command_file, "r", encoding="utf-8-sig", newline="") as handle:
            command = handle.read()
    result = run_macos_remote_api_command(
        ssh_target=args.ssh_target,
        ssh_key=args.ssh_key,
        mac_artifact_root=args.mac_artifact_root,
        mac_windows_project_mount=args.mac_windows_project_mount,
        windows_project_root=args.windows_project_root,
        artifact_subdir=args.artifact_subdir,
        command=str(command or ""),
        timeout_sec=args.timeout_sec,
        connect_timeout_sec=args.connect_timeout_sec,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["returncode"] == 0 else 1)


if __name__ == "__main__":
    main()
