"""Run a macOS remote command and force its artifacts into the shared folder.

The Mac side writes all output files under the provided mounted artifact root,
for example ``/Users/andrejviluha/win_artifacts/mac_api/c001`` which maps to
``D:\\AIMusicProject\\artifacts\\mac_api\\c001`` on the Windows host.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence


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


def build_remote_script(
    *,
    command: str,
    mac_artifact_root: str,
    artifact_subdir: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    safe_subdir = validate_artifact_subdir(artifact_subdir)
    output_dir = str(PurePosixPath(str(mac_artifact_root).rstrip("/")) / safe_subdir)
    command_script_b64 = _b64(str(command) + "\n")
    metadata_b64 = _b64(json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True))
    return f"""#!/bin/sh
set -u
OUT={_quote_posix(output_dir)}
CMD_B64={_quote_posix(command_script_b64)}
META_B64={_quote_posix(metadata_b64)}
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
python3 - "$OUT" "$RC" "$START" "$END" "$CMD_B64" "$META_B64" <<'PY'
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
stdout_path = out / "stdout.txt"
stderr_path = out / "stderr.txt"
stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
manifest = {{
    "schema": "{SCHEMA}",
    "output_dir": str(out),
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
    artifact_subdir: str,
    command: str,
    timeout_sec: int = 600,
    connect_timeout_sec: int = 8,
    metadata: Optional[Mapping[str, Any]] = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    script = build_remote_script(
        command=command,
        mac_artifact_root=mac_artifact_root,
        artifact_subdir=artifact_subdir,
        metadata={
            **dict(metadata or {}),
            "ssh_target": ssh_target,
            "mac_artifact_root": mac_artifact_root,
            "artifact_subdir": validate_artifact_subdir(artifact_subdir),
        },
    )
    started = time.monotonic()
    ssh_command = build_ssh_command(ssh_target=ssh_target, ssh_key=ssh_key, connect_timeout_sec=connect_timeout_sec)
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
            "artifact_subdir": validate_artifact_subdir(artifact_subdir),
            "mac_artifact_root": mac_artifact_root,
            "remote_manifest_path": str(
                PurePosixPath(str(mac_artifact_root).rstrip("/")) / validate_artifact_subdir(artifact_subdir) / "manifest.json"
            ),
            "returncode": 127,
            "status": "failed",
            "error": {"type": "ssh_client_unavailable", "message": str(exc)},
            "elapsed_sec": round(time.monotonic() - started, 6),
            "stdout_preview": "",
            "stderr_preview": "",
            "delivery_allowed": False,
            "presentation_status": PRESENTATION_STATUS,
        }
    return {
        "schema": "macos_remote_api_client_result.v1",
        "ssh_command": ssh_command,
        "artifact_subdir": validate_artifact_subdir(artifact_subdir),
        "mac_artifact_root": mac_artifact_root,
        "remote_manifest_path": str(PurePosixPath(str(mac_artifact_root).rstrip("/")) / validate_artifact_subdir(artifact_subdir) / "manifest.json"),
        "returncode": int(completed.returncode),
        "status": "passed" if completed.returncode == 0 else "failed",
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
    parser.add_argument("--mac-artifact-root", default="/Users/andrejviluha/win_artifacts")
    parser.add_argument("--artifact-subdir", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--connect-timeout-sec", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_macos_remote_api_command(
        ssh_target=args.ssh_target,
        ssh_key=args.ssh_key,
        mac_artifact_root=args.mac_artifact_root,
        artifact_subdir=args.artifact_subdir,
        command=args.command,
        timeout_sec=args.timeout_sec,
        connect_timeout_sec=args.connect_timeout_sec,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["returncode"] == 0 else 1)


if __name__ == "__main__":
    main()
