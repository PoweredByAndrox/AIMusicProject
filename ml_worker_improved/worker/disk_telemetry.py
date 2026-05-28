"""
Disk telemetry - lightweight disk usage monitoring for ML worker.

IMPORTANT: This module was rewritten to eliminate expensive recursive file
walking (rglob + stat on every file).  The old _safe_dir_size() was the
primary source of heavy C:\\ disk I/O in WSL Docker, causing the Docker
daemon to become unresponsive during ML analysis.

Now uses only shutil.disk_usage() which is a single syscall (statvfs).
Recursive directory scanning is entirely removed.
"""

import json
import logging
import os
import shutil
import threading
import time
import resource
from pathlib import Path
from typing import Dict, Optional

# Master switch: set DISK_TELEMETRY_ENABLED=0 to disable all telemetry I/O.
_TELEMETRY_ENABLED = os.getenv('DISK_TELEMETRY_ENABLED', '1') != '0'
_SAMPLER_INTERVAL_SECONDS = max(1.0, float(os.getenv('DISK_TELEMETRY_SAMPLE_SECONDS', '5')))
_SAMPLER_STARTED = False
_SAMPLER_LOCK = threading.Lock()


def _safe_disk_usage(path: Path) -> Dict[str, Optional[float]]:
    """Return free/used/total for the filesystem containing *path*.

    Uses shutil.disk_usage() — a single statvfs syscall, zero file traversal.
    """
    try:
        usage = shutil.disk_usage(path)
        return {
            "free_gb": round(usage.free / (1024 ** 3), 3),
            "used_gb": round((usage.total - usage.free) / (1024 ** 3), 3),
            "total_gb": round(usage.total / (1024 ** 3), 3),
        }
    except Exception:
        return {
            "free_gb": None,
            "used_gb": None,
            "total_gb": None,
        }


def _safe_dir_size_shallow(path: Path) -> Dict[str, Optional[float]]:
    """Return an *approximate* size of a directory by listing only its
    immediate children (depth=1).  This is orders of magnitude cheaper than
    the old rglob("*") approach and still useful for spotting leak growth in
    demix-*/spec-* temp dirs that are typically flat.

    For large trees (storage root) this intentionally under-counts — use
    _safe_disk_usage() for filesystem-level numbers instead.
    """
    try:
        if not path.exists():
            return {"size_gb": 0.0, "file_count": 0}
        total_bytes = 0
        file_count = 0
        # iterdir is non-recursive — only one readdir + stat per entry
        for child in path.iterdir():
            try:
                if child.is_file():
                    total_bytes += child.stat().st_size
                    file_count += 1
            except Exception:
                pass
        return {
            "size_gb": round(total_bytes / (1024 ** 3), 3),
            "file_count": file_count,
        }
    except Exception:
        return {
            "size_gb": None,
            "file_count": None,
        }


def _safe_open_fd_count() -> Optional[int]:
    try:
        return len(list(Path('/proc/self/fd').iterdir()))
    except Exception:
        return None


def _read_proc_key_values(path: Path) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            payload[key.strip()] = value.strip()
    except Exception:
        return {}
    return payload


def snapshot_process_state() -> Dict[str, object]:
    if not _TELEMETRY_ENABLED:
        return {"telemetry": "disabled"}

    io_stats = _read_proc_key_values(Path('/proc/self/io'))
    status_stats = _read_proc_key_values(Path('/proc/self/status'))
    usage = resource.getrusage(resource.RUSAGE_SELF)

    def _int_value(source: Dict[str, str], key: str) -> Optional[int]:
        raw = source.get(key)
        if raw is None:
            return None
        try:
            return int(raw.split()[0])
        except Exception:
            return None

    return {
        "pid": os.getpid(),
        "rss_mb": round((_int_value(status_stats, 'VmRSS') or 0) / 1024.0, 2),
        "threads": _int_value(status_stats, 'Threads'),
        "fd_count": _safe_open_fd_count(),
        "cpu_user_s": round(usage.ru_utime, 3),
        "cpu_system_s": round(usage.ru_stime, 3),
        "read_bytes": _int_value(io_stats, 'read_bytes'),
        "write_bytes": _int_value(io_stats, 'write_bytes'),
        "cancelled_write_bytes": _int_value(io_stats, 'cancelled_write_bytes'),
        "read_syscalls": _int_value(io_stats, 'syscr'),
        "write_syscalls": _int_value(io_stats, 'syscw'),
    }


def _diff_process_state(previous: Optional[Dict[str, object]], current: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not previous:
        return None

    delta: Dict[str, object] = {}
    for key in (
        'cpu_user_s',
        'cpu_system_s',
        'read_bytes',
        'write_bytes',
        'cancelled_write_bytes',
        'read_syscalls',
        'write_syscalls',
    ):
        previous_value = previous.get(key)
        current_value = current.get(key)
        if previous_value is None or current_value is None:
            continue
        delta[key] = round(float(current_value) - float(previous_value), 3)

    if not delta:
        return None

    return delta


def snapshot_disk_state(
    *,
    storage_path: Path,
    root_path: Path,
    tmp_root: Optional[Path] = None,
    demix_dir: Optional[Path] = None,
    spec_dir: Optional[Path] = None,
    audio_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Capture a lightweight disk state snapshot.

    Only uses statvfs-level calls and shallow (depth=1) listing for temp dirs.
    Never walks storage_path recursively.
    """
    if not _TELEMETRY_ENABLED:
        return {"telemetry": "disabled"}

    payload: Dict[str, object] = {
        "storage_path": str(storage_path),
        "storage_disk": _safe_disk_usage(storage_path),
        "root_path": str(root_path),
        "root_disk": _safe_disk_usage(root_path),
    }

    if tmp_root is not None:
        payload["tmp_root"] = str(tmp_root)
        # Shallow listing is fine: tmp_root children are demix-*/spec-* dirs
        payload["tmp_root_state"] = _safe_dir_size_shallow(tmp_root)

    if demix_dir is not None:
        payload["demix_dir"] = str(demix_dir)
        payload["demix_dir_state"] = _safe_dir_size_shallow(demix_dir)

    if spec_dir is not None:
        payload["spec_dir"] = str(spec_dir)
        payload["spec_dir_state"] = _safe_dir_size_shallow(spec_dir)

    if audio_path is not None:
        try:
            payload["audio_path"] = str(audio_path)
            payload["audio_size_gb"] = round(audio_path.stat().st_size / (1024 ** 3), 6)
        except Exception:
            payload["audio_path"] = str(audio_path)
            payload["audio_size_gb"] = None

    return payload


def start_disk_telemetry_sampler(
    logger: logging.Logger,
    *,
    storage_path: Path,
    root_path: Path,
    tmp_root: Optional[Path] = None,
) -> None:
    if not _TELEMETRY_ENABLED:
        return

    global _SAMPLER_STARTED
    with _SAMPLER_LOCK:
        if _SAMPLER_STARTED:
            return
        _SAMPLER_STARTED = True

    def loop() -> None:
        previous_process_state: Optional[Dict[str, object]] = None
        logger.info(json.dumps({
            "event": "process_sampler.start",
            "interval_seconds": _SAMPLER_INTERVAL_SECONDS,
            "snapshot": snapshot_disk_state(
                storage_path=storage_path,
                root_path=root_path,
                tmp_root=tmp_root,
            ),
            "process": snapshot_process_state(),
        }, ensure_ascii=True, sort_keys=True))

        while True:
            time.sleep(_SAMPLER_INTERVAL_SECONDS)
            process_state = snapshot_process_state()
            logger.info(json.dumps({
                "event": "process_sampler.sample",
                "interval_seconds": _SAMPLER_INTERVAL_SECONDS,
                "snapshot": snapshot_disk_state(
                    storage_path=storage_path,
                    root_path=root_path,
                    tmp_root=tmp_root,
                ),
                "process": process_state,
                "process_delta": _diff_process_state(previous_process_state, process_state),
            }, ensure_ascii=True, sort_keys=True))
            previous_process_state = process_state

    thread = threading.Thread(
        target=loop,
        name="disk-telemetry-sampler",
        daemon=True,
    )
    thread.start()


def emit_disk_telemetry(logger: logging.Logger, event: str, **fields: object) -> None:
    """Emit a telemetry event.  No-op when DISK_TELEMETRY_ENABLED=0."""
    if not _TELEMETRY_ENABLED:
        return
    payload = {"event": event, "process": snapshot_process_state(), **fields}
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
