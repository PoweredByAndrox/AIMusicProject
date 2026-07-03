"""Persistent separator backend contract for audio separation routes.

This module is deliberately audio-specific. It does not use text vLLM models;
it carries vLLM-style serving ideas into the existing Demucs/MDX/MelBand/SCNet
route boundary: one long-lived service, sanitized request cache keys, explicit
GPU requirements, and fail-closed backend errors.
"""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODE_SUBPROCESS = "subprocess"
MODE_PERSISTENT_SERVER = "persistent-server"
VALID_BACKEND_MODES = {MODE_SUBPROCESS, MODE_PERSISTENT_SERVER}
DEFAULT_SEPARATOR_SERVER_URL = "http://separator-server:8791"


class PersistentSeparatorError(RuntimeError):
    """Base persistent separator backend error."""


class PersistentSeparatorUnavailable(PersistentSeparatorError):
    """Raised when the persistent separator server cannot be reached."""


class PersistentSeparatorRejected(PersistentSeparatorError):
    """Raised when the server reports a fail-closed route rejection."""


@dataclass(frozen=True)
class SeparatorBackendConfig:
    mode: str = MODE_SUBPROCESS
    server_url: str = ""
    require_gpu: bool = True
    device: str = "cuda"
    request_timeout_sec: float = 0.0

    def validated(self) -> "SeparatorBackendConfig":
        mode = str(self.mode or MODE_SUBPROCESS).strip().lower()
        if mode not in VALID_BACKEND_MODES:
            raise ValueError(f"unknown_separator_backend:{self.mode}")
        device = str(self.device or ("cuda" if self.require_gpu else "cpu")).strip().lower()
        if self.require_gpu and device == "mps" and mode != MODE_PERSISTENT_SERVER:
            raise ValueError("mps_requires_persistent_separator_backend")
        if self.require_gpu and device not in {"cuda", "mps"}:
            raise ValueError("cpu_fallback_forbidden:require_gpu_true")
        server_url = str(self.server_url or "").strip().rstrip("/")
        if mode == MODE_PERSISTENT_SERVER and not server_url:
            server_url = DEFAULT_SEPARATOR_SERVER_URL
        return SeparatorBackendConfig(
            mode=mode,
            server_url=server_url,
            require_gpu=bool(self.require_gpu),
            device=device,
            request_timeout_sec=float(self.request_timeout_sec),
        )

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_PERSISTENT_SERVER

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "separator_backend.v1",
            "mode": self.mode,
            "server_url": self.server_url if self.enabled else "",
            "require_gpu": self.require_gpu,
            "device": self.device,
            "request_timeout_sec": self.request_timeout_sec,
            "can_accept_delivery": False,
        }


def separator_cache_key(
    *,
    route_id: str,
    source_audio_path: str | Path,
    command: Sequence[str],
    stems: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "route_id": str(route_id or ""),
        "source_audio_path_sha256": hashlib.sha256(str(source_audio_path).encode("utf-8", errors="surrogatepass")).hexdigest(),
        "command": [str(part) for part in command],
        "stems": [str(stem) for stem in stems],
        "params": dict(params or {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"sep_{digest[:20]}"


class PersistentSeparatorClient:
    def __init__(self, server_url: str, *, request_timeout_sec: float = 30.0) -> None:
        self.server_url = str(server_url or "").strip().rstrip("/")
        self.request_timeout_sec = float(request_timeout_sec)

    def separate(
        self,
        *,
        route_id: str,
        source_audio_path: str | Path,
        output_dir: str | Path,
        device: str,
        stems: Sequence[str],
        batch_id: str,
        command: Sequence[str],
        cache_key: str,
        timeout_sec: int,
        require_gpu: bool,
        prefer_resident: bool = False,
    ) -> dict[str, Any]:
        if not self.server_url:
            raise PersistentSeparatorUnavailable("separator_server_url_missing")
        payload = {
            "schema": "persistent_separator_request.v1",
            "route_id": str(route_id),
            "source_audio_path": str(source_audio_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "stems": [str(stem) for stem in stems],
            "batch_id": str(batch_id),
            "command": [str(part) for part in command],
            "cache_key": str(cache_key),
            "timeout_sec": int(timeout_sec),
            "require_gpu": bool(require_gpu),
            "prefer_resident": bool(prefer_resident),
        }
        request = urllib.request.Request(
            f"{self.server_url}/separate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            try:
                error_data = json.loads(raw_error)
            except json.JSONDecodeError:
                error_data = {}
            reason = str(error_data.get("reason") or error_data.get("error") or f"separator_server_http_{exc.code}")
            raise PersistentSeparatorRejected(reason) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise PersistentSeparatorUnavailable(f"separator_server_unavailable:{exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PersistentSeparatorRejected("separator_server_invalid_json") from exc
        if not isinstance(data, dict):
            raise PersistentSeparatorRejected("separator_server_invalid_response")
        status = str(data.get("status") or "").strip().lower()
        if status not in {"ok", "cached"}:
            reason = str(data.get("reason") or data.get("error") or "separator_server_rejected")
            raise PersistentSeparatorRejected(reason)
        data.setdefault("cache_hit", status == "cached")
        return data


def make_separator_backend_config(
    *,
    mode: str = MODE_SUBPROCESS,
    server_url: str = "",
    require_gpu: bool = True,
    device: str | None = None,
    request_timeout_sec: float = 0.0,
) -> SeparatorBackendConfig:
    return SeparatorBackendConfig(
        mode=mode,
        server_url=server_url,
        require_gpu=require_gpu,
        device=device or ("cuda" if require_gpu else "cpu"),
        request_timeout_sec=request_timeout_sec,
    ).validated()
