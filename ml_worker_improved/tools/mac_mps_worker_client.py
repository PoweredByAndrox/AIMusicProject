"""Lightweight Unix-socket client for the Mac MPS model worker."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Mapping, Sequence


def send_request(socket_path: Path, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
    if timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be positive")
    encoded = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise ValueError("request exceeds 1 MiB")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_sec)
    try:
        client.connect(str(socket_path))
        client.sendall(encoded)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 1024 * 1024:
                raise RuntimeError("worker response exceeds 1 MiB")
            if b"\n" in chunk:
                break
    finally:
        client.close()
    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise RuntimeError("worker returned an empty response")
    response = json.loads(raw.decode("utf-8"))
    if not isinstance(response, dict):
        raise ValueError("worker response must be a JSON object")
    return response


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=("health", "release_cache", "shutdown"),
        required=True,
    )
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, Any] = {"action": str(args.action)}
    response = send_request(args.socket_path, payload, float(args.timeout_sec))
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0 if response.get("status") in {"ok", "ready", "stopping"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
