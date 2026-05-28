from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse

from app.config import DATA_ROOT

router = APIRouter(prefix="/media", tags=["media"])

CHUNK_SIZE = 1024 * 1024


def _resolve_media_path(asset_path: str) -> Path:
    candidate = (DATA_ROOT / asset_path).resolve()
    data_root = DATA_ROOT.resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return candidate


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes="):
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Invalid range")

    start_text, _, end_text = range_header[6:].partition("-")
    if not start_text and not end_text:
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Invalid range")

    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:
        suffix = int(end_text)
        if suffix <= 0:
            raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Invalid range")
        start = max(0, file_size - suffix)
        end = file_size - 1

    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, detail="Invalid range")

    return start, min(end, file_size - 1)


def _iter_file_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.head("/{asset_path:path}")
@router.get("/{asset_path:path}")
def get_media(asset_path: str, request: Request) -> Response:
    file_path = _resolve_media_path(asset_path)
    file_size = file_path.stat().st_size
    media_type, _ = mimetypes.guess_type(str(file_path))
    media_type = media_type or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        response = FileResponse(file_path, media_type=media_type)
        response.headers["Accept-Ranges"] = "bytes"
        return response

    start, end = _parse_range_header(range_header, file_size)
    content_length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(content_length),
    }

    if request.method == "HEAD":
        return Response(status_code=status.HTTP_206_PARTIAL_CONTENT, media_type=media_type, headers=headers)

    return StreamingResponse(
        _iter_file_range(file_path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers=headers,
    )
