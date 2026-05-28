from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import DATA_ROOT, MEDIA_PREFIX

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
TEXT_EXTENSIONS = {".txt", ".lrc", ".srt", ".vtt", ".ass"}


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return safe.strip("._") or "upload"


def _save_upload(file: UploadFile, folder: str, allowed_exts: set[str]) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'none'}")
    stem = Path(file.filename or "upload").stem
    safe_name = f"{_sanitize_name(stem)}_{uuid.uuid4().hex[:8]}{suffix}"
    target = DATA_ROOT / folder / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    container_path = f"/app/data/{folder}/{safe_name}"
    return {
        "filename": safe_name,
        "container_path": container_path,
        "media_url": f"{MEDIA_PREFIX}/{folder}/{safe_name}",
        "size_bytes": target.stat().st_size,
    }


def _save_upload_preserving_name(file: UploadFile, folder: str, allowed_exts: set[str]) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'none'}")
    safe_name = _sanitize_name(file.filename or f"upload{suffix}")
    target = DATA_ROOT / folder / safe_name
    if target.exists():
        safe_name = f"{Path(safe_name).stem}_{uuid.uuid4().hex[:6]}{suffix}"
        target = DATA_ROOT / folder / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    container_path = f"/app/data/{folder}/{safe_name}"
    return {
        "filename": safe_name,
        "container_path": container_path,
        "media_url": f"{MEDIA_PREFIX}/{folder}/{safe_name}",
        "size_bytes": target.stat().st_size,
    }


@router.post("/audio")
def upload_audio(file: UploadFile = File(...)) -> dict:
    return _save_upload(file, "inputs", AUDIO_EXTENSIONS)


@router.post("/text")
def upload_text(file: UploadFile = File(...)) -> dict:
    return _save_upload(file, "inputs", TEXT_EXTENSIONS)


@router.post("/audio-pack")
def upload_audio_pack(
    files: list[UploadFile] = File(...),
    folder_name: str = Form("virtualdj_pack"),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    safe_folder = _sanitize_name(folder_name)
    folder = f"inputs/virtualdj/{safe_folder}_{uuid.uuid4().hex[:8]}"
    saved = [_save_upload_preserving_name(file, folder, AUDIO_EXTENSIONS) for file in files]
    return {
        "container_root": f"/app/data/{folder}",
        "count": len(saved),
        "files": saved,
    }
