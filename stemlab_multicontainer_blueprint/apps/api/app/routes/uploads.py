import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import DATA_ROOT

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])

INPUTS_ROOT = DATA_ROOT / "inputs"


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "upload.bin").name
    stem = Path(name).stem or "upload"
    suffix = Path(name).suffix or ".bin"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "upload"
    return f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix.lower()}"


@router.get("")
def list_uploads() -> dict:
    files = []
    for path in sorted(INPUTS_ROOT.glob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "name": path.name,
                "input_path": f"/app/data/inputs/{path.name}",
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files}


@router.post("")
async def upload_audio(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    safe_name = sanitize_filename(file.filename)
    destination = INPUTS_ROOT / safe_name

    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    finally:
        await file.close()

    return {
        "filename": safe_name,
        "original_filename": file.filename,
        "size_bytes": destination.stat().st_size,
        "input_path": f"/app/data/inputs/{safe_name}",
    }
