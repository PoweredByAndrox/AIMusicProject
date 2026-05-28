from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATA_ROOT
from app.schemas import TandemSearchRequest, TandemSearchStatus
from app.services.job_store import load_job
from app.services.search_store import load_search, save_search, search_dir

PROJECT_ROOT = Path("/app")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "search_tandem_presets.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def create_search(request: TandemSearchRequest) -> TandemSearchStatus:
    _ = load_job(request.base_job_id)
    search_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
    root = search_dir(search_id)
    status = TandemSearchStatus(
        search_id=search_id,
        status="queued",
        request=request,
        output_root=f"/app/data/searches/{search_id}",
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )
    save_search(status)
    return status


def run_search_in_thread(search_id: str) -> None:
    thread = threading.Thread(target=run_tandem_search, args=(search_id,), daemon=True)
    thread.start()


def run_tandem_search(search_id: str) -> None:
    status = load_search(search_id)
    output_root = search_dir(search_id)
    updated = status.model_copy(update={"status": "running", "updated_at": _now_iso(), "output_root": f"/app/data/searches/{search_id}"})
    save_search(updated)

    if not SCRIPT_PATH.exists():
        failed = updated.model_copy(
            update={
                "status": "failed",
                "updated_at": _now_iso(),
                "error": f"search script is missing: {SCRIPT_PATH}",
            }
        )
        save_search(failed)
        return

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--root",
        str(PROJECT_ROOT),
        "--base-job",
        updated.request.base_job_id,
        "--max-candidates",
        str(updated.request.max_candidates),
        "--lyrics-validate-top-k",
        str(updated.request.lyrics_validate_top_k),
        "--poll-sec",
        "10",
        "--search-id",
        updated.search_id,
        "--output-dir",
        str(output_root),
    ]
    if updated.request.reference_instrumental_path:
        command.extend(["--reference-instrumental", updated.request.reference_instrumental_path])
    if updated.request.force_reference_target:
        command.append("--force-reference-target")

    env = dict(os.environ)
    env.setdefault("STEMLAB_API_BASE", "http://127.0.0.1:8000")

    result = subprocess.run(command, capture_output=True, text=True, env=env)
    stdout_tail = _tail(result.stdout or "")
    stderr_tail = _tail(result.stderr or "")

    report_path = output_root / "search_report.json"
    report_data = None
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report_data = None

    if result.returncode == 0 and report_data is not None:
        completed = updated.model_copy(
            update={
                "status": "completed",
                "updated_at": _now_iso(),
                "result": report_data,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "error": None,
            }
        )
        save_search(completed)
        return

    failed = updated.model_copy(
        update={
            "status": "failed",
            "updated_at": _now_iso(),
            "result": report_data,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "error": f"search runner failed with exit code {result.returncode}",
        }
    )
    save_search(failed)
