from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from app.config import DATA_ROOT
from app.schemas import TandemSearchStatus

SEARCHES_ROOT = DATA_ROOT / "searches"
SEARCHES_ROOT.mkdir(parents=True, exist_ok=True)


def search_dir(search_id: str) -> Path:
    path = SEARCHES_ROOT / search_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_search(search: TandemSearchStatus) -> None:
    path = search_dir(search.search_id) / "search.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(search.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_search(search_id: str) -> TandemSearchStatus:
    path = search_dir(search_id) / "search.json"
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"search file is empty: {path}")
    return TandemSearchStatus.model_validate_json(raw)


def list_searches() -> Dict[str, TandemSearchStatus]:
    searches: Dict[str, TandemSearchStatus] = {}
    search_files = sorted(SEARCHES_ROOT.glob("*/search.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in search_files:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        try:
            search = TandemSearchStatus.model_validate_json(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        searches[search.search_id] = search
    return searches
