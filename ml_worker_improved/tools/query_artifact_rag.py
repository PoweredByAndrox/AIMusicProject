#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for import_root in (APP_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from docker_runtime import require_docker_runtime  # noqa: E402

DEFAULT_RAG_ROOT = Path("/app/artifact_rag")
REQUIRED_PACK_FILES = ("roots.jsonl", "manifest_refs.jsonl", "chunks.jsonl")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def terms_for_query(query: str) -> list[str]:
    return [term for term in re.split(r"[^0-9A-Za-zА-Яа-я_:-]+", query.lower()) if len(term) >= 2]


def score_text(text: str, terms: Sequence[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def excerpt(text: str, terms: Sequence[str], *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lower = text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]


def resolve_inside(root: Path, relative_path: str | Path, *, field_name: str) -> Path:
    root_resolved = root.resolve()
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ValueError(f"{field_name} must be relative to artifact rag root")
    resolved = (root_resolved / raw_path).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes artifact rag root: {relative_path}") from exc
    return resolved


def validate_pack_files(pack_dir: Path) -> None:
    missing = [name for name in REQUIRED_PACK_FILES if not (pack_dir / name).is_file()]
    if missing:
        raise ValueError(f"artifact RAG pack is incomplete; missing required files: {', '.join(missing)}")


def resolve_pack(rag_root: Path, rag_id: str | None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    current_path = rag_root / "current.json"
    current = read_json(current_path)
    if rag_id:
        idx_path = resolve_inside(rag_root, Path(rag_id) / "idx.json", field_name="rag_id")
        pack_dir = idx_path.parent
    else:
        idx_rel = str(current.get("idx") or "")
        if not idx_rel:
            raise ValueError(f"current.json does not contain idx: {current_path}")
        idx_path = resolve_inside(rag_root, idx_rel, field_name="current.idx")
        pack_dir = idx_path.parent
    if idx_path.name != "idx.json":
        raise ValueError(f"artifact RAG index path must end with idx.json: {idx_path}")
    idx = read_json(idx_path)
    return current, pack_dir, idx


def root_lookup(pack_dir: Path) -> dict[str, dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(pack_dir / "roots.jsonl"):
        alias = str(row.get("root_alias") or "")
        if alias:
            roots[alias] = row
    return roots


def row_text(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("root") or ""),
        str(row.get("root_alias") or ""),
        str(row.get("relative_path") or ""),
        json.dumps(row.get("status_fields") or {}, ensure_ascii=False, sort_keys=True),
        str(row.get("excerpt") or ""),
        str(row.get("text") or ""),
    ]
    return "\n".join(pieces)


def query_artifact_rag(
    *,
    rag_root: Path,
    query: str,
    rag_id: str | None = None,
    limit: int = 12,
    excerpt_chars: int = 900,
) -> dict[str, Any]:
    current, pack_dir, idx = resolve_pack(rag_root, rag_id)
    terms = terms_for_query(query)
    if not terms:
        raise ValueError("query must contain at least one searchable term")
    validate_pack_files(pack_dir)
    roots = root_lookup(pack_dir)
    candidates: list[dict[str, Any]] = []
    for source_name in ("manifest_refs.jsonl", "chunks.jsonl"):
        for row in iter_jsonl(pack_dir / source_name):
            text = row_text(row)
            score = score_text(text, terms)
            if score <= 0:
                continue
            root_alias = str(row.get("root_alias") or "")
            root = roots.get(root_alias, {})
            candidates.append(
                {
                    "score": score,
                    "source": source_name,
                    "root": row.get("root") or root.get("root") or "",
                    "root_alias": root_alias,
                    "relative_path": row.get("relative_path") or "",
                    "status_fields": row.get("status_fields") or {},
                    "excerpt": excerpt(str(row.get("text") or row.get("excerpt") or ""), terms, max_chars=excerpt_chars),
                    "root_summary": {
                        "gib": root.get("gib"),
                        "files": root.get("files"),
                        "text_snapshot_files": root.get("text_snapshot_files"),
                    },
                }
            )
    candidates.sort(key=lambda item: (-int(item["score"]), str(item["root_alias"]), str(item["relative_path"])))
    return {
        "schema": "artifact_rag_query_result.v1",
        "query": query,
        "terms": terms,
        "rag_id": idx.get("rag_id") or pack_dir.name,
        "current": {
            "rag_id": current.get("rag_id"),
            "idx": current.get("idx"),
            "status": current.get("status") or idx.get("status"),
            "current_retention_mode": current.get("current_retention_mode") or idx.get("current_retention_mode"),
            "binary_audio_model_bytes_available": current.get(
                "binary_audio_model_bytes_available",
                idx.get("binary_audio_model_bytes_available", True),
            ),
            "query_hint": current.get("query_hint") or idx.get("query_hint"),
        },
        "pack_summary": idx.get("summary") or {},
        "retention_contract": idx.get("retention_contract") or {},
        "result_count": min(len(candidates), limit),
        "matches": candidates[:limit],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the current artifact_rag pack for LLM-readable evidence.")
    parser.add_argument("--artifact-rag-root", type=Path, default=DEFAULT_RAG_ROOT)
    parser.add_argument("--rag-id")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--excerpt-chars", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="query_artifact_rag.py", entrypoint_path="/app/tools/query_artifact_rag.py")
    args = parse_args(argv)
    result = query_artifact_rag(
        rag_root=args.artifact_rag_root,
        rag_id=args.rag_id,
        query=str(args.query),
        limit=int(args.limit),
        excerpt_chars=int(args.excerpt_chars),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
