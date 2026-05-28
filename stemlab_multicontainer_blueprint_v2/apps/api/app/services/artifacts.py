from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from app.config import DATA_ROOT, MEDIA_PREFIX
from app.schemas import ArtifactInfo, JobStatus

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}
LYRICS_TEXT_EXTENSIONS = {".txt", ".lrc", ".srt", ".vtt", ".ass"}
CONTAINER_DATA_PREFIX = "/app/data"
PREFERRED_KEYS = [
    "vocals",
    "instrumental_mix",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "lyrics_wordlevel_json",
    "lyrics_json",
    "lyrics_wordlevel_lrc",
    "lyrics_lrc",
    "lyrics_wordlevel_srt",
    "lyrics_srt",
    "lyrics_wordlevel_txt",
    "lyrics_txt",
    "job_manifest.json",
    "input_audio",
    "lyrics_reference",
]


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def resolve_container_data_path(path_or_container_path: str | Path) -> Path:
    path = Path(path_or_container_path)
    raw = str(path_or_container_path)
    if raw.startswith(CONTAINER_DATA_PREFIX):
        relative = raw[len(CONTAINER_DATA_PREFIX) :].lstrip("/\\")
        path = DATA_ROOT / relative
    if not path.is_absolute():
        path = DATA_ROOT / path
    resolved = _safe_resolve(path)
    try:
        resolved.relative_to(_safe_resolve(DATA_ROOT))
    except ValueError as exc:
        raise ValueError(f"Path escapes DATA_ROOT: {path_or_container_path}") from exc
    return resolved


def path_to_media_url(path: Path) -> str | None:
    resolved = _safe_resolve(path)
    try:
        relative = resolved.relative_to(_safe_resolve(DATA_ROOT))
    except ValueError:
        return None
    return f"{MEDIA_PREFIX}/{relative.as_posix()}"


def classify_artifact(path: Path, key: str) -> str:
    ext = path.suffix.lower()
    stem_l = path.stem.lower()
    key_l = key.lower()
    if ext in AUDIO_EXTENSIONS:
        if stem_l == "vocals" or key_l == "vocals":
            return "audio-vocals"
        if stem_l in {"instrumental_mix", "instrumental", "minus", "no_vocals"} or key_l == "instrumental_mix":
            return "audio-instrumental"
        return "audio"
    if ext == ".json":
        if "lyrics" in key_l or "transcript" in key_l:
            return "lyrics-json"
        if path.name == "job_manifest.json":
            return "job-manifest"
        return "json"
    if ext in LYRICS_TEXT_EXTENSIONS:
        return "lyrics-text"
    if ext in {".md", ".log"}:
        return "text"
    return "other"


def _artifact_sort_rank(item: ArtifactInfo) -> tuple[int, int, str]:
    source_rank = {"job-index": 0, "request": 1, "scan": 2}.get(item.source, 3)
    try:
        key_rank = PREFERRED_KEYS.index(item.key)
    except ValueError:
        key_rank = len(PREFERRED_KEYS)
    return (source_rank, key_rank, item.key)


def build_artifact_info(key: str, path: Path, source: str) -> ArtifactInfo:
    exists = path.exists() and path.is_file()
    size = path.stat().st_size if exists else 0
    try:
        relative = path.relative_to(_safe_resolve(DATA_ROOT)).as_posix()
    except ValueError:
        relative = None
    return ArtifactInfo(
        key=key,
        path=str(path),
        kind=classify_artifact(path, key),
        exists=exists,
        size_bytes=size,
        url=path_to_media_url(path),
        source=source,
        relative_path=relative,
    )


def _artifact_from_path(key: str, path: Path, source: str) -> ArtifactInfo:
    return build_artifact_info(key, path, source)


def collect_job_artifacts(job: JobStatus) -> List[ArtifactInfo]:
    collected: Dict[str, ArtifactInfo] = {}
    path_index: Dict[Path, str] = {}
    canonical_paths: Dict[Path, str] = {}
    canonical_signatures: set[tuple[str, str]] = set()

    for path_str in job.artifacts.values():
        candidate = Path(path_str)
        resolved = _safe_resolve(candidate)
        canonical_paths[resolved] = str(candidate)
        canonical_signatures.add((candidate.stem.lower(), candidate.suffix.lower()))

    def add(key: str, path: Path, source: str) -> None:
        resolved = _safe_resolve(path)
        signature = (path.stem.lower(), path.suffix.lower())

        if source == "scan":
            if resolved in canonical_paths:
                return
            if signature in canonical_signatures:
                return

        existing_key = path_index.get(resolved)
        if existing_key and existing_key != key:
            existing = collected[existing_key]
            if source == "scan":
                return
            if existing.source == "scan" and source != "scan":
                collected.pop(existing_key, None)
                path_index[resolved] = key
            else:
                return

        if key in collected:
            current = collected[key]
            if not current.exists and path.exists():
                collected[key] = _artifact_from_path(key, path, source)
            return
        collected[key] = _artifact_from_path(key, path, source)
        path_index[resolved] = key

    input_path = Path(job.request.input_path)
    add("input_audio", input_path, "request")
    if job.request.lyrics_reference_path:
        add("lyrics_reference", Path(job.request.lyrics_reference_path), "request")

    for key, path_str in job.artifacts.items():
        add(key, Path(path_str), "job-index")

    output_root = Path(job.output_root)
    if output_root.exists():
        for file_path in sorted(output_root.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                key = file_path.relative_to(output_root).as_posix()
                add(key, file_path, "scan")

    return sorted(collected.values(), key=_artifact_sort_rank)
