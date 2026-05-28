from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.schemas import JobStatus
from app.services.artifacts import (
    AUDIO_EXTENSIONS,
    build_artifact_info,
    collect_job_artifacts,
    resolve_container_data_path,
)

ROLE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("vocals", ("acapella", "acap", "vocals", "vocal")),
    ("instrumental_mix", ("instrumental", "karaoke", "minus", "no_vocals", "novocals")),
    ("drums", ("beat", "drums", "drum")),
    ("bass", ("bass",)),
    ("melody", ("melody",)),
]

ROLE_LABELS = {
    "vocals": "Acapella / Vocals",
    "instrumental_mix": "Instrumental",
    "drums": "Beat / Drums",
    "bass": "Bass",
    "melody": "Melody",
    "unknown": "Unclassified audio",
}

ROLE_NOTES = {
    "vocals": "Direct 1:1 compare against our `vocals` stem.",
    "instrumental_mix": "Direct 1:1 compare against our `instrumental_mix` minus.",
    "drums": "Usually closest to our `drums`, but VirtualDJ beat can contain transient-only emphasis.",
    "bass": "Direct 1:1 compare against our `bass` stem.",
    "melody": "Not 1:1 with our graph. Start from `other`, then sanity-check `guitar`, `piano`, `synth_lead`, `strings_orchestra`.",
    "unknown": "Filename did not map cleanly. Compare manually before trusting any match.",
}

ROLE_COMPARE_CANDIDATES = {
    "vocals": ["vocals"],
    "instrumental_mix": ["instrumental_mix", "other"],
    "drums": ["drums"],
    "bass": ["bass"],
    "melody": ["other", "guitar", "piano", "synth_lead", "strings_orchestra", "instrumental_mix"],
    "unknown": ["instrumental_mix", "other"],
}

VIRTUALDJ_CHECKLIST = {
    "what_to_export": [
        "From VirtualDJ export at least `Acapella` and `Instrumental` for the same track.",
        "If available, also export `Beat` and `Melody` to get a broader A/B against our stem families.",
        "Keep one track per folder. Mixed folders make auto-match noisy and misleading.",
    ],
    "preferred_format": [
        "Use WAV first. FLAC is acceptable. MP3 is only fallback because codec loss muddies A/B.",
        "Do not rename files into random names. Keep `Acapella`, `Instrumental`, `Beat`, `Melody` in filenames.",
        "Upload the whole VirtualDJ export pack into one folder so the scanner can map roles automatically.",
    ],
    "how_to_compare": [
        "Compare `Acapella` against our `vocals` and `Instrumental` against our `instrumental_mix` first.",
        "Treat `Melody` as approximate, not exact. It is broader than our single `other` stem.",
        "Listen to the same short problem windows on both sides instead of full-track vague impressions.",
    ],
}


def _detect_role(path: Path) -> str:
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    for role, patterns in ROLE_PATTERNS:
        if any(pattern in name for pattern in patterns):
            return role
    return "unknown"


def _choose_pipeline_match(role: str, job_artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = ROLE_COMPARE_CANDIDATES.get(role, ROLE_COMPARE_CANDIDATES["unknown"])
    for candidate in candidates:
        for artifact in job_artifacts:
            if artifact["key"] == candidate:
                return artifact
    return None


def _normalize_job_audio(job: JobStatus | None) -> list[dict[str, Any]]:
    if job is None:
        return []
    artifacts = collect_job_artifacts(job)
    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not artifact.exists:
            continue
        if not artifact.kind.startswith("audio"):
            continue
        if artifact.source == "request":
            continue
        normalized.append(artifact.model_dump())
    return normalized


def scan_virtualdj_baseline(baseline_root: str, job: JobStatus | None = None) -> dict[str, Any]:
    root = resolve_container_data_path(baseline_root)
    if not root.exists():
        raise FileNotFoundError(f"VirtualDJ baseline folder not found: {baseline_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"VirtualDJ baseline path is not a folder: {baseline_root}")

    job_audio_artifacts = _normalize_job_audio(job)
    items: List[dict[str, Any]] = []
    detected_roles: set[str] = set()
    audio_files = [
        file_path
        for file_path in sorted(root.iterdir())
        if file_path.is_file() and not file_path.name.startswith(".") and file_path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    for file_path in audio_files:
        role = _detect_role(file_path)
        if role == "unknown" and len(audio_files) == 1:
            role = "instrumental_mix"
        detected_roles.add(role)
        artifact = build_artifact_info(file_path.name, file_path, "virtualdj-baseline").model_dump()
        matched = _choose_pipeline_match(role, job_audio_artifacts)
        items.append(
            {
                "role": role,
                "role_label": ROLE_LABELS.get(role, role),
                "note": ROLE_NOTES.get(role, ROLE_NOTES["unknown"]),
                "recommended_pipeline_keys": ROLE_COMPARE_CANDIDATES.get(role, ROLE_COMPARE_CANDIDATES["unknown"]),
                "baseline_artifact": artifact,
                "pipeline_artifact": matched,
            }
        )

    missing_recommended = [
        role
        for role in ["vocals", "instrumental_mix"]
        if role not in detected_roles
    ]

    return {
        "baseline_root": baseline_root,
        "resolved_baseline_root": str(root),
        "job_id": job.job_id if job else None,
        "checklist": VIRTUALDJ_CHECKLIST,
        "items": items,
        "summary": {
            "file_count": len(items),
            "detected_roles": sorted(detected_roles),
            "missing_recommended_roles": missing_recommended,
        },
    }
