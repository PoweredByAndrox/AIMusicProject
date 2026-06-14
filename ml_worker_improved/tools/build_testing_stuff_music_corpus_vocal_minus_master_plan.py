#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_NONVOCAL_STEMS = ("bass", "drums", "melody", "other")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def accepted_listen_contract(manifest: dict[str, Any]) -> bool:
    return (
        str(manifest.get("delivery_status") or "") == "accepted"
        and manifest.get("delivery_allowed") is True
        and str(manifest.get("presentation_status") or "") == "accepted_for_listen_pack"
        and str(manifest.get("release_status") or "") == "listen_preview_only"
        and manifest.get("release_ready") is not True
        and manifest.get("can_mark_release_ready") is not True
        and manifest.get("can_publish_as_release") is not True
    )


def iter_manifest_files(roots: Sequence[Path], names: Sequence[str]) -> Iterable[Path]:
    wanted = set(names)
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name in wanted:
            yield root
            continue
        if root.is_dir():
            for path in root.rglob("*.json"):
                if path.name in wanted:
                    yield path


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def manifest_track_id(manifest: dict[str, Any]) -> str:
    return str(manifest.get("track_id") or manifest.get("source_track_id") or "")


def manifest_source_audio_path(manifest: dict[str, Any]) -> str:
    for key in ("source_mix_path", "source_audio_path", "audio_path", "mix_path"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def accepted_manifest_indexes(
    roots: Sequence[Path],
    names: Sequence[str],
    *,
    stem_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, Any]] = {}
    source_index: dict[str, dict[str, Any]] = {}
    for path in iter_manifest_files(roots, names):
        try:
            manifest = read_json(path)
        except (OSError, ValueError):
            continue
        track_id = manifest_track_id(manifest)
        manifest_stem = str(manifest.get("stem_type") or "")
        if not track_id or (manifest_stem and manifest_stem != stem_type):
            continue
        if not accepted_listen_contract(manifest):
            continue
        enriched = dict(manifest)
        enriched["_manifest_path"] = str(path)
        enriched["_source_audio_path"] = manifest_source_audio_path(manifest)
        index.setdefault(track_id, enriched)
        if enriched["_source_audio_path"]:
            source_index.setdefault(enriched["_source_audio_path"], enriched)
    return index, source_index


def planned_track_ids(gate_report: dict[str, Any]) -> list[str]:
    planned = gate_report.get("planned_target_stems_by_track")
    if isinstance(planned, dict):
        return stable_unique(planned.keys())
    return stable_unique(row.get("track_id") for row in gate_report.get("rows", []) if isinstance(row, dict))


def source_audio_by_track(gate_report: dict[str, Any]) -> dict[str, str]:
    source = gate_report.get("source_corpus_plan_jsonl")
    if not isinstance(source, dict):
        return {}
    path_text = str(source.get("path") or "")
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for row in iter_jsonl(path):
        track_id = str(row.get("track_id") or "")
        source_audio = str(row.get("audio_path") or row.get("source_audio_path") or "")
        if track_id and source_audio:
            result[track_id] = source_audio
    return result


def accepted_nonvocal_by_track(gate_report: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in gate_report.get("rows", []):
        if not isinstance(row, dict) or row.get("delivery_allowed") is not True:
            continue
        stem = str(row.get("stem_type") or "")
        track_id = str(row.get("track_id") or "")
        if stem in REQUIRED_NONVOCAL_STEMS and track_id:
            result.setdefault(track_id, set()).add(stem)
    return result


def executed_stems_by_track(gate_report: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in gate_report.get("rows", []):
        if not isinstance(row, dict):
            continue
        track_id = str(row.get("track_id") or "")
        stem = str(row.get("stem_type") or "")
        if track_id and stem:
            result.setdefault(track_id, set()).add(stem)
    return result


def selected_path(manifest: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    rows = manifest.get("accepted_listen_preview_rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        for key in keys:
            value = rows[0].get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def track_plan_row(
    track_id: str,
    *,
    track_source_audio_path: str,
    track_identity_match_method: str,
    accepted_nonvocal_stems: set[str],
    executed_stems: set[str],
    vocal_manifest: dict[str, Any] | None,
    minus_manifest: dict[str, Any] | None,
    master_manifest: dict[str, Any] | None,
    full_pack_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    has_vocal = vocal_manifest is not None
    has_minus = minus_manifest is not None
    has_master = master_manifest is not None
    has_full_pack = full_pack_manifest is not None
    missing_nonvocal = [stem for stem in REQUIRED_NONVOCAL_STEMS if stem not in accepted_nonvocal_stems]
    if not executed_stems and not has_full_pack:
        next_gate = "run_corpus_job_matrix_for_track"
    elif has_full_pack:
        next_gate = "full_stem_pack_listen_preview_ready_not_release"
    elif not has_vocal:
        next_gate = "run_vocal_delivery_pipeline"
    elif not has_minus:
        next_gate = "run_minus_delivery_pipeline"
    elif not has_master:
        next_gate = "run_pair_compatibility_and_master_manifest"
    elif missing_nonvocal:
        next_gate = "complete_nonvocal_stem_score_matrices"
    else:
        next_gate = "build_full_stem_pack_manifest"
    missing_gates: list[str] = []
    if not has_full_pack:
        if not has_vocal:
            missing_gates.append("vocal_delivery")
        if not has_minus:
            missing_gates.append("minus_delivery")
        if not has_master:
            missing_gates.append("master_manifest")
        if missing_nonvocal:
            missing_gates.append("nonvocal_stems")
    return {
        "track_id": track_id,
        "source_audio_path": track_source_audio_path,
        "track_identity_match_method": track_identity_match_method,
        "next_gate": next_gate,
        "missing_gates": missing_gates,
        "accepted_nonvocal_stems": sorted(accepted_nonvocal_stems),
        "missing_nonvocal_stems": missing_nonvocal,
        "executed_stems": sorted(executed_stems),
        "ready_for_master": has_vocal and has_minus,
        "ready_for_full_stem_pack": has_master and not missing_nonvocal,
        "vocal_manifest_path": str((vocal_manifest or {}).get("_manifest_path") or ""),
        "minus_manifest_path": str((minus_manifest or {}).get("_manifest_path") or ""),
        "master_manifest_path": str((master_manifest or {}).get("_manifest_path") or ""),
        "full_stem_pack_manifest_path": str((full_pack_manifest or {}).get("_manifest_path") or ""),
        "selected_vocal_path": selected_path(vocal_manifest or {}, "selected_vocal_path", "listen_vocals_flac", "selected_audio_path"),
        "selected_minus_path": selected_path(minus_manifest or {}, "selected_minus_path", "listen_minus_flac", "selected_audio_path"),
        "release_ready": False,
        "delivery_allowed": False,
    }


def build_summary(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("next_gate") or "") for row in tracks)
    summary = {
        "schema": "testing_stuff_music_corpus_vocal_minus_master_plan_summary.v1",
        "track_count": len(tracks),
        "release_ready": False,
        "delivery_allowed": False,
        "next_gate_counts": dict(sorted(counts.items())),
    }
    summary.update(dict(sorted(counts.items())))
    return summary


def build_plan(
    *,
    corpus_gate_report_json: Path,
    vocal_manifest_roots: Sequence[Path] = (),
    minus_manifest_roots: Sequence[Path] = (),
    master_manifest_roots: Sequence[Path] = (),
    full_stem_pack_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    gate_report = read_json(corpus_gate_report_json)
    tracks = planned_track_ids(gate_report)
    source_by_track = source_audio_by_track(gate_report)
    nonvocal = accepted_nonvocal_by_track(gate_report)
    executed = executed_stems_by_track(gate_report)
    vocal_index, vocal_source_index = accepted_manifest_indexes(
        vocal_manifest_roots,
        ("no_reference_vocal_listen_preview_manifest.json",),
        stem_type="vocal",
    )
    minus_index, minus_source_index = accepted_manifest_indexes(
        minus_manifest_roots,
        ("no_reference_minus_listen_preview_manifest.json",),
        stem_type="minus",
    )
    master_index, master_source_index = accepted_manifest_indexes(
        master_manifest_roots,
        ("no_reference_vocal_minus_master_manifest.json",),
        stem_type="both",
    )
    full_pack_index, full_pack_source_index = accepted_manifest_indexes(
        full_stem_pack_roots,
        ("no_reference_full_stem_pack_manifest.json",),
        stem_type="full_stem_pack",
    )
    def manifest_for(
        track_id: str,
        by_track: dict[str, dict[str, Any]],
        by_source: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str]:
        if track_id in by_track:
            return by_track[track_id], "track_id"
        source_audio = source_by_track.get(track_id, "")
        if source_audio and source_audio in by_source:
            return by_source[source_audio], "source_audio_path"
        return None, ""

    prepared: list[dict[str, Any]] = []
    for track_id in tracks:
        vocal_manifest, vocal_match = manifest_for(track_id, vocal_index, vocal_source_index)
        minus_manifest, minus_match = manifest_for(track_id, minus_index, minus_source_index)
        master_manifest, master_match = manifest_for(track_id, master_index, master_source_index)
        full_pack_manifest, full_pack_match = manifest_for(track_id, full_pack_index, full_pack_source_index)
        track_identity_match_method = (
            full_pack_match or master_match or vocal_match or minus_match or ("source_audio_path" if source_by_track.get(track_id) else "track_id")
        )
        prepared.append(
            track_plan_row(
                track_id,
                track_source_audio_path=source_by_track.get(track_id, ""),
                track_identity_match_method=track_identity_match_method,
                accepted_nonvocal_stems=nonvocal.get(track_id, set()),
                executed_stems=executed.get(track_id, set()),
                vocal_manifest=vocal_manifest,
                minus_manifest=minus_manifest,
                master_manifest=master_manifest,
                full_pack_manifest=full_pack_manifest,
            )
        )
    rows = [
        row for row in prepared
    ]
    rows.sort(key=lambda row: str(row.get("track_id") or ""))
    return {
        "schema": "testing_stuff_music_corpus_vocal_minus_master_plan.v1",
        "manifest_type": "testing_stuff_music_corpus_vocal_minus_master_plan",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_corpus_master_plan_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "source_corpus_gate_report_json": source_ref(corpus_gate_report_json),
        "summary": build_summary(rows),
        "tracks": rows,
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
    }


def build_summary_md(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# Testing Stuff Music Corpus Vocal/Minus/Master Plan",
        "",
        f"- track_count: {summary['track_count']}",
        f"- delivery_allowed: {str(plan.get('delivery_allowed') is True).lower()}",
        f"- release_ready: {str(plan.get('release_ready') is True).lower()}",
        "",
        "## Next Gate Counts",
        "",
    ]
    for gate, count in summary["next_gate_counts"].items():
        lines.append(f"- {gate}: {count}")
    lines.extend(["", "## Tracks", ""])
    for row in plan["tracks"]:
        missing = ",".join(row.get("missing_gates") or []) or "-"
        lines.append(f"- {row['track_id']} next={row['next_gate']} missing={missing}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed corpus plan for vocal/minus/master/full-stem progression."
    )
    parser.add_argument("--corpus-gate-report-json", type=Path, required=True)
    parser.add_argument("--vocal-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--minus-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--master-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--full-stem-pack-root", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        corpus_gate_report_json=args.corpus_gate_report_json,
        vocal_manifest_roots=args.vocal_manifest_root,
        minus_manifest_roots=args.minus_manifest_root,
        master_manifest_roots=args.master_manifest_root,
        full_stem_pack_roots=args.full_stem_pack_root,
    )
    write_json(args.output_manifest_json, plan)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary_md(plan), encoding="utf-8")
    print(json.dumps(plan["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
