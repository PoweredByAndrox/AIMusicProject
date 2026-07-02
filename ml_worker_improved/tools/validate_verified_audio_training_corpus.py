from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime


START_TIME = time.monotonic()
MANIFEST_NAME = "verified_audio_training_corpus_manifest.json"
ROW_REPORT_NAME = "verified_audio_training_corpus_rows.jsonl"
PROGRESS_NAME = "verified_audio_training_corpus_progress.jsonl"
SHORT_NAMES = {
    "manifest": "val.json",
    "rows": "rows.jsonl",
    "progress": "progress.jsonl",
}
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "metadata_only_verified_corpus_validation",
        "expected_acceleration_mode": "external_only_cpu_metadata_curation",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def output_names(*, short_output_names: bool) -> dict[str, str]:
    if short_output_names:
        return dict(SHORT_NAMES)
    return {
        "manifest": MANIFEST_NAME,
        "rows": ROW_REPORT_NAME,
        "progress": PROGRESS_NAME,
    }


def normalize_split(value: Any) -> str:
    split = str(value or "train").strip().lower()
    return split or "train"


def row_unsafe_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    permission_status = str(row.get("permission_status") or "").strip().lower()
    source_audio_path = str(row.get("source_audio_path") or "")
    target_audio_path = str(row.get("target_audio_path") or "")
    source_gate = row.get("source_gate") if isinstance(row.get("source_gate"), dict) else {}
    if row.get("schema") != "generic_finetuning_audio_training_example.v1":
        reasons.append("unexpected_schema")
    if row.get("trainable") is not True:
        reasons.append("not_marked_trainable")
    if permission_status not in TRAINABLE_PERMISSION_STATUSES:
        reasons.append("permission_not_trainable")
    if row.get("human_verified") is not True and row.get("reference_verified") is not True:
        reasons.append("not_human_or_reference_verified")
    if row.get("human_verified") is True:
        if source_gate.get("accepted_machine_gate") is not True:
            reasons.append("missing_accepted_machine_gate_provenance")
    elif row.get("reference_verified") is True:
        has_trainable_ground_truth = source_gate.get("trainable_ground_truth") is True
        has_exact_reference_audit = source_gate.get("exact_reference_waveform_audit_accepted") is True
        if not (has_trainable_ground_truth or has_exact_reference_audit):
            reasons.append("missing_trainable_ground_truth_provenance")
        if not has_exact_reference_audit and source_gate.get("full_mix_reconstruction_status") != "accepted":
            reasons.append("rejected_full_mix_reconstruction")
        if not has_exact_reference_audit and source_gate.get("source_mix_1_to_1_claim_allowed") is not True:
            reasons.append("source_mix_1_to_1_not_claimable")
    if row.get("release_ready") is True:
        reasons.append("input_row_release_ready_true")
    if row.get("delivery_allowed") is True:
        reasons.append("input_row_delivery_allowed_true")
    if not str(row.get("id") or ""):
        reasons.append("missing_id")
    if not str(row.get("track_id") or ""):
        reasons.append("missing_track_id")
    if not str(row.get("stem_type") or ""):
        reasons.append("missing_stem_type")
    if not source_audio_path:
        reasons.append("missing_source_audio_path")
    elif not Path(source_audio_path).exists():
        reasons.append("source_audio_not_found")
    if not target_audio_path:
        reasons.append("missing_target_audio_path")
    elif not Path(target_audio_path).exists():
        reasons.append("target_audio_not_found")
    return reasons


def validate_verified_audio_training_corpus(
    *,
    audio_training_examples_jsonl: Path,
    output_dir: Path,
    min_unique_tracks: int = 2,
    short_output_names: bool = False,
) -> dict[str, Any]:
    if min_unique_tracks <= 0:
        raise ValueError("min_unique_tracks must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(short_output_names=short_output_names)
    progress_path = output_dir / names["progress"]
    if progress_path.exists():
        progress_path.unlink()
    progress(
        progress_path,
        "start",
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        min_unique_tracks=min_unique_tracks,
    )
    rows = list(iter_jsonl(audio_training_examples_jsonl))
    progress(progress_path, "rows_loaded", row_count=len(rows))

    row_reports: list[dict[str, Any]] = []
    split_by_track: dict[str, set[str]] = defaultdict(set)
    verified_trainable_examples = 0
    for index, row in enumerate(rows, start=1):
        track_id = str(row.get("track_id") or "")
        split = normalize_split(row.get("split"))
        if track_id:
            split_by_track[track_id].add(split)
        reasons = row_unsafe_reasons(row)
        safe_for_training = not reasons
        if safe_for_training:
            verified_trainable_examples += 1
        row_reports.append(
            {
                "schema": "verified_audio_training_corpus_row.v1",
                "id": str(row.get("id") or stable_hash([str(audio_training_examples_jsonl), index])[:16]),
                "track_id": track_id,
                "stem_type": str(row.get("stem_type") or ""),
                "split": split,
                "source_audio_path": str(row.get("source_audio_path") or ""),
                "target_audio_path": str(row.get("target_audio_path") or ""),
                "human_verified": row.get("human_verified") is True,
                "reference_verified": row.get("reference_verified") is True,
                "permission_status": str(row.get("permission_status") or "").strip().lower(),
                "safe_for_training": safe_for_training,
                "unsafe_reasons": reasons,
                "release_ready": False,
            }
        )

    track_split_leakage = [
        {"track_id": track_id, "splits": sorted(splits)}
        for track_id, splits in sorted(split_by_track.items())
        if len(splits) > 1
    ]
    unique_tracks = sorted({report["track_id"] for report in row_reports if report["safe_for_training"] and report["track_id"]})
    failed_checks: list[str] = []
    if any(not report["safe_for_training"] for report in row_reports):
        failed_checks.append("unsafe_rows_present")
    if len(unique_tracks) < min_unique_tracks:
        failed_checks.append("insufficient_unique_tracks")
    if track_split_leakage:
        failed_checks.append("track_split_leakage")
    status = "passed" if not failed_checks else "failed"
    manifest = {
        "schema": "verified_audio_training_corpus_manifest.v1",
        "manifest_type": "verified_audio_training_corpus",
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "output_dir": str(output_dir),
        "manifest_json": str(output_dir / names["manifest"]),
        "corpus_validation_status": status,
        "failed_checks": failed_checks,
        "row_count": len(rows),
        "verified_trainable_examples": verified_trainable_examples,
        "unsafe_row_count": len(row_reports) - verified_trainable_examples,
        "unique_track_count": len(unique_tracks),
        "min_unique_tracks": int(min_unique_tracks),
        "track_split_leakage_count": len(track_split_leakage),
        "track_split_leakage": track_split_leakage,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "verified_training_corpus_artifact_not_for_listen_pack",
        "release_status": "training_corpus_validation_only",
        "release_ready": False,
        "short_output_names": short_output_names,
        "row_report_jsonl": str(output_dir / names["rows"]),
        "progress_jsonl": str(progress_path),
        "safety": {
            "metadata_only": True,
            "dirty_stems_never_ground_truth": True,
            "requires_human_or_reference_verification": True,
            "fail_closed_on_missing_audio": True,
            "track_split_leakage_blocks_training": True,
            "not_release_artifact": True,
        },
    }
    write_jsonl(output_dir / names["rows"], row_reports)
    write_json(output_dir / names["manifest"], manifest)
    progress(progress_path, "finish", corpus_validation_status=status, failed_checks=failed_checks)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a verified audio training corpus before GPU training.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-unique-tracks", type=int, default=2)
    parser.add_argument("--short-output-names", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="validate_verified_audio_training_corpus.py",
        entrypoint_path="/app/tools/validate_verified_audio_training_corpus.py",
    )
    args = parse_args(argv)
    manifest = validate_verified_audio_training_corpus(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        output_dir=args.output_dir,
        min_unique_tracks=args.min_unique_tracks,
        short_output_names=bool(args.short_output_names),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_json"],
                "corpus_validation_status": manifest["corpus_validation_status"],
                "verified_trainable_examples": manifest["verified_trainable_examples"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["corpus_validation_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
