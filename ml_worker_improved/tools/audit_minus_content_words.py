"""Fail-closed ASR content audit for possible word leakage in minus candidates.

This audit is independent diagnostic evidence. It can confirm source-matching
words in a candidate, but silence from ASR never proves that a minus is clean.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

from docker_runtime import require_docker_runtime


SCHEMA = "minus_content_word_audit.v1"
DEFAULT_MODELS = ("base", "small")
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def mark(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "elapsed_sec": round(time.monotonic() - self.started_at, 3),
            "expected_acceleration_mode": "cpu_int8_asr_diagnostic",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "apple_linux_container_has_no_mps_or_ane_access",
        }
        payload.update(fields)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def normalize_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(text).casefold()) if token]


def lcs_tokens(left: Sequence[str], right: Sequence[str]) -> list[str]:
    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    matched: list[str] = []
    i = len(left)
    j = len(right)
    while i > 0 and j > 0:
        if left[i - 1] == right[j - 1]:
            matched.append(left[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matched.reverse()
    return matched


def _segment_value(segment: Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(name, default)
    return getattr(segment, name, default)


def _word_rows(segment: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for word in _segment_value(segment, "words", None) or []:
        raw_word = _segment_value(word, "word", "")
        normalized = normalize_tokens(str(raw_word))
        if not normalized:
            continue
        rows.append(
            {
                "start": round(float(_segment_value(word, "start", 0.0) or 0.0), 3),
                "end": round(float(_segment_value(word, "end", 0.0) or 0.0), 3),
                "word": str(raw_word).strip(),
                "tokens": normalized,
                "probability": round(float(_segment_value(word, "probability", 0.0) or 0.0), 6),
            }
        )
    return rows


def materialize_transcript(
    segments: Sequence[Any],
    info: Any,
    *,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    max_compression_ratio: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reliable_tokens: list[str] = []
    for index, segment in enumerate(segments):
        text = str(_segment_value(segment, "text", "")).strip()
        tokens = normalize_tokens(text)
        no_speech_prob = float(_segment_value(segment, "no_speech_prob", 1.0) or 0.0)
        avg_logprob = float(_segment_value(segment, "avg_logprob", -99.0) or 0.0)
        compression_ratio = float(_segment_value(segment, "compression_ratio", 99.0) or 0.0)
        reliable = bool(
            tokens
            and no_speech_prob <= max_no_speech_prob
            and avg_logprob >= min_avg_logprob
            and compression_ratio <= max_compression_ratio
        )
        if reliable:
            reliable_tokens.extend(tokens)
        rows.append(
            {
                "segment_index": index,
                "start": round(float(_segment_value(segment, "start", 0.0) or 0.0), 3),
                "end": round(float(_segment_value(segment, "end", 0.0) or 0.0), 3),
                "text": text,
                "tokens": tokens,
                "word_timestamps": _word_rows(segment),
                "avg_logprob": round(avg_logprob, 6),
                "no_speech_prob": round(no_speech_prob, 6),
                "compression_ratio": round(compression_ratio, 6),
                "reliable": reliable,
            }
        )
    return {
        "language": str(getattr(info, "language", "") or ""),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 6),
        "duration_sec": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
        "text": " ".join(row["text"] for row in rows if row["text"]).strip(),
        "reliable_tokens": reliable_tokens,
        "reliable_token_count": len(reliable_tokens),
        "segments": rows,
    }


def evaluate_transcript_pair(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    min_matched_tokens: int,
    min_candidate_match_ratio: float,
    time_tolerance_sec: float,
) -> dict[str, Any]:
    source_segments = [row for row in source.get("segments", []) if row.get("reliable") is True]
    evidence_rows: list[dict[str, Any]] = []
    for candidate_segment in candidate.get("segments", []):
        if candidate_segment.get("reliable") is not True:
            continue
        window_start = float(candidate_segment["start"])
        window_end = float(candidate_segment["end"])
        aligned_source = [
            row
            for row in source_segments
            if float(row["start"]) < window_end + time_tolerance_sec
            and window_start - time_tolerance_sec < float(row["end"])
        ]
        source_tokens = [token for row in aligned_source for token in row.get("tokens", [])]
        candidate_tokens = list(candidate_segment.get("tokens", []))
        matched_tokens = lcs_tokens(source_tokens, candidate_tokens)
        match_ratio = len(matched_tokens) / max(1, len(candidate_tokens))
        confirmed = bool(
            len(candidate_tokens) >= min_matched_tokens
            and len(matched_tokens) >= min_matched_tokens
            and match_ratio >= min_candidate_match_ratio
        )
        evidence_rows.append(
            {
                "window_sec": [window_start, window_end],
                "candidate_text": candidate_segment.get("text", ""),
                "candidate_tokens": candidate_tokens,
                "aligned_source_tokens": source_tokens,
                "matched_tokens": matched_tokens,
                "matched_token_count": len(matched_tokens),
                "candidate_match_ratio": round(match_ratio, 6),
                "confirmed_source_matching_words": confirmed,
            }
        )
    confirmed_rows = [row for row in evidence_rows if row["confirmed_source_matching_words"]]
    if confirmed_rows:
        status = "confirmed_source_matching_words"
    elif int(candidate.get("reliable_token_count") or 0) == 0:
        status = "words_not_detected"
    elif int(source.get("reliable_token_count") or 0) == 0:
        status = "inconclusive_source_words_not_detected"
    else:
        status = "inconclusive_unmatched_candidate_words"
    return {
        "status": status,
        "confirmed": bool(confirmed_rows),
        "source_reliable_token_count": int(source.get("reliable_token_count") or 0),
        "candidate_reliable_token_count": int(candidate.get("reliable_token_count") or 0),
        "confirmed_windows": [row["window_sec"] for row in confirmed_rows],
        "evidence": evidence_rows,
    }


def aggregate_candidate(
    model_results: Sequence[Mapping[str, Any]],
    *,
    min_model_confirmations: int,
) -> dict[str, Any]:
    completed = [row for row in model_results if row.get("runtime_status") == "completed"]
    failed = [row for row in model_results if row.get("runtime_status") != "completed"]
    confirmations = [row for row in completed if row.get("comparison", {}).get("confirmed") is True]
    if len(confirmations) >= min_model_confirmations:
        status = "content_leak_confirmed"
    elif failed:
        status = "content_unknown_runtime_failure"
    elif confirmations:
        status = "content_inconclusive_single_model_evidence"
    elif completed and all(
        row.get("comparison", {}).get("status") == "words_not_detected" for row in completed
    ):
        status = "content_words_not_detected"
    else:
        status = "content_inconclusive"
    return {
        "status": status,
        "content_leak_confirmed": status == "content_leak_confirmed",
        "repair_eligible": status == "content_leak_confirmed",
        "clean_content_proven": False,
        "model_confirmation_count": len(confirmations),
        "required_model_confirmations": min_model_confirmations,
        "completed_model_count": len(completed),
        "failed_model_count": len(failed),
        "confirmed_models": [str(row.get("model_id") or "") for row in confirmations],
        "confirmed_windows_by_model": {
            str(row.get("model_id") or ""): row.get("comparison", {}).get("confirmed_windows", [])
            for row in confirmations
        },
    }


def _parse_candidate(value: str) -> tuple[str, Path]:
    candidate_id, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("candidate must use id=/path/to/audio form")
    candidate_id = candidate_id.strip().lower()
    if not SAFE_ID_RE.fullmatch(candidate_id):
        raise argparse.ArgumentTypeError("candidate id must match [a-z0-9][a-z0-9_]{0,31}")
    return candidate_id, Path(raw_path)


def _validate_project_path(path: Path, runtime_project_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(runtime_project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside runtime_project_root") from exc
    return resolved


def _windows_path(path: Path, runtime_project_root: Path, windows_project_root: str) -> str:
    relative = path.resolve().relative_to(runtime_project_root.resolve())
    return str(PureWindowsPath(windows_project_root).joinpath(*relative.parts))


def _mount_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"probe_{os.getpid()}.tmp"
    token = f"minus-content-audit:{os.getpid()}"
    path.write_text(token, encoding="utf-8")
    if path.read_text(encoding="utf-8") != token:
        raise RuntimeError(f"Artifact mount probe read mismatch at {path}")
    path.unlink()
    return {
        "probe_path": str(path),
        "write_read_verified": True,
        "cleanup_verified": not path.exists(),
    }


def _load_whisper_model(model_id: str, model_cache_dir: Path, cpu_threads: int) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
        download_root=str(model_cache_dir),
    )


def _transcribe(model: Any, audio_path: Path, language: str | None, beam_size: int) -> tuple[list[Any], Any]:
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=False,
        temperature=0.0,
    )
    return list(segments), info


def audit_minus_content_words(
    *,
    source_audio_path: Path,
    candidates: Sequence[tuple[str, Path]],
    output_dir: Path,
    model_ids: Sequence[str] = DEFAULT_MODELS,
    model_cache_dir: Path,
    runtime_project_root: Path,
    windows_project_root: str,
    language: str | None = None,
    beam_size: int = 5,
    cpu_threads: int = 4,
    min_model_confirmations: int = 2,
    min_matched_tokens: int = 2,
    min_candidate_match_ratio: float = 0.5,
    time_tolerance_sec: float = 2.0,
    max_no_speech_prob: float = 0.65,
    min_avg_logprob: float = -1.2,
    max_compression_ratio: float = 2.4,
    model_loader: Callable[[str, Path, int], Any] = _load_whisper_model,
    transcriber: Callable[[Any, Path, str | None, int], tuple[list[Any], Any]] = _transcribe,
) -> dict[str, Any]:
    if len(set(model_ids)) != len(model_ids) or not model_ids:
        raise ValueError("model_ids must be non-empty and unique")
    if not 1 <= min_model_confirmations <= len(model_ids):
        raise ValueError("min_model_confirmations must be between 1 and model count")
    if not candidates:
        raise ValueError("at least one candidate is required")
    candidate_ids = [candidate_id for candidate_id, _ in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")

    runtime_root = runtime_project_root.resolve()
    source_path = _validate_project_path(source_audio_path, runtime_root, "source_audio_path")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    checked_candidates: list[tuple[str, Path]] = []
    for candidate_id, candidate_path in candidates:
        if not SAFE_ID_RE.fullmatch(candidate_id):
            raise ValueError(f"invalid candidate id: {candidate_id}")
        resolved = _validate_project_path(candidate_path, runtime_root, f"candidate:{candidate_id}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        checked_candidates.append((candidate_id, resolved))
    output_path = _validate_project_path(output_dir, runtime_root, "output_dir")
    relative_output = output_path.relative_to(runtime_root)
    if not relative_output.parts or relative_output.parts[0] not in {"artifacts", "lab_artifacts", "artifact_rag"}:
        raise ValueError("output_dir must be inside an approved project artifact root")

    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    rows_path = output_path / "rows.jsonl"
    progress_path = output_path / "progress.jsonl"
    progress = ProgressWriter(progress_path)
    mount_probe = _mount_probe(output_path)
    progress.mark(
        "start",
        source_audio_path=str(source_path),
        candidate_count=len(checked_candidates),
        model_ids=list(model_ids),
        output_manifest_path=str(manifest_path),
    )

    audio_inputs = [("source", source_path), *checked_candidates]
    transcript_rows: list[dict[str, Any]] = []
    per_model: list[dict[str, Any]] = []
    candidate_model_results: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id, _ in checked_candidates
    }
    for model_id in model_ids:
        model_started = time.monotonic()
        progress.mark("model_load_start", model_id=model_id)
        try:
            model = model_loader(str(model_id), model_cache_dir, cpu_threads)
            progress.mark(
                "model_load_finish",
                model_id=model_id,
                model_load_elapsed_sec=round(time.monotonic() - model_started, 3),
            )
            transcripts: dict[str, dict[str, Any]] = {}
            for audio_id, audio_path in audio_inputs:
                phase_started = time.monotonic()
                progress.mark("transcribe_start", model_id=model_id, audio_id=audio_id)
                segments, info = transcriber(model, audio_path, language, beam_size)
                transcript = materialize_transcript(
                    segments,
                    info,
                    max_no_speech_prob=max_no_speech_prob,
                    min_avg_logprob=min_avg_logprob,
                    max_compression_ratio=max_compression_ratio,
                )
                transcripts[audio_id] = transcript
                for segment in transcript["segments"]:
                    transcript_rows.append({"model_id": model_id, "audio_id": audio_id, **segment})
                progress.mark(
                    "transcribe_finish",
                    model_id=model_id,
                    audio_id=audio_id,
                    reliable_token_count=transcript["reliable_token_count"],
                    phase_elapsed_sec=round(time.monotonic() - phase_started, 3),
                )
            source_transcript = transcripts["source"]
            comparisons: dict[str, Any] = {}
            for candidate_id, _ in checked_candidates:
                comparison = evaluate_transcript_pair(
                    source_transcript,
                    transcripts[candidate_id],
                    min_matched_tokens=min_matched_tokens,
                    min_candidate_match_ratio=min_candidate_match_ratio,
                    time_tolerance_sec=time_tolerance_sec,
                )
                comparisons[candidate_id] = comparison
                candidate_model_results[candidate_id].append(
                    {
                        "model_id": model_id,
                        "runtime_status": "completed",
                        "comparison": comparison,
                    }
                )
            per_model.append(
                {
                    "model_id": model_id,
                    "runtime_status": "completed",
                    "model_elapsed_sec": round(time.monotonic() - model_started, 3),
                    "source": source_transcript,
                    "candidates": {candidate_id: transcripts[candidate_id] for candidate_id, _ in checked_candidates},
                    "comparisons": comparisons,
                }
            )
        except Exception as exc:  # fail-closed evidence is written for every requested model
            error = f"{type(exc).__name__}: {exc}"
            per_model.append(
                {
                    "model_id": model_id,
                    "runtime_status": "failed",
                    "error": error,
                    "model_elapsed_sec": round(time.monotonic() - model_started, 3),
                }
            )
            for candidate_id, _ in checked_candidates:
                candidate_model_results[candidate_id].append(
                    {
                        "model_id": model_id,
                        "runtime_status": "failed",
                        "error": error,
                    }
                )
            progress.mark("model_failed", model_id=model_id, error=error)
        finally:
            if "model" in locals():
                del model
            gc.collect()

    candidate_summaries = []
    for candidate_id, candidate_path in checked_candidates:
        aggregate = aggregate_candidate(
            candidate_model_results[candidate_id],
            min_model_confirmations=min_model_confirmations,
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "path": str(candidate_path),
                "windows_path": _windows_path(candidate_path, runtime_root, windows_project_root),
                "sha256": sha256_file(candidate_path),
                **aggregate,
            }
        )

    confirmed = [row for row in candidate_summaries if row["content_leak_confirmed"]]
    runtime_failures = [row for row in per_model if row["runtime_status"] != "completed"]
    if confirmed:
        overall_status = "content_leak_confirmed"
    elif runtime_failures:
        overall_status = "content_unknown_runtime_failure"
    elif all(row["status"] == "content_words_not_detected" for row in candidate_summaries):
        overall_status = "content_words_not_detected_not_clean_proof"
    else:
        overall_status = "content_inconclusive"

    try:
        package_version = importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    manifest = {
        "schema": SCHEMA,
        "mode": "multi_model_source_matched_word_leak_audit",
        "source": {
            "path": str(source_path),
            "windows_path": _windows_path(source_path, runtime_root, windows_project_root),
            "sha256": sha256_file(source_path),
        },
        "models": per_model,
        "candidates": candidate_summaries,
        "overall": {
            "status": overall_status,
            "content_leak_confirmed": bool(confirmed),
            "confirmed_candidate_ids": [row["candidate_id"] for row in confirmed],
            "runtime_failure_count": len(runtime_failures),
            "clean_content_proven": False,
            "absence_interpretation": "asr_non_detection_is_inconclusive_and_never_proves_clean_minus",
        },
        "thresholds": {
            "min_model_confirmations": min_model_confirmations,
            "min_matched_tokens": min_matched_tokens,
            "min_candidate_match_ratio": min_candidate_match_ratio,
            "time_tolerance_sec": time_tolerance_sec,
            "max_no_speech_prob": max_no_speech_prob,
            "min_avg_logprob": min_avg_logprob,
            "max_compression_ratio": max_compression_ratio,
        },
        "execution_contract": {
            "route_name": "audit_minus_content_words",
            "runtime_environment": "apple_container",
            "faster_whisper_version": package_version,
            "model_ids": list(model_ids),
            "device": "cpu",
            "compute_type": "int8",
            "expected_acceleration_mode": "cpu_int8_asr_diagnostic",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "apple_linux_container_has_no_mps_or_ane_access",
            "separation_model_runtime_affected": False,
            "fallback_used": False,
            "diagnostic_only": True,
        },
        "mount_contract": {
            **mount_probe,
            "runtime_project_root": str(runtime_root),
            "windows_project_root": windows_project_root,
            "runtime_output_path": str(output_path),
            "windows_output_path": _windows_path(output_path, runtime_root, windows_project_root),
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "rows": str(rows_path),
            "progress": str(progress_path),
        },
        "delivery_contract": {
            "delivery_allowed": False,
            "delivery_status": "not_a_delivery_artifact",
            "presentation_status": "diagnostic_content_evidence_only",
            "listen_pack_allowed": False,
            "acceptance_blocker_reasons": ["exact_reference_and_full_quality_gate_required"],
        },
    }
    write_jsonl(rows_path, transcript_rows)
    write_json(manifest_path, manifest)
    progress.mark(
        "finish",
        status=overall_status,
        confirmed_candidate_ids=manifest["overall"]["confirmed_candidate_ids"],
        manifest_path=str(manifest_path),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audio")
    parser.add_argument("output_dir")
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--model-cache-dir", default="/models")
    parser.add_argument("--runtime-project-root", required=True)
    parser.add_argument("--windows-project-root", required=True)
    parser.add_argument("--language")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--min-model-confirmations", type=int, default=2)
    parser.add_argument("--min-matched-tokens", type=int, default=2)
    parser.add_argument("--min-candidate-match-ratio", type=float, default=0.5)
    parser.add_argument("--time-tolerance-sec", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_minus_content_words.py",
        entrypoint_path="/app/tools/audit_minus_content_words.py",
    )
    args = parse_args(argv)
    manifest = audit_minus_content_words(
        source_audio_path=Path(args.source_audio),
        candidates=args.candidate,
        output_dir=Path(args.output_dir),
        model_ids=tuple(args.models or DEFAULT_MODELS),
        model_cache_dir=Path(args.model_cache_dir),
        runtime_project_root=Path(args.runtime_project_root),
        windows_project_root=str(args.windows_project_root),
        language=str(args.language) if args.language else None,
        beam_size=int(args.beam_size),
        cpu_threads=int(args.cpu_threads),
        min_model_confirmations=int(args.min_model_confirmations),
        min_matched_tokens=int(args.min_matched_tokens),
        min_candidate_match_ratio=float(args.min_candidate_match_ratio),
        time_tolerance_sec=float(args.time_tolerance_sec),
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["artifacts"]["manifest"],
                "status": manifest["overall"]["status"],
                "confirmed_candidate_ids": manifest["overall"]["confirmed_candidate_ids"],
                "delivery_allowed": manifest["delivery_contract"]["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
