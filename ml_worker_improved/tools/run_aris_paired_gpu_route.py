from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_name_policy import track_alias, validate_artifact_relative_path
from docker_runtime import require_docker_runtime
from run_no_reference_result_judge import judge_candidate_row


AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "progress.jsonl"
JOBS_NAME = "jobs.jsonl"
WAVE_NAME = "wave.json"
SUMMARY_NAME = "summary.md"
START_TIME = time.monotonic()

CommandRunner = Callable[[list[str]], None]
StageCommandRunner = Callable[..., None]
JudgeRunner = Callable[[dict[str, Any]], dict[str, Any]]


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_inside_rag(*, artifact_rag_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = _resolved(artifact_rag_root)
    output = _resolved(output_dir)
    if output != root and root not in output.parents:
        raise ValueError("output_dir must be inside artifact_rag_root")
    try:
        validate_artifact_relative_path(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError("output_dir must use short Windows-safe relative path inside artifact_rag_root") from exc
    return root, output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_progress(path: Path, event: str, **fields: Any) -> None:
    row = {
        "event": event,
        "phase": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
        "gpu_inference_required": True,
        "gpu_inference_skipped": False,
        "delivery_allowed": False,
        "release_ready": False,
    }
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _discover_audio_files(corpus_dir: Path, *, limit: int) -> list[Path]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus_dir is missing: {corpus_dir}")
    files = sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if limit > 0:
        return files[:limit]
    return files


def _default_command_runner(command: list[str], *, stage: str, track_alias: str) -> None:
    del stage, track_alias
    subprocess.run(command, check=True)


def _default_judge_runner(row: dict[str, Any], *, score_mode: str) -> dict[str, Any]:
    return judge_candidate_row(row, score_mode=score_mode)


def _copy_required(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _artifact_ref(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    validate_artifact_relative_path(relative)
    return {
        "path": str(path),
        "relative_path": relative,
        "sha256": _sha256_file(path) if path.exists() else "",
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _candidate_manifest(
    *,
    path: Path,
    root: Path,
    track_id: str,
    track_alias_value: str,
    stem: str,
    audio_path: Path,
    source_audio_path: Path,
    model_family: str,
    model_id: str,
) -> dict[str, Any]:
    payload = {
        "schema": "aris_paired_gpu_stem_manifest.v1",
        "track_id": track_id,
        "track_alias": track_alias_value,
        "stem": stem,
        "target_stem": stem,
        "target_audio_path": str(audio_path),
        "short_audio_path": audio_path.relative_to(root).as_posix(),
        "source_audio_path": str(source_audio_path),
        "model_family": model_family,
        "model_id": model_id,
        "contract": {
            "application_reference_free": True,
            "target_reference_used": False,
            "fallback_used": False,
            "fallback_as_success": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
            "quality_gate_required": True,
        },
        "audio": _artifact_ref(audio_path, root=root),
        "delivery_allowed": False,
        "release_ready": False,
    }
    _write_json(path, payload)
    return payload


def _execution_manifest(
    *,
    path: Path,
    candidate_manifest_path: Path,
    candidate_payload: dict[str, Any],
    status: str,
    failed_checks: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "schema": "registered_model_candidate_execution_manifest.v1",
        "status": status,
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "target_stem": candidate_payload["stem"],
            "target_audio_path": candidate_payload["short_audio_path"],
            "binary_audio_available": candidate_payload["audio"]["bytes"] > 0,
            "audio_bytes_available": candidate_payload["audio"]["bytes"] > 0,
            "metadata_only": candidate_payload["audio"]["bytes"] <= 0,
        },
        "model_id": candidate_payload["model_id"],
        "failed_checks": list(failed_checks),
        "delivery_allowed": False,
        "release_ready": False,
        "fallback_used": False,
        "fallback_as_success": False,
    }
    _write_json(path, payload)
    return payload


def _failure_labels(judged: dict[str, Any], *, decision_key: str, accepted_value: str, blocker_keys: Sequence[str]) -> list[str]:
    if str(judged.get(decision_key) or "") == accepted_value:
        return []
    labels: list[str] = [str(judged.get(decision_key) or "gate_failed")]
    for key in blocker_keys:
        value = judged.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    labels.append(str(item.get("label") or item.get("reason") or "gate_blocker"))
                else:
                    labels.append(str(item))
    for key in ("failed_checks", "reasons"):
        value = judged.get(key)
        if isinstance(value, list):
            labels.extend(str(item) for item in value if str(item))
    return sorted({label for label in labels if label})


def _judge_pair(
    *,
    track_id: str,
    candidate_id: str,
    mix_path: Path,
    vocal_path: Path,
    minus_path: Path,
    candidate_manifest_path: Path,
    judge_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    row = {
        "track_id": track_id,
        "candidate_id": candidate_id,
        "mix_path": str(mix_path),
        "vocal_path": str(vocal_path),
        "instrumental_path": str(minus_path),
        "candidate_manifest": str(candidate_manifest_path),
    }
    vocal = judge_runner(row, score_mode="vocal")
    minus = judge_runner(row, score_mode="minus")
    pair = judge_runner(row, score_mode="both")
    failures = []
    failures.extend(
        _failure_labels(
            vocal,
            decision_key="vocal_decision",
            accepted_value="accept_candidate",
            blocker_keys=("vocal_blockers",),
        )
    )
    failures.extend(
        _failure_labels(
            minus,
            decision_key="minus_decision",
            accepted_value="accept_candidate",
            blocker_keys=("minus_blockers",),
        )
    )
    detail_gate = minus.get("minus_musical_detail_preservation_gate")
    if isinstance(detail_gate, dict) and detail_gate.get("status") != "proxy_pass":
        failures.extend(
            _failure_labels(
                {"detail_gate": "minus_detail_gate_blocked", "blockers": detail_gate.get("blockers") or []},
                decision_key="detail_gate",
                accepted_value="proxy_pass",
                blocker_keys=("blockers",),
            )
        )
    failures.extend(
        _failure_labels(
            pair,
            decision_key="pair_decision",
            accepted_value="pair_compatible",
            blocker_keys=("pair_blockers",),
        )
    )
    failed_checks = sorted({label for label in failures if label})
    accepted = not failed_checks
    return {
        "schema": "aris_paired_gpu_no_reference_gate.v1",
        "track_id": track_id,
        "candidate_id": candidate_id,
        "accepted": accepted,
        "failed_checks": failed_checks,
        "vocal_judge": vocal,
        "minus_judge": minus,
        "pair_judge": pair,
        "reference_gate": "not_available",
        "no_reference_gate": "passed" if accepted else "failed",
        "pair_consistency_gate": "passed" if accepted else "failed",
    }


def _failed_track_jobs(
    *,
    track_id: str,
    track_alias_value: str,
    source_audio_path: Path,
    failed_checks: Sequence[str],
    meta_dir: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for stem, family, model_id, alias, exec_name in (
        ("vocal", "scnet", "scnet_xl_ihf_musdb18", "v001", "vexec.json"),
        ("minus", "bs_roformer", "bs_roformer_instrumental_resurrection_unwa", "m001", "mexec.json"),
    ):
        exec_path = meta_dir / exec_name
        _write_json(
            exec_path,
            {
                "schema": "registered_model_candidate_execution_manifest.v1",
                "status": "failed",
                "candidate_manifest": {
                    "target_stem": stem,
                    "target_audio_path": f"audio/{track_alias_value}/{stem}.flac",
                    "binary_audio_available": False,
                    "audio_bytes_available": False,
                    "metadata_only": True,
                },
                "failed_checks": list(failed_checks),
                "delivery_allowed": False,
                "release_ready": False,
            },
        )
        jobs.append(
            {
                "schema": "registered_candidate_family_wave_job.v1",
                "id": f"{track_alias_value}-{stem}",
                "track_id": track_id,
                "track_alias": track_alias_value,
                "stem_type": stem,
                "source_audio_path": str(source_audio_path),
                "source_candidate_family": family,
                "source_candidate_model_id": model_id,
                "candidate_alias": alias,
                "execution_manifest_path": str(exec_path),
                "status": "failed",
                "candidate_status": "failed",
                "reference_gate": "not_available",
                "no_reference_gate": "failed",
                "pair_consistency_gate": "failed",
                "failed_checks": list(failed_checks),
                "binary_audio_available": False,
                "audio_bytes_available": False,
                "metadata_only": True,
                "delivery_allowed": False,
                "release_ready": False,
                "fallback_used": False,
                "fallback_as_success": False,
            }
        )
    return jobs


def run_aris_paired_gpu_route(
    *,
    artifact_rag_root: Path,
    corpus_dir: Path,
    output_dir: Path,
    limit: int = 0,
    device_id: int = 0,
    command_runner: Callable[..., None] | None = None,
    judge_runner: Callable[..., dict[str, Any]] | None = None,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    root, output = _ensure_output_inside_rag(artifact_rag_root=artifact_rag_root, output_dir=output_dir)
    runner = command_runner or _default_command_runner
    judge = judge_runner or _default_judge_runner
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / PROGRESS_NAME
    progress_path.write_text("", encoding="utf-8")
    _append_progress(
        progress_path,
        "run_start",
        route_name="aris_paired_gpu_route",
        corpus_dir=str(corpus_dir),
        output_dir=str(output),
        device_id=int(device_id),
        expected_output_manifest=str(output / MANIFEST_NAME),
    )
    source_files = _discover_audio_files(corpus_dir, limit=int(limit))
    jobs: list[dict[str, Any]] = []
    track_results: list[dict[str, Any]] = []
    for index, source_audio_path in enumerate(source_files, start=1):
        track_alias_value = track_alias(index)
        track_id = _sha256_file(source_audio_path)[:16]
        audio_dir = output / "audio" / track_alias_value
        work_dir = output / "work" / track_alias_value
        meta_dir = output / "meta" / track_alias_value
        scnet_dir = work_dir / "scn"
        minus_dir = work_dir / "irm"
        vocal_path = audio_dir / "vocal.flac"
        minus_path = audio_dir / "minus.flac"
        mix_path = audio_dir / "mix.flac"
        candidate_id = f"{track_alias_value}-p001"
        _append_progress(
            progress_path,
            "track_start",
            track_alias=track_alias_value,
            track_id=track_id,
            source_audio_path=str(source_audio_path),
            candidate_id=candidate_id,
        )
        try:
            scnet_command = [
                sys.executable,
                str(TOOLS_ROOT / "export_scnet_full.py"),
                str(source_audio_path),
                str(scnet_dir),
                "--device-id",
                str(int(device_id)),
            ]
            _append_progress(progress_path, "model_start", track_alias=track_alias_value, model_family="scnet")
            runner(scnet_command, stage="scnet", track_alias=track_alias_value)
            _append_progress(progress_path, "model_finish", track_alias=track_alias_value, model_family="scnet")

            minus_command = [
                sys.executable,
                str(TOOLS_ROOT / "export_instrumental_resurrection_minus.py"),
                str(source_audio_path),
                str(minus_dir),
                "--device",
                f"cuda:{int(device_id)}",
                "--minus-output-name",
                "minus.flac",
                "--progress-name",
                "prog.jsonl",
            ]
            _append_progress(progress_path, "model_start", track_alias=track_alias_value, model_family="bs_roformer")
            runner(minus_command, stage="minus", track_alias=track_alias_value)
            _append_progress(progress_path, "model_finish", track_alias=track_alias_value, model_family="bs_roformer")

            _copy_required(scnet_dir / "00_raw_ref.flac", mix_path)
            _copy_required(scnet_dir / "vocals.flac", vocal_path)
            _copy_required(minus_dir / "minus.flac", minus_path)
            vocal_manifest_path = meta_dir / "vstem.json"
            minus_manifest_path = meta_dir / "mstem.json"
            vocal_payload = _candidate_manifest(
                path=vocal_manifest_path,
                root=output,
                track_id=track_id,
                track_alias_value=track_alias_value,
                stem="vocal",
                audio_path=vocal_path,
                source_audio_path=source_audio_path,
                model_family="scnet",
                model_id="scnet_xl_ihf_musdb18",
            )
            minus_payload = _candidate_manifest(
                path=minus_manifest_path,
                root=output,
                track_id=track_id,
                track_alias_value=track_alias_value,
                stem="minus",
                audio_path=minus_path,
                source_audio_path=source_audio_path,
                model_family="bs_roformer",
                model_id="bs_roformer_instrumental_resurrection_unwa",
            )
            _append_progress(
                progress_path,
                "gate_start",
                track_alias=track_alias_value,
                candidate_id=candidate_id,
                expected_acceleration_mode="cpu_dsp_scoring",
                gpu_inference_required=False,
                gpu_inference_skipped=True,
                gpu_inference_skip_reason="no_reference_gate_reads_generated_audio",
            )
            gate = _judge_pair(
                track_id=track_id,
                candidate_id=candidate_id,
                mix_path=mix_path,
                vocal_path=vocal_path,
                minus_path=minus_path,
                candidate_manifest_path=vocal_manifest_path,
                judge_runner=judge,
            )
            _write_json(meta_dir / "judge.json", gate)
            candidate_status = "passed" if gate["accepted"] else "failed"
            failed_checks = list(gate["failed_checks"])
            vocal_exec = _execution_manifest(
                path=meta_dir / "vexec.json",
                candidate_manifest_path=vocal_manifest_path,
                candidate_payload=vocal_payload,
                status="completed",
                failed_checks=failed_checks,
            )
            minus_exec = _execution_manifest(
                path=meta_dir / "mexec.json",
                candidate_manifest_path=minus_manifest_path,
                candidate_payload=minus_payload,
                status="completed",
                failed_checks=failed_checks,
            )
            for stem, family, model_id, alias, exec_path, audio_path, execution_manifest in (
                ("vocal", "scnet", "scnet_xl_ihf_musdb18", "v001", meta_dir / "vexec.json", vocal_path, vocal_exec),
                (
                    "minus",
                    "bs_roformer",
                    "bs_roformer_instrumental_resurrection_unwa",
                    "m001",
                    meta_dir / "mexec.json",
                    minus_path,
                    minus_exec,
                ),
            ):
                audio_available = audio_path.exists() and audio_path.stat().st_size > 0
                jobs.append(
                    {
                        "schema": "registered_candidate_family_wave_job.v1",
                        "id": f"{track_alias_value}-{stem}",
                        "track_id": track_id,
                        "track_alias": track_alias_value,
                        "stem_type": stem,
                        "source_audio_path": str(source_audio_path),
                        "source_candidate_family": family,
                        "source_candidate_model_id": model_id,
                        "candidate_alias": alias,
                        "execution_manifest_path": str(exec_path),
                        "status": "completed",
                        "candidate_status": candidate_status,
                        "reference_gate": gate["reference_gate"],
                        "no_reference_gate": gate["no_reference_gate"],
                        "pair_consistency_gate": gate["pair_consistency_gate"],
                        "failed_checks": failed_checks,
                        "short_audio_path": f"audio/{track_alias_value}/{stem}.flac",
                        "binary_audio_available": audio_available,
                        "audio_bytes_available": audio_available,
                        "metadata_only": not audio_available,
                        "delivery_allowed": False,
                        "release_ready": False,
                        "fallback_used": False,
                        "fallback_as_success": False,
                        "execution_status": execution_manifest["status"],
                    }
                )
            track_results.append(
                {
                    "track_alias": track_alias_value,
                    "track_id": track_id,
                    "candidate_id": candidate_id,
                    "status": "accepted" if gate["accepted"] else "blocked",
                    "failed_checks": failed_checks,
                    "audio": {
                        "mix": _artifact_ref(mix_path, root=output),
                        "vocal": _artifact_ref(vocal_path, root=output),
                        "minus": _artifact_ref(minus_path, root=output),
                    },
                }
            )
            _append_progress(
                progress_path,
                "gate_finish",
                track_alias=track_alias_value,
                status="accepted" if gate["accepted"] else "blocked",
                failed_checks=failed_checks,
                expected_acceleration_mode="cpu_dsp_scoring",
                gpu_inference_required=False,
                gpu_inference_skipped=True,
                gpu_inference_skip_reason="no_reference_gate_reads_generated_audio",
            )
        except Exception as exc:
            failed_checks = [f"route_exception:{exc.__class__.__name__}"]
            jobs.extend(
                _failed_track_jobs(
                    track_id=track_id,
                    track_alias_value=track_alias_value,
                    source_audio_path=source_audio_path,
                    failed_checks=failed_checks,
                    meta_dir=meta_dir,
                )
            )
            track_results.append(
                {
                    "track_alias": track_alias_value,
                    "track_id": track_id,
                    "candidate_id": candidate_id,
                    "status": "blocked",
                    "failed_checks": failed_checks,
                    "error": str(exc),
                }
            )
            _append_progress(
                progress_path,
                "gate_finish",
                track_alias=track_alias_value,
                status="blocked",
                failed_checks=failed_checks,
                error=str(exc),
                expected_acceleration_mode="cpu_dsp_scoring",
                gpu_inference_required=False,
                gpu_inference_skipped=True,
                gpu_inference_skip_reason="no_reference_gate_or_route_failed",
            )
        _append_progress(progress_path, "track_finish", track_alias=track_alias_value)

    jobs_path = output / JOBS_NAME
    wave_path = output / WAVE_NAME
    _write_jsonl(jobs_path, jobs)
    accepted_pair_count = sum(1 for row in track_results if row["status"] == "accepted")
    failed_pair_count = sum(1 for row in track_results if row["status"] != "accepted")
    wave_manifest = {
        "schema": "registered_candidate_family_wave_manifest.v1",
        "manifest_type": "registered_candidate_family_wave",
        "status": "completed" if source_files else "blocked",
        "stem": "vocal_minus",
        "jobs_jsonl": str(jobs_path),
        "track_count": len(source_files),
        "job_count": len(jobs),
        "delivery_allowed": False,
        "release_ready": False,
    }
    _write_json(wave_path, wave_manifest)
    status = "completed" if accepted_pair_count and failed_pair_count == 0 else "blocked"
    manifest = {
        "schema": "aris_paired_gpu_route_manifest.v1",
        "manifest_type": "aris_paired_gpu_route",
        "status": status,
        "artifact_rag_root": str(root),
        "output_dir": str(output),
        "corpus_dir": str(corpus_dir),
        "progress_jsonl": str(progress_path),
        "wave_manifest_json": str(wave_path),
        "jobs_jsonl": str(jobs_path),
        "summary_md": str(output / SUMMARY_NAME),
        "track_count": len(source_files),
        "job_count": len(jobs),
        "accepted_pair_count": accepted_pair_count,
        "failed_pair_count": failed_pair_count,
        "delivery_status": "blocked",
        "delivery_allowed": False,
        "presentation_status": "diagnostic_only_not_listen_pack",
        "release_ready": False,
        "can_publish_as_result": False,
        "track_results": track_results,
        "contract": {
            "rag_controlled_outputs_only": True,
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "external_only": False,
            "no_human_listening_required": True,
            "rejected_candidates_are_diagnostic_only": True,
            "fallback_as_success": False,
            "per_track_hardcode_allowed": False,
            "qwen_controller_forbidden": True,
            "diffusion_repair_forbidden": True,
        },
        "command_args": list(command_args or []),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    (output / SUMMARY_NAME).write_text(
        "\n".join(
            [
                "# ARIS Paired GPU Route",
                "",
                f"- status: `{manifest['status']}`",
                f"- track_count: `{manifest['track_count']}`",
                f"- accepted_pair_count: `{manifest['accepted_pair_count']}`",
                f"- failed_pair_count: `{manifest['failed_pair_count']}`",
                "- delivery_allowed: `false`",
                "- presentation_status: `diagnostic_only_not_listen_pack`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _append_progress(
        progress_path,
        "run_finish",
        status=status,
        manifest_path=str(output / MANIFEST_NAME),
        accepted_pair_count=accepted_pair_count,
        failed_pair_count=failed_pair_count,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG-controlled paired SCNet vocal + BS-RoFormer minus GPU route.")
    parser.add_argument("--artifact-rag-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_aris_paired_gpu_route.py",
        entrypoint_path="/app/tools/run_aris_paired_gpu_route.py",
    )
    args = parse_args(argv)
    manifest = run_aris_paired_gpu_route(
        artifact_rag_root=args.artifact_rag_root,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        device_id=args.device_id,
        command_args=list(argv or sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
