#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402

MANIFEST_NAME = "generic_finetuning_corpus_manifest.json"
INVENTORY_NAME = "inventory.jsonl"
CONTROLLER_NAME = "controller_decision_examples.jsonl"
AUDIO_TRAIN_NAME = "audio_training_examples.jsonl"
HARD_EVAL_NAME = "hard_eval_cases.jsonl"
EXCLUDED_NAME = "excluded_examples.jsonl"
PROGRESS_NAME = "generic_finetuning_corpus_progress.jsonl"

ACCEPTED_PRESENTATION_STATUSES = {"accepted_for_listen_pack"}
MACHINE_STEM_ACCEPTANCE_GATE_SCHEMA = "machine_stem_acceptance_gate.v1"
MACHINE_STEM_ACCEPTANCE_GATE_TYPES = {"machine_stem_acceptance_gate"}
TRAINING_REQUIRED_MACHINE_GATES = {
    "aggregate_pack_validation_passed",
    "full_required_stem_coverage",
    "required_stem_machine_quality_evidence",
    "source_vocal_minus_master_pair_compatible",
    "minus_quality_audit_passed_without_warnings",
}
POSITIVE_RATINGS = {"ideal", "good", "accepted", "accept", "pass", "approved", "better", "prefer", "preferred"}
NEGATIVE_RATINGS = {"bad", "reject", "rejected", "fail", "failed", "poor", "worse"}
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
GROUND_TRUTH_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned", "user_owned"}
BLOCKED_MARKERS = ("blocked", "rejected", "failed", "skip")
BLOCKING_FAILURE_LABEL_MARKERS = (
    *BLOCKED_MARKERS,
    "leak",
    "bleed",
    "contamination",
    "foreign_source",
    "mix_conservation",
    "conservation",
    "vocal_contains",
    "voice_presence",
    "residual",
    "separator_server",
    "gpu_free_memory",
    "delivery_pair",
    "no_clean_candidate",
)
FAILURE_TEXT_KEYS = ("dominant_failure_reason", "failure_reason", "reason", "skip_reason")
FAILURE_LIST_KEYS = (
    "reasons",
    "failure_reasons",
    "blocking_issue_labels",
    "issue_labels",
    "blockers",
    "blocking_reasons",
    "failed_checks",
)
FAILURE_ITEM_KEYS = ("code", "label", "name", "id", "check")
BOOLEAN_BLOCKER_KEYS = {"blocked", "is_blocked", "has_blocker", "has_blockers"}

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}


@dataclass(frozen=True)
class HumanFeedback:
    track_id: str
    stem_type: str
    rating: str
    human_verified: bool
    permission_status: str
    split: str
    notes: str
    manifest_path: str = ""
    manifest_path_contains: str = ""
    artifact_path_contains: str = ""
    human_score: float | None = None


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def mark(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "elapsed_sec": round(time.monotonic() - START_TIME, 3),
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "curation_reads_existing_manifests_only",
        }
        payload.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > 5_000_000:
            return None
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_path_text(value: Any) -> str:
    return str(value or "").replace("\\", "/").lower()


def first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def stem_type(payload: dict[str, Any], path: Path) -> str:
    explicit = str(payload.get("stem_type") or payload.get("target_stem") or payload.get("stem") or "").lower()
    if explicit in {"vocals"}:
        return "vocal"
    if explicit in {"instrumental"}:
        return "minus"
    if explicit:
        return explicit
    text = normalize_path_text(path)
    for value in ("full_stem_pack", "minus", "vocal", "melody", "drums", "bass", "other", "both"):
        if value in text:
            return value
    return "unknown"


def track_id(payload: dict[str, Any]) -> str:
    return str(payload.get("track_id") or payload.get("track") or "")


def model_id(payload: dict[str, Any]) -> str:
    for key in ("model_id", "model_name", "base_candidate_id", "selected_candidate_id", "candidate_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    selected = payload.get("selected_candidate")
    if isinstance(selected, dict):
        for key in ("model_id", "model_name", "candidate_id"):
            value = selected.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def selected_audio_path(payload: dict[str, Any]) -> str:
    direct_keys = (
        "selected_vocal_path",
        "rendered_vocal_candidate_path",
        "selected_minus_path",
        "minus_path",
        "instrumental_path",
        "audio_path",
        "rendered_audio_path",
        "selected_audio_path",
    )
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    stems = payload.get("stems")
    if isinstance(stems, dict):
        for stem in ("vocals", "vocal", "instrumental", "minus", "melody", "drums", "bass", "other"):
            value = stems.get(stem)
            if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]:
                return value["path"]
    selected = payload.get("selected_candidate")
    if isinstance(selected, dict):
        for key in direct_keys:
            value = selected.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def source_audio_path(payload: dict[str, Any]) -> str:
    for key in ("source_audio_path", "mix_path", "raw_audio_path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    selected = payload.get("selected_candidate")
    if isinstance(selected, dict):
        for key in ("source_audio_path", "mix_path"):
            value = selected.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def docker_visible_manifest_path(manifest_path: Path, candidate_path: Any) -> str:
    text = str(candidate_path or "").strip()
    if not text:
        return ""
    if text.startswith("/"):
        return text
    return str(manifest_path.parent / text).replace("\\", "/")


def delivery_status(payload: dict[str, Any]) -> str:
    return str(payload.get("delivery_status") or payload.get("status") or "")


def presentation_status(payload: dict[str, Any]) -> str:
    return str(payload.get("presentation_status") or "")


def release_status(payload: dict[str, Any]) -> str:
    return str(payload.get("release_status") or "")


def accepted_contract(payload: dict[str, Any]) -> bool:
    return payload.get("delivery_allowed") is True and presentation_status(payload) in ACCEPTED_PRESENTATION_STATUSES


def accepted_machine_gate_contract(payload: dict[str, Any]) -> bool:
    schema = str(payload.get("schema") or "")
    manifest_type = str(payload.get("manifest_type") or "")
    if schema != MACHINE_STEM_ACCEPTANCE_GATE_SCHEMA and manifest_type not in MACHINE_STEM_ACCEPTANCE_GATE_TYPES:
        return False
    if str(payload.get("gate_status") or "").lower() != "accepted":
        return False
    if payload.get("final_result_allowed") is not True:
        return False
    if delivery_status(payload).lower() != "accepted":
        return False
    if payload.get("delivery_allowed") is not True:
        return False
    if presentation_status(payload) not in ACCEPTED_PRESENTATION_STATUSES:
        return False
    if payload.get("blocker_count") != 0:
        return False
    if payload.get("failed_checks") != []:
        return False
    required_gates = payload.get("required_machine_gates")
    if not isinstance(required_gates, list):
        return False
    return TRAINING_REQUIRED_MACHINE_GATES.issubset({str(gate) for gate in required_gates})


def label_blocked_like(labels: Sequence[str]) -> bool:
    text = " ".join(str(label or "") for label in labels).lower()
    return any(marker in text for marker in BLOCKING_FAILURE_LABEL_MARKERS)


def explicit_blocked_flag(payload: Any, *, root: bool = True) -> bool:
    if isinstance(payload, dict):
        for key, child in payload.items():
            normalized_key = str(key).lower()
            if normalized_key in BOOLEAN_BLOCKER_KEYS and child is True:
                return True
            if root and normalized_key == "delivery_allowed" and child is False:
                return True
            if explicit_blocked_flag(child, root=False):
                return True
    elif isinstance(payload, list):
        return any(explicit_blocked_flag(child, root=False) for child in payload)
    return False


def blocked_like(payload: dict[str, Any], failure_labels: Sequence[str] | None = None) -> bool:
    text = " ".join(
        [
            delivery_status(payload),
            presentation_status(payload),
            release_status(payload),
            str(payload.get("dominant_failure_reason") or ""),
        ]
    ).lower()
    return (
        explicit_blocked_flag(payload)
        or any(marker in text for marker in BLOCKED_MARKERS)
        or label_blocked_like(failure_labels or deep_failure_labels(payload))
    )


def reference_verified(payload: dict[str, Any]) -> bool:
    contracts = (
        payload.get("ground_truth_contract"),
        payload.get("exact_reference_audit"),
        payload.get("exact_reference_listen_audit"),
        payload.get("reference_contract"),
    )
    for contract in contracts:
        if isinstance(contract, dict) and (
            contract.get("verified") is True
            or str(contract.get("delivery_status") or contract.get("status") or "").lower() == "accepted"
        ):
            return True
    return False


def deep_failure_labels(payload: Any) -> list[str]:
    labels: list[str] = []

    def add_label(value: Any) -> None:
        for part in str(value or "").split(","):
            clean = part.strip()
            if clean:
                labels.append(clean)

    def add_structured_item(value: Any) -> None:
        if isinstance(value, dict):
            for item_key in (*FAILURE_ITEM_KEYS, *FAILURE_TEXT_KEYS):
                child = value.get(item_key)
                if isinstance(child, str) and child.strip():
                    add_label(child)
            for item_key, child in value.items():
                if isinstance(item_key, str) and child is True:
                    add_label(item_key)
        else:
            add_label(value)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in FAILURE_TEXT_KEYS and isinstance(child, str):
                    add_label(child)
                elif key in FAILURE_LIST_KEYS and isinstance(child, list):
                    for item in child:
                        add_structured_item(item)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        clean = label.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result[:20]


def discover_human_feedback_jsonls(artifact_root: Path, *, output_dir: Path | None = None) -> list[Path]:
    feedback_paths: list[Path] = []
    generated_names = {
        INVENTORY_NAME,
        CONTROLLER_NAME,
        AUDIO_TRAIN_NAME,
        HARD_EVAL_NAME,
        EXCLUDED_NAME,
        PROGRESS_NAME,
    }
    output_root = output_dir.resolve() if output_dir else None
    for path in artifact_root.rglob("*.jsonl"):
        lower_name = path.name.lower()
        if path.name in generated_names:
            continue
        if output_root and output_root in path.resolve().parents:
            continue
        if not any(marker in lower_name for marker in ("feedback", "human", "review", "validated_review_decisions")):
            continue
        feedback_paths.append(path)
    return sorted(feedback_paths)


def load_human_feedback(paths: Sequence[Path]) -> list[HumanFeedback]:
    feedback: list[HumanFeedback] = []
    for path in paths:
        for row in iter_jsonl(path):
            rating = str(
                row.get("rating")
                or row.get("verdict")
                or row.get("human_rating")
                or row.get("human_ab_verdict")
                or row.get("reviewer_decision")
                or row.get("decision")
                or ""
            ).strip().lower()
            feedback.append(
                HumanFeedback(
                    track_id=str(row.get("track_id") or ""),
                    stem_type=str(row.get("stem_type") or row.get("target_stem") or "").lower(),
                    rating=rating,
                    human_verified=row.get("human_verified") is True or row.get("listener_verified") is True,
                    permission_status=str(row.get("permission_status") or "requires_review"),
                    split=str(row.get("split") or row.get("feedback_split") or "train"),
                    notes=str(row.get("notes") or row.get("reason") or ""),
                    manifest_path=first_text(row, ("manifest_path", "ab_pack_manifest_path")),
                    manifest_path_contains=first_text(row, ("manifest_path_contains",)),
                    artifact_path_contains=first_text(
                        row,
                        (
                            "artifact_path_contains",
                            "source_artifact",
                            "candidate_audio_path",
                            "candidate_clip_path",
                            "source_audio_path",
                            "baseline_clip_path",
                        ),
                    ),
                    human_score=float(row["human_score"]) if isinstance(row.get("human_score"), (int, float)) else None,
                )
            )
    return feedback


def feedback_match_score(feedback: HumanFeedback, *, path: Path, row: dict[str, Any]) -> int:
    path_texts = [
        normalize_path_text(path),
        normalize_path_text(row.get("selected_audio_path")),
        normalize_path_text(row.get("source_audio_path")),
    ]
    path_text = path_texts[0]
    if feedback.manifest_path and normalize_path_text(feedback.manifest_path) == path_text:
        return 100
    if feedback.manifest_path_contains and normalize_path_text(feedback.manifest_path_contains) in path_text:
        return 80
    artifact_text = normalize_path_text(feedback.artifact_path_contains)
    if artifact_text and any(artifact_text in item or item in artifact_text for item in path_texts if item):
        return 70
    track_matches = feedback.track_id and feedback.track_id == str(row.get("track_id") or "")
    stem_matches = feedback.stem_type and feedback.stem_type == str(row.get("stem_type") or "")
    if track_matches and stem_matches:
        return 50
    if track_matches:
        return 30
    return 0


def best_feedback(feedback_rows: Sequence[HumanFeedback], *, path: Path, row: dict[str, Any]) -> HumanFeedback | None:
    scored = [(feedback_match_score(item, path=path, row=row), item) for item in feedback_rows]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].track_id, item[1].stem_type, item[1].rating))
    return scored[0][1]


def inventory_row(path: Path, artifact_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    audio_path = selected_audio_path(payload)
    audio_file = Path(audio_path) if audio_path else None
    failure_labels = deep_failure_labels(payload)
    return {
        "schema": "generic_finetuning_lab_artifact_inventory_row.v1",
        "manifest_path": str(path),
        "manifest_relpath": str(path.relative_to(artifact_root)),
        "track_id": track_id(payload),
        "stem_type": stem_type(payload, path),
        "model_id": model_id(payload),
        "delivery_status": delivery_status(payload),
        "delivery_allowed": payload.get("delivery_allowed") is True,
        "presentation_status": presentation_status(payload),
        "release_status": release_status(payload),
        "accepted_contract": accepted_contract(payload),
        "accepted_machine_gate_contract": accepted_machine_gate_contract(payload),
        "blocked_like": blocked_like(payload, failure_labels),
        "reference_verified": reference_verified(payload),
        "source_audio_path": source_audio_path(payload),
        "selected_audio_path": audio_path,
        "selected_audio_exists": bool(audio_file and audio_file.exists()),
        "selected_audio_sha256": None,
        "failure_labels": failure_labels,
    }


def controller_example(row: dict[str, Any], feedback: HumanFeedback | None) -> dict[str, Any]:
    accepted = bool(row["accepted_contract"] and not row["blocked_like"])
    if feedback and feedback.rating in NEGATIVE_RATINGS:
        accepted = False
    return {
        "schema": "generic_finetuning_controller_decision_example.v1",
        "id": stable_hash(["controller", row["manifest_path"]])[:16],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "input": {
            "manifest_path": row["manifest_path"],
            "model_id": row["model_id"],
            "delivery_status": row["delivery_status"],
            "presentation_status": row["presentation_status"],
            "release_status": row["release_status"],
            "failure_labels": row["failure_labels"],
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": row["model_id"],
            "acceptance_tests": ["artifact_status_contract", "human_or_reference_verification_before_audio_training"],
        },
        "result": {
            "accepted": accepted,
            "human_verified": bool(feedback and feedback.human_verified),
            "human_rating": feedback.rating if feedback else "",
            "reference_verified": row["reference_verified"],
            "accepted_machine_gate": row["accepted_machine_gate_contract"],
        },
        "trainable": accepted or row["blocked_like"],
        "training_target": "llm_controller",
    }


def controller_training_allowed(row: dict[str, Any], feedback: HumanFeedback | None) -> bool:
    if feedback and feedback.rating in NEGATIVE_RATINGS:
        return feedback.split == "train"
    if row["blocked_like"]:
        return bool(feedback and feedback.split == "train")
    if row["accepted_contract"]:
        return not (feedback and feedback.split == "hard_eval")
    return False


def audio_training_example(
    row: dict[str, Any],
    feedback: HumanFeedback,
    *,
    permission_default: str,
) -> dict[str, Any]:
    permission_status = feedback.permission_status or permission_default
    target_audio_path = Path(str(row["selected_audio_path"]))
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": stable_hash(["audio", row["manifest_path"], feedback.rating])[:16],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "manifest_path": row["manifest_path"],
        "source_audio_path": row["source_audio_path"],
        "target_audio_path": row["selected_audio_path"],
        "target_audio_sha256": sha256_file(target_audio_path),
        "model_id": row["model_id"],
        "rating": feedback.rating,
        "human_score": feedback.human_score,
        "human_verified": True,
        "permission_status": permission_status,
        "split": feedback.split,
        "notes": feedback.notes,
        "trainable": permission_status in TRAINABLE_PERMISSION_STATUSES,
        "training_target": "audio_separator_or_stem_adapter",
        "source_gate": {
            "accepted_machine_gate": row["accepted_machine_gate_contract"],
            "reference_verified": row["reference_verified"],
        },
        "safety": {
            "not_release_artifact": True,
            "requires_eval_before_promotion": True,
        },
    }


def permission_status_from_ground_truth(payload: dict[str, Any]) -> str:
    report = payload.get("report_alignment_contract")
    if not isinstance(report, dict):
        report = {}
    rights_status = str(report.get("rights_status") or payload.get("rights_status") or "").strip().lower()
    if rights_status == "user_owned":
        return "project_owned"
    return rights_status or "requires_review"


def ground_truth_source_gate(payload: dict[str, Any]) -> dict[str, Any]:
    full_mix_contract = payload.get("full_mix_reconstruction_contract")
    if not isinstance(full_mix_contract, dict):
        full_mix_contract = {}
    reconstruction_status = str(full_mix_contract.get("status") or "").strip().lower()
    source_mix_1_to_1_claim_allowed = full_mix_contract.get("can_claim_source_mix_1_to_1") is True
    blocker_reasons: list[str] = []
    if reconstruction_status != "accepted":
        blocker_reasons.append("full_mix_reconstruction_not_accepted")
    if not source_mix_1_to_1_claim_allowed:
        blocker_reasons.append("source_mix_1_to_1_not_claimable")
    return {
        "reference_verified": True,
        "trainable_ground_truth": not blocker_reasons,
        "full_mix_reconstruction_status": reconstruction_status or "missing",
        "source_mix_1_to_1_claim_allowed": source_mix_1_to_1_claim_allowed,
        "blocker_reasons": blocker_reasons,
    }


def ground_truth_context(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    contract = payload.get("ground_truth_contract")
    if not isinstance(contract, dict) or contract.get("verified") is not True:
        return None
    permission_status = permission_status_from_ground_truth(payload)
    if permission_status not in TRAINABLE_PERMISSION_STATUSES and permission_status not in GROUND_TRUTH_PERMISSION_STATUSES:
        return None
    mix = payload.get("mix")
    stems = payload.get("stems")
    if not isinstance(mix, dict) or not isinstance(stems, dict):
        return None
    source_path = docker_visible_manifest_path(path, mix.get("path"))
    if not source_path:
        return None
    return {
        "contract": contract,
        "permission_status": permission_status,
        "source_path": source_path,
        "stems": stems,
        "source_gate": ground_truth_source_gate(payload),
    }


def ground_truth_audio_training_examples(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    context = ground_truth_context(path, payload)
    if context is None:
        return []
    source_gate = context["source_gate"]
    if source_gate["trainable_ground_truth"] is not True:
        return []
    examples: list[dict[str, Any]] = []
    for raw_stem, stem_payload in context["stems"].items():
        if not isinstance(stem_payload, dict):
            continue
        target_path = docker_visible_manifest_path(path, stem_payload.get("path"))
        if not target_path:
            continue
        normalized_stem = "vocal" if raw_stem == "vocals" else ("minus" if raw_stem == "instrumental" else str(raw_stem))
        examples.append(
            {
                "schema": "generic_finetuning_audio_training_example.v1",
                "id": stable_hash(["ground_truth", str(path), normalized_stem, target_path])[:16],
                "track_id": str(payload.get("track_name") or path.parent.name),
                "stem_type": normalized_stem,
                "manifest_path": str(path),
                "source_audio_path": context["source_path"],
                "target_audio_path": target_path,
                "target_audio_sha256": str(stem_payload.get("sha256") or ""),
                "model_id": "verified_ground_truth_bundle",
                "rating": "reference_verified",
                "human_score": None,
                "human_verified": False,
                "reference_verified": True,
                "permission_status": "project_owned" if context["permission_status"] == "user_owned" else context["permission_status"],
                "split": "train",
                "notes": (
                    f"ground_truth_contract={context['contract'].get('mode')}; "
                    f"can_claim_source_mix_1_to_1={source_gate['source_mix_1_to_1_claim_allowed']}"
                ),
                "trainable": True,
                "training_target": "audio_separator_or_stem_adapter",
                "source_gate": source_gate,
                "safety": {
                    "not_release_artifact": True,
                    "requires_eval_before_promotion": True,
                    "source_mix_1_to_1_claim_allowed": source_gate["source_mix_1_to_1_claim_allowed"],
                },
            }
        )
    return examples


def ground_truth_excluded_examples(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    context = ground_truth_context(path, payload)
    if context is None or context["source_gate"]["trainable_ground_truth"] is True:
        return []
    rows: list[dict[str, Any]] = []
    for raw_stem, stem_payload in context["stems"].items():
        if not isinstance(stem_payload, dict):
            continue
        target_path = docker_visible_manifest_path(path, stem_payload.get("path"))
        if not target_path:
            continue
        normalized_stem = "vocal" if raw_stem == "vocals" else ("minus" if raw_stem == "instrumental" else str(raw_stem))
        rows.append(
            {
                "schema": "generic_finetuning_excluded_example.v1",
                "id": stable_hash(["ground_truth_excluded", str(path), normalized_stem, target_path])[:16],
                "track_id": str(payload.get("track_name") or path.parent.name),
                "stem_type": normalized_stem,
                "manifest_path": str(path),
                "exclude_reason": "ground_truth_reconstruction_not_trainable",
                "delivery_status": "",
                "presentation_status": "",
                "human_verified": False,
                "human_rating": "",
                "source_gate": context["source_gate"],
            }
        )
    return rows


def hard_eval_case(row: dict[str, Any], feedback: HumanFeedback | None = None) -> dict[str, Any]:
    failure_labels = list(row["failure_labels"])
    if not failure_labels and feedback and feedback.rating in NEGATIVE_RATINGS:
        failure_labels = [f"human_negative_feedback:{feedback.rating}"]
    return {
        "schema": "generic_finetuning_hard_eval_case.v1",
        "id": stable_hash(["hard_eval", row["manifest_path"]])[:16],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "manifest_path": row["manifest_path"],
        "model_id": row["model_id"],
        "dominant_failure_reason": failure_labels[0] if failure_labels else "",
        "failure_labels": failure_labels,
        "human_verified": bool(feedback and feedback.human_verified),
        "human_rating": feedback.rating if feedback else "",
        "split": feedback.split if feedback else "hard_eval",
        "training_target": "regression_eval",
    }


def excluded_example(row: dict[str, Any], reason: str, feedback: HumanFeedback | None = None) -> dict[str, Any]:
    return {
        "schema": "generic_finetuning_excluded_example.v1",
        "id": stable_hash(["excluded", row["manifest_path"], reason])[:16],
        "track_id": row["track_id"],
        "stem_type": row["stem_type"],
        "manifest_path": row["manifest_path"],
        "exclude_reason": reason,
        "delivery_status": row["delivery_status"],
        "presentation_status": row["presentation_status"],
        "human_verified": bool(feedback and feedback.human_verified),
        "human_rating": feedback.rating if feedback else "",
    }


def build_corpus(
    *,
    artifact_root: Path,
    output_dir: Path,
    human_feedback_jsonls: Sequence[Path] = (),
    auto_discover_human_feedback: bool = True,
    permission_default: str = "requires_review",
    previous_state_json: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered_feedback = discover_human_feedback_jsonls(artifact_root, output_dir=output_dir) if auto_discover_human_feedback else []
    feedback_paths = sorted({path.resolve() for path in [*human_feedback_jsonls, *discovered_feedback]})
    progress = ProgressWriter(output_dir / PROGRESS_NAME)
    progress.mark(
        "start",
        artifact_root=str(artifact_root),
        human_feedback_jsonls=[str(path) for path in feedback_paths],
        auto_discover_human_feedback=auto_discover_human_feedback,
        discovered_human_feedback_jsonls=len(discovered_feedback),
        expected_acceleration_mode="external_only_cpu_metadata_curation",
    )
    feedback_rows = load_human_feedback(feedback_paths)
    progress.mark("feedback_loaded", feedback_count=len(feedback_rows), feedback_files=len(feedback_paths))

    inventory: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    hard_eval_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    ground_truth_manifest_count = 0
    ground_truth_audio_count = 0

    json_paths = sorted(path for path in artifact_root.rglob("*.json") if path.is_file())
    progress.mark("scan_manifests", json_file_count=len(json_paths))
    for index, path in enumerate(json_paths, start=1):
        if index == 1 or index % 500 == 0:
            progress.mark("scan_progress", scanned=index, total=len(json_paths))
        payload = read_json(path)
        if payload is None:
            continue
        ground_truth_examples = ground_truth_audio_training_examples(path, payload)
        ground_truth_exclusions = ground_truth_excluded_examples(path, payload)
        if ground_truth_examples or ground_truth_exclusions:
            ground_truth_manifest_count += 1
            ground_truth_audio_count += len(ground_truth_examples)
            audio_rows.extend(ground_truth_examples)
            excluded_rows.extend(ground_truth_exclusions)
        row = inventory_row(path, artifact_root, payload)
        if not row["track_id"] and not row["delivery_status"] and not row["presentation_status"]:
            continue
        feedback = best_feedback(feedback_rows, path=path, row=row)
        if feedback is not None:
            row["human_feedback"] = {
                "rating": feedback.rating,
                "human_verified": feedback.human_verified,
                "permission_status": feedback.permission_status,
                "split": feedback.split,
                "notes": feedback.notes,
            }
        inventory.append(row)
        if controller_training_allowed(row, feedback):
            controller_rows.append(controller_example(row, feedback))
        if row["blocked_like"] or (feedback and feedback.rating in NEGATIVE_RATINGS):
            hard_eval_rows.append(hard_eval_case(row, feedback))
            continue
        if row["accepted_contract"]:
            positive_feedback = feedback and feedback.human_verified and feedback.rating in POSITIVE_RATINGS
            if positive_feedback and row["selected_audio_exists"]:
                if not (row["accepted_machine_gate_contract"] or row["reference_verified"]):
                    excluded_rows.append(
                        excluded_example(row, "accepted_but_missing_accepted_machine_gate_provenance", feedback)
                    )
                else:
                    candidate = audio_training_example(row, feedback, permission_default=permission_default)
                    if candidate["trainable"]:
                        audio_rows.append(candidate)
                    else:
                        excluded_rows.append(excluded_example(row, "permission_not_trainable", feedback))
            elif row["reference_verified"] and row["selected_audio_exists"]:
                synthetic_feedback = HumanFeedback(
                    track_id=row["track_id"],
                    stem_type=row["stem_type"],
                    rating="accepted",
                    human_verified=False,
                    permission_status=permission_default,
                    split="train",
                    notes="reference_verified_manifest",
                )
                candidate = audio_training_example(row, synthetic_feedback, permission_default=permission_default)
                if candidate["trainable"]:
                    audio_rows.append(candidate)
                else:
                    excluded_rows.append(excluded_example(row, "permission_not_trainable", synthetic_feedback))
            else:
                excluded_rows.append(excluded_example(row, "accepted_but_not_human_or_reference_verified", feedback))

    inventory.sort(key=lambda item: (item["track_id"], item["stem_type"], item["manifest_path"]))
    controller_rows.sort(key=lambda item: item["id"])
    audio_rows.sort(key=lambda item: item["id"])
    hard_eval_rows.sort(key=lambda item: item["id"])
    excluded_rows.sort(key=lambda item: item["id"])
    fingerprint_payload = {
        "audio_training": [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "stem_type": row["stem_type"],
                "target_audio_sha256": row["target_audio_sha256"],
                "permission_status": row["permission_status"],
                "rating": row["rating"],
                "split": row["split"],
            }
            for row in audio_rows
        ],
        "controller": [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "stem_type": row["stem_type"],
                "action": row["target"]["action"],
                "accepted": row["result"]["accepted"],
            }
            for row in controller_rows
        ],
        "hard_eval": [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "stem_type": row["stem_type"],
                "failure_labels": row["failure_labels"],
            }
            for row in hard_eval_rows
        ],
    }
    fingerprint = stable_hash(fingerprint_payload)
    previous_fingerprint = ""
    if previous_state_json and previous_state_json.exists():
        previous = read_json(previous_state_json) or {}
        previous_fingerprint = str(previous.get("fingerprint") or "")
    retrain_reason = "force" if force else ("unchanged" if previous_fingerprint == fingerprint else "fingerprint_changed")
    if not previous_fingerprint and not force:
        retrain_reason = "state_missing"

    summary = {
        "json_files_scanned": len(json_paths),
        "inventory_rows": len(inventory),
        "accepted_like": sum(1 for row in inventory if row["accepted_contract"]),
        "accepted_machine_gate_like": sum(1 for row in inventory if row["accepted_machine_gate_contract"]),
        "blocked_like": sum(1 for row in inventory if row["blocked_like"]),
        "controller_decision_examples": len(controller_rows),
        "audio_training_examples": len(audio_rows),
        "ground_truth_manifests": ground_truth_manifest_count,
        "ground_truth_audio_training_examples": ground_truth_audio_count,
        "hard_eval_cases": len(hard_eval_rows),
        "excluded_examples": len(excluded_rows),
        "human_feedback_rows": len(feedback_rows),
        "human_feedback_files": len(feedback_paths),
        "auto_discovered_human_feedback_files": len(discovered_feedback),
        "retrain_reason": retrain_reason,
    }
    manifest = {
        "schema": "generic_finetuning_corpus_manifest.v1",
        "manifest_type": "generic_finetuning_corpus",
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "previous_fingerprint": previous_fingerprint,
        "retrain_reason": retrain_reason,
        "should_train": retrain_reason != "unchanged" and bool(audio_rows or controller_rows),
        "training_started": False,
        "fine_tune_started": False,
        "delivery_allowed": False,
        "presentation_status": "training_corpus_not_for_listen_pack",
        "release_status": "training_artifact_only",
        "release_ready": False,
        "summary": summary,
        "inputs": {
            "human_feedback_jsonls": [str(path) for path in human_feedback_jsonls],
            "auto_discover_human_feedback": auto_discover_human_feedback,
            "auto_discovered_human_feedback_jsonls": [str(path) for path in discovered_feedback],
            "all_human_feedback_jsonls": [str(path) for path in feedback_paths],
            "permission_default": permission_default,
            "previous_state_json": str(previous_state_json) if previous_state_json else "",
            "force": force,
        },
        "outputs": {
            "inventory_jsonl": str(output_dir / INVENTORY_NAME),
            "controller_decision_examples_jsonl": str(output_dir / CONTROLLER_NAME),
            "audio_training_examples_jsonl": str(output_dir / AUDIO_TRAIN_NAME),
            "hard_eval_cases_jsonl": str(output_dir / HARD_EVAL_NAME),
            "excluded_examples_jsonl": str(output_dir / EXCLUDED_NAME),
            "progress_jsonl": str(output_dir / PROGRESS_NAME),
        },
        "recommended_training_jobs": {
            "controller": {
                "enabled": bool(controller_rows),
                "target": "LLM/controller route selector",
                "reason": "Playbook allows training the decision controller on accepted and rejected route decisions.",
            },
            "audio_vocal": {
                "enabled": any(row["stem_type"] == "vocal" for row in audio_rows),
                "target": "vocal separator/adapter",
                "requires": "human_verified_or_reference_verified target audio with cleared permission",
            },
            "audio_stems": {
                "enabled": any(row["stem_type"] != "vocal" for row in audio_rows),
                "target": "stem/minus separator or segment preference adapter",
                "requires": "human_verified_or_reference_verified target audio with cleared permission",
            },
        },
        "safety": dict(SAFETY_FLAGS),
    }
    progress.mark("write_outputs", **summary)
    write_jsonl(output_dir / INVENTORY_NAME, inventory)
    write_jsonl(output_dir / CONTROLLER_NAME, controller_rows)
    write_jsonl(output_dir / AUDIO_TRAIN_NAME, audio_rows)
    write_jsonl(output_dir / HARD_EVAL_NAME, hard_eval_rows)
    write_jsonl(output_dir / EXCLUDED_NAME, excluded_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress.mark("finish", manifest_path=str(output_dir / MANIFEST_NAME), fingerprint=fingerprint)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a generic fine-tuning corpus from lab_artifacts using the project playbook contract."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-feedback-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--no-auto-discover-human-feedback", action="store_true")
    parser.add_argument("--permission-default", default="requires_review")
    parser.add_argument("--previous-state-json", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_generic_finetuning_corpus_from_lab_artifacts.py",
        entrypoint_path="/app/tools/build_generic_finetuning_corpus_from_lab_artifacts.py",
    )
    args = parse_args(argv)
    manifest = build_corpus(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        human_feedback_jsonls=args.human_feedback_jsonl,
        auto_discover_human_feedback=not bool(args.no_auto_discover_human_feedback),
        permission_default=args.permission_default,
        previous_state_json=args.previous_state_json,
        force=bool(args.force),
    )
    print(json.dumps({"summary": manifest["summary"], "fingerprint": manifest["fingerprint"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
