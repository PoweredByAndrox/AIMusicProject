from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LISTENABLE_PRESENTATION_STATUSES = {"accepted_for_listen_pack"}
DELIVERABLE_RELEASE_STATUSES = {"release_ready"}
ACCEPTED_DELIVERY_STATUSES = {"accepted"}
SEMANTIC_JUDGE_INPUT_DELIVERY_STATUS = "pending_semantic_judge"
SEMANTIC_JUDGE_INPUT_PRESENTATION_STATUS = "semantic_judge_input_not_for_listen_pack"
SEMANTIC_JUDGE_INPUT_RELEASE_STATUS = "semantic_judge_input_only"
BLOCKED_STATUSES = {
    "blocked",
    "blocked_for_listening",
    "blocked_from_delivery",
    "blocked_from_listen_pack",
    "candidate_space_exhausted",
    "diagnostic_only",
    "no_accepted_candidate",
    "rejected",
    "skipped",
}
BLOCKING_FAILURE_LABEL_MARKERS = (
    "blocked",
    "rejected",
    "failed",
    "skip",
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
    "candidate_delivery_blocked",
    "no_clean_candidate",
    "artifact_or_timbre_damage",
)
FAILURE_LABEL_KEYS = {
    "failure_labels",
    "issue_labels",
    "failed_checks",
    "blocked_checks",
    "blocking_issue_labels",
    "blockers",
    "blocking_reasons",
    "check_names",
    "forbidden_actions",
}
FAILURE_ITEM_KEYS = ("code", "label", "name", "id", "check", "reason")
BOOLEAN_BLOCKER_KEYS = {"blocked", "is_blocked", "has_blocker", "has_blockers"}


class ArtifactContractError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactStatusView:
    delivery_status: str
    delivery_allowed: bool
    presentation_status: str
    release_status: str
    source_manifest: str
    manifest_sha256: str
    dominant_failure_reason: str
    stem_type: str
    model_id: str | None = None
    candidate_id: str | None = None
    audio_path: str | None = None
    blocking_failure_labels: tuple[str, ...] = ()

    @property
    def listenable(self) -> bool:
        return (
            self.delivery_allowed
            and not self.blocking_failure_labels
            and self.presentation_status in LISTENABLE_PRESENTATION_STATUSES
        )

    @property
    def deliverable(self) -> bool:
        return (
            self.delivery_allowed
            and not self.blocking_failure_labels
            and self.release_status in DELIVERABLE_RELEASE_STATUSES
        )

    @property
    def blocked(self) -> bool:
        values = {self.delivery_status, self.presentation_status, self.release_status}
        return not self.delivery_allowed or bool(values & BLOCKED_STATUSES) or bool(self.blocking_failure_labels)

    def assert_promotable(self) -> None:
        if not self.source_manifest or not self.manifest_sha256:
            raise ArtifactContractError("artifact_without_manifest_provenance_cannot_be_promoted")
        if self.blocking_failure_labels:
            raise ArtifactContractError("artifact_with_blocking_failure_labels_cannot_be_result")
        if self.blocked:
            raise ArtifactContractError("diagnostic_or_blocked_artifact_cannot_be_result")
        if not (self.listenable or self.deliverable):
            raise ArtifactContractError("artifact_without_listenable_or_release_ready_status_cannot_be_result")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_track_hash(*parts: Any) -> str:
    normalized = "|".join(str(part or "").strip().lower().replace("\\", "/") for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def stable_track_split(track_hash: str) -> str:
    bucket = int(hashlib.sha256(track_hash.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "holdout"


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_string(payload: dict[str, Any], paths: Iterable[tuple[str, ...]], default: str = "") -> str:
    for path in paths:
        value = _nested_get(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _first_bool(payload: dict[str, Any], paths: Iterable[tuple[str, ...]], default: bool = False) -> bool:
    for path in paths:
        value = _nested_get(payload, path)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
    return default


def infer_stem_type(payload: dict[str, Any], manifest_path: Path | None = None) -> str:
    explicit_values = (
        payload.get("stem_type"),
        payload.get("target_stem"),
        payload.get("stem"),
        payload.get("selected_stem"),
        payload.get("mode"),
    )
    for value in explicit_values:
        normalized = str(value or "").strip().lower()
        if normalized in {"both", "pair", "vocal_minus", "vocal+minus", "vocal_minus_pair"}:
            return "both"
        if normalized in {"minus", "instrumental"}:
            return "minus"
        if normalized in {"vocal", "vocals"}:
            return "vocal"
        if normalized in {"melody", "drums", "bass", "other"}:
            return normalized
        if normalized in {"full_stem_pack", "full-stem-pack", "stem_pack", "stem-pack"}:
            return "full_stem_pack"
    text = " ".join(
        str(value).lower()
        for value in (
            manifest_path.name if manifest_path else "",
            str(manifest_path.parent) if manifest_path else "",
        )
        if value
    )
    if "minus" in text or "instrumental" in text:
        return "minus"
    if "vocal" in text:
        return "vocal"
    return "unknown"


def add_failure_label(labels: set[str], value: Any) -> None:
    for part in str(value or "").split(","):
        clean = part.strip()
        if clean:
            labels.add(clean)


def add_structured_failure_item(labels: set[str], value: Any) -> None:
    if isinstance(value, dict):
        for key in FAILURE_ITEM_KEYS:
            child = value.get(key)
            if isinstance(child, str) and child.strip():
                add_failure_label(labels, child)
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in BOOLEAN_BLOCKER_KEYS and child is True:
                add_failure_label(labels, key)
            elif isinstance(child, (dict, list)):
                add_structured_failure_item(labels, child)
    elif isinstance(value, list):
        for item in value:
            add_structured_failure_item(labels, item)
    else:
        add_failure_label(labels, value)


def collect_failure_labels(payload: Any) -> set[str]:
    labels: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in FAILURE_LABEL_KEYS:
                if isinstance(value, list):
                    for item in value:
                        add_structured_failure_item(labels, item)
                elif isinstance(value, str) and value.strip():
                    add_failure_label(labels, value)
                elif isinstance(value, dict):
                    add_structured_failure_item(labels, value)
            elif key_l in BOOLEAN_BLOCKER_KEYS and value is True:
                add_failure_label(labels, key_l)
            elif key_l in {"status", "delivery_status", "presentation_status", "release_status"}:
                normalized_status = str(value).strip().lower()
                if normalized_status in BLOCKED_STATUSES:
                    add_failure_label(labels, normalized_status)
            labels.update(collect_failure_labels(value))
    elif isinstance(payload, list):
        for item in payload:
            labels.update(collect_failure_labels(item))
    return labels


def blocking_failure_labels(payload: dict[str, Any]) -> tuple[str, ...]:
    labels = collect_failure_labels(payload)
    return blocking_labels_from_label_set(labels)


def blocking_labels_from_label_set(labels: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            label
            for label in labels
            if any(marker in label.lower() for marker in BLOCKING_FAILURE_LABEL_MARKERS)
        )
    )


def blocking_labels_from_values(*values: Any) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()

    def add_if_blocking(value: Any) -> None:
        for part in str(value or "").split(","):
            clean = part.strip()
            if (
                clean
                and clean not in seen
                and any(marker in clean.lower() for marker in BLOCKING_FAILURE_LABEL_MARKERS)
            ):
                seen.add(clean)
                labels.append(clean)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in FAILURE_ITEM_KEYS:
                child = value.get(key)
                if isinstance(child, str) and child.strip():
                    add_if_blocking(child)
            for key, child in value.items():
                if isinstance(key, str) and key.lower() in BOOLEAN_BLOCKER_KEYS and child is True:
                    add_if_blocking(key)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        else:
            add_if_blocking(value)

    for value in values:
        visit(value)
    return tuple(labels)


def infer_dominant_failure_reason(payload: dict[str, Any]) -> str:
    for path in (
        ("dominant_failure_reason",),
        ("dominant_failure_class",),
        ("decision_contract", "dominant_failure_class"),
        ("selected_minus_release_gate", "blockers", 0, "reason"),
        ("defect_effectiveness_gate", "blockers", 0, "reason"),
        ("next_action", "status"),
        ("overall", "status"),
        ("delivery_status",),
        ("status",),
        ("reason",),
    ):
        current: Any = payload
        for part in path:
            if isinstance(part, int):
                if not isinstance(current, list) or len(current) <= part:
                    current = None
                    break
                current = current[part]
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                current = None
                break
        if isinstance(current, str) and current.strip():
            return current.strip()
    labels = collect_failure_labels(payload)
    if labels:
        return sorted(labels)[0]
    return "unknown_blocked_reason"


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"manifest_must_be_object:{path}")
    return payload


def normalize_manifest_status(payload: dict[str, Any], manifest_path: Path) -> ArtifactStatusView:
    delivery_status = _first_string(
        payload,
        (
            ("delivery_status",),
            ("selected_artifacts", "delivery_status"),
            ("candidate_status",),
            ("overall", "status"),
            ("status",),
        ),
        default="unknown",
    )
    presentation_status = _first_string(
        payload,
        (
            ("presentation_status",),
            ("selected_artifacts", "presentation_status"),
            ("result_presentation_status",),
            ("presentation", "status"),
        ),
        default=delivery_status,
    )
    release_status = _first_string(
        payload,
        (
            ("release_status",),
            ("selected_artifacts", "release_status"),
            ("selected_minus_release_gate", "release_status"),
            ("selected_minus_release_gate", "status"),
            ("defect_effectiveness_gate", "release_status"),
            ("defect_effectiveness_gate", "status"),
            ("release", "status"),
        ),
        default=presentation_status,
    )
    delivery_allowed = _first_bool(
        payload,
        (
            ("delivery_allowed",),
            ("selected_artifacts", "delivery_allowed"),
            ("result_delivery_allowed",),
            ("release_allowed",),
        ),
        default=delivery_status in ACCEPTED_DELIVERY_STATUSES
        and (
            presentation_status in LISTENABLE_PRESENTATION_STATUSES
            or release_status in DELIVERABLE_RELEASE_STATUSES
        ),
    )
    blocking_labels = blocking_failure_labels(payload)
    dominant = infer_dominant_failure_reason(payload)
    if blocking_labels and dominant in {
        "",
        "accepted",
        "accepted_for_listen_pack",
        "none",
        "release_ready",
        "unknown",
    }:
        dominant = blocking_labels[0]
    if not delivery_allowed and dominant in {
        "",
        "accepted",
        "accepted_for_listen_pack",
        "none",
        "release_ready",
        "unknown",
    }:
        dominant = "unknown_blocked_reason"

    model_id = _first_string(
        payload,
        (
            ("model_id",),
            ("registered_model_id",),
            ("source_model_id",),
            ("model", "model_id"),
            ("model", "model_name"),
            ("model", "name"),
        ),
    ) or None
    candidate_id = _first_string(
        payload,
        (
            ("candidate_id",),
            ("selected_candidate", "candidate_id"),
            ("selected_candidate", "label"),
            ("selected_candidate",),
            ("id",),
        ),
    ) or None
    audio_path = _first_string(
        payload,
        (
            ("audio_path",),
            ("result_audio_path",),
            ("selected_audio_path",),
            ("flac_path",),
            ("selected_artifacts", "deliverable_vocals_flac"),
            ("selected_artifacts", "deliverable_listen_vocals_flac"),
            ("selected_artifacts", "listen_selected_vocals_flac"),
            ("selected_artifacts", "selected_minus_flac"),
            ("selected_artifacts", "listen_minus_flac"),
            ("selected_artifacts", "listen_stem_flac"),
        ),
    ) or None

    return ArtifactStatusView(
        delivery_status=delivery_status,
        delivery_allowed=delivery_allowed,
        presentation_status=presentation_status,
        release_status=release_status,
        source_manifest=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        dominant_failure_reason=dominant or "unknown_blocked_reason",
        stem_type=infer_stem_type(payload, manifest_path),
        model_id=model_id,
        candidate_id=candidate_id,
        audio_path=audio_path,
        blocking_failure_labels=blocking_labels,
    )


def status_view_from_manifest(path: Path) -> ArtifactStatusView:
    return normalize_manifest_status(read_manifest(path), path)


def can_promote_to_listen_result(path: Path) -> bool:
    status_view_from_manifest(path).assert_promotable()
    return True
