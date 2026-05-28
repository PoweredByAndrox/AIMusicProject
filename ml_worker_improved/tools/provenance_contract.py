from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path("/app")

UNSAFE_TRUE_FLAGS = (
    "target_reference_assisted",
    "target_reference_used",
    "target_reference_used_at_application",
    "reference_teacher_pack",
    "fallback_used",
    "hidden_inference",
    "per_track_hardcode_allowed",
)
UNSAFE_FALSE_FLAGS = (
    "application_reference_free",
    "not_timestamp_hardcoded",
)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def resolve_manifest_path(path_value: str, *, current_dir: Path | None) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        try:
            relative_to_app = candidate.relative_to(APP_ROOT)
        except ValueError:
            return candidate
        return PROJECT_ROOT / relative_to_app
    if current_dir is None:
        return candidate
    return current_dir / candidate


def assert_no_reference_contract(
    manifest: dict[str, Any],
    *,
    role: str,
    required_contract: dict[str, bool] | None = None,
) -> None:
    contract = manifest.get("contract")
    candidate_contract = manifest.get("candidate_contract")
    contracts: list[tuple[str, dict[str, Any]]] = []
    if isinstance(contract, dict):
        contracts.append(("contract", contract))
    if isinstance(candidate_contract, dict):
        contracts.append(("candidate_contract", candidate_contract))
    if required_contract is not None and not isinstance(contract, dict):
        raise ValueError(f"{role} manifest has no contract")

    for contract_name, contract_payload in contracts:
        if bool(contract_payload.get("target_reference_assisted")):
            raise ValueError(f"{role} manifest is target-reference-assisted")
        if bool(contract_payload.get("target_reference_used_at_application")) or bool(contract_payload.get("target_reference_used")):
            raise ValueError(f"{role} manifest used a target reference at application time")
        for flag in UNSAFE_TRUE_FLAGS:
            if bool(contract_payload.get(flag)):
                raise ValueError(f"{role} {contract_name} has unsafe reference/provenance flag: {flag}=true")
        for flag in UNSAFE_FALSE_FLAGS:
            if contract_payload.get(flag) is False:
                raise ValueError(f"{role} {contract_name} has unsafe reference/provenance flag: {flag}=false")
        if contract_payload.get("delivery_allowed_without_reference") is False:
            raise ValueError(
                f"{role} {contract_name} is not deliverable without reference"
            )

    if required_contract is None:
        return
    for key, expected_value in required_contract.items():
        if key not in contract:
            raise ValueError(f"{role} contract is missing required key: {key}")
        if contract[key] is not expected_value:
            raise ValueError(f"{role} contract has unsafe value: {key}={contract[key]!r}")


def collect_declared_parent_manifests(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> list[Path]:
    current_dir = manifest_path.parent if manifest_path is not None else None
    current_resolved = manifest_path.resolve() if manifest_path is not None else None
    found: list[Path] = []
    seen_values: set[str] = set()

    def visit(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered_key = str(key).lower()
                if isinstance(value, str) and "manifest" in lowered_key and value.endswith(".json"):
                    if lowered_key == "manifest_path":
                        continue
                    if value in seen_values:
                        continue
                    seen_values.add(value)
                    resolved = resolve_manifest_path(value, current_dir=current_dir).resolve()
                    if current_resolved is not None and resolved == current_resolved:
                        continue
                    found.append(resolved)
                elif isinstance(value, dict) and "manifest" in lowered_key:
                    path_value = value.get("path")
                    if isinstance(path_value, str) and path_value.endswith(".json") and path_value not in seen_values:
                        seen_values.add(path_value)
                        resolved = resolve_manifest_path(path_value, current_dir=current_dir).resolve()
                        if current_resolved is None or resolved != current_resolved:
                            found.append(resolved)
                    visit(value, lowered_key)
                else:
                    visit(value, lowered_key)
        elif isinstance(node, list):
            for value in node:
                visit(value, parent_key)

    visit(manifest)
    return found


def assert_no_reference_lineage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    role: str,
    required_contract: dict[str, bool] | None = None,
    max_depth: int = 12,
) -> dict[str, Any]:
    checked_paths: list[str] = []
    visiting: set[Path] = set()

    def visit(current_manifest: dict[str, Any], current_path: Path | None, current_role: str, depth: int) -> None:
        if depth > max_depth:
            raise ValueError(f"{role} manifest lineage exceeds max depth: {max_depth}")
        assert_no_reference_contract(
            current_manifest,
            role=current_role,
            required_contract=required_contract if depth == 0 else None,
        )
        for parent_path in collect_declared_parent_manifests(current_manifest, manifest_path=current_path):
            if parent_path in visiting:
                raise ValueError(f"{role} manifest lineage contains a cycle: {parent_path}")
            if not parent_path.exists():
                raise FileNotFoundError(f"{role} ancestor manifest is not visible for provenance verification: {parent_path}")
            visiting.add(parent_path)
            checked_paths.append(str(parent_path))
            parent_manifest = load_manifest(parent_path)
            visit(parent_manifest, parent_path, f"{current_role} ancestor", depth + 1)
            visiting.remove(parent_path)

    if manifest_path is not None:
        checked_paths.append(str(manifest_path.resolve()))
    visit(manifest, manifest_path.resolve() if manifest_path is not None else None, role, 0)
    return {
        "no_reference_lineage_verified": True,
        "checked_manifest_paths": checked_paths,
    }
