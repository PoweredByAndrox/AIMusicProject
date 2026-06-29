from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .stem_route_policy_model import save_stem_route_policy_checkpoint, train_stem_route_policy_from_rows
except ImportError:  # pragma: no cover - direct script execution
    from stem_route_policy_model import save_stem_route_policy_checkpoint, train_stem_route_policy_from_rows  # type: ignore


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return path


def train_stem_route_policy_model(*, rows_jsonl_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(rows_jsonl_path)
    usable_rows = [row for row in rows if bool(row.get("usable_for_training", True))]
    model = train_stem_route_policy_from_rows(usable_rows)
    model_path = save_stem_route_policy_checkpoint(model, output_dir / "model.pt")
    action_counts = Counter(str(row.get("oracle_action") or "") for row in usable_rows)
    feature_schema = {
        "schema": "stem_route_policy_feature_schema.v1",
        "categorical": ["issue_type", "root_cause", "affected_stem", "route_signature", "lane_routes"],
        "numeric_groups": ["gate_verdicts", "forensic_features", "damage_features", "reconstruction_features"],
    }
    history = {
        "schema": "stem_route_policy_training_history.v1",
        "epochs": 1,
        "model_class": model["model_class"],
        "usable_training_row_count": len(usable_rows),
        "action_counts": dict(sorted(action_counts.items())),
    }
    manifest = {
        "schema": "stem_route_policy_model_manifest.v1",
        "model_path": str(model_path),
        "manifest_path": str(output_dir / "manifest.json"),
        "history_json": str(output_dir / "history.json"),
        "feature_schema_json": str(output_dir / "feature_schema.json"),
        "source_rows_jsonl": str(rows_jsonl_path),
        "model_class": model["model_class"],
        "shadow_only": True,
        "promoted": False,
        "delivery_allowed": False,
        "usable_training_row_count": len(usable_rows),
        "usable_for_promotion_row_count": 0,
        "action_counts": dict(sorted(action_counts.items())),
    }
    _write_json(output_dir / "feature_schema.json", feature_schema)
    _write_json(output_dir / "history.json", history)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = train_stem_route_policy_model(rows_jsonl_path=args.rows_jsonl, output_dir=args.out)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
