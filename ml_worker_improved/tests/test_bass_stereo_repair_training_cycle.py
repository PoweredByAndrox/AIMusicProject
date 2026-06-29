from __future__ import annotations

import json
from pathlib import Path

from tools import run_bass_stereo_repair_training_cycle as cycle


def test_bass_stereo_repair_cycle_writes_training_only_contract(monkeypatch, tmp_path: Path) -> None:
    def fake_train_stem_region_repair_model(*, rows_jsonl: Path, output_dir: Path, config):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "model.pt"
        checkpoint.write_bytes(b"fixture")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "rows_jsonl": str(rows_jsonl),
                    "shadow_only": True,
                    "delivery_allowed": False,
                }
            ),
            encoding="utf-8",
        )
        return {"checkpoint": str(checkpoint)}

    def fake_validate_stem_repair_checkpoint(*, checkpoint_path: Path, benchmark_rows_jsonl: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint": str(checkpoint_path),
            "benchmark_rows_jsonl": str(benchmark_rows_jsonl),
            "evaluated_count": 1,
            "improved_count": 1,
            "delivery_allowed": False,
            "promotion_allowed": False,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(cycle, "train_stem_region_repair_model", fake_train_stem_region_repair_model)
    monkeypatch.setattr(cycle, "validate_stem_repair_checkpoint", fake_validate_stem_repair_checkpoint)

    manifest = cycle.run_cycle(
        output_dir=tmp_path / "cycle",
        epochs=1,
        device="cpu",
        hidden_channels=8,
        learning_rate=0.001,
        root_cause_row_id="rc_test",
    )

    assert manifest["contract"]["synthetic_ground_truth_only"] is True
    assert manifest["contract"]["generated_separator_ground_truth_allowed"] is False
    assert manifest["contract"]["delivery_allowed"] is False
    assert manifest["validation"]["improved_count"] == 1
    row = json.loads((tmp_path / "cycle" / "ds" / "rows.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["is_ground_truth"] is True
    assert row["label_type"] == "synthetic_clean"
    assert row["root_cause_row_id"] == "rc_test"
