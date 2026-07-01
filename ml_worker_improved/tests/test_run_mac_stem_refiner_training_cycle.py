from __future__ import annotations

import torch
from pathlib import Path

from tools import run_mac_stem_refiner_training_cycle as cycle


def test_mac_stem_refiner_cycle_blocks_when_mps_is_required_but_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cycle, "resolve_region_repair_training_device", lambda requested: torch.device("cpu"))

    manifest = cycle.run_training_cycle(output_dir=tmp_path / "run", target_stem="drums", epochs=1, require_mps=True)

    assert manifest["status"] == "blocked_mps_unavailable"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "training_diagnostic_only"
    assert manifest["resolved_device"] == "cpu"
    assert (tmp_path / "run" / "manifest.json").exists()
    assert "blocked_mps_unavailable" in (tmp_path / "run" / "progress.jsonl").read_text(encoding="utf-8")


def test_mac_stem_refiner_cycle_keeps_delivery_disabled_for_trained_artifact(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        pass

    def fake_train_region_repair_model_from_arrays(**kwargs):
        return FakeModel(), {
            "resolved_device": "mps",
            "final_loss": 0.001,
            "config": {"device": "mps"},
            "loss_history": [0.01, 0.001],
        }

    def fake_apply_region_repair_model(**kwargs):
        target = kwargs["bad_stem_audio"] * 0.0
        return target, {"changed_outside_ratio": 0.0}

    monkeypatch.setattr(cycle, "resolve_region_repair_training_device", lambda requested: torch.device("mps"))
    monkeypatch.setattr(cycle, "train_region_repair_model_from_arrays", fake_train_region_repair_model_from_arrays)
    monkeypatch.setattr(cycle, "apply_region_repair_model", fake_apply_region_repair_model)
    monkeypatch.setattr(cycle, "save_region_repair_checkpoint", lambda model, path, training_history: path.write_bytes(b"pt") or path)
    monkeypatch.setattr(cycle, "write_training_history", lambda path, history: cycle.write_json(path, history))

    manifest = cycle.run_training_cycle(output_dir=tmp_path / "run", target_stem="drums", epochs=1, require_mps=True)

    assert manifest["contract"]["weights_updated"] is True
    assert manifest["contract"]["diagnostic_only"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["presentation_status"] == "training_diagnostic_only"
