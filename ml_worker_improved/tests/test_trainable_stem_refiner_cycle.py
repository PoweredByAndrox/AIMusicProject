from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.run_trainable_stem_refiner_cycle import run_trainable_stem_refiner_cycle
from tools.stem_region_repair_common import ISSUE_VOCAL_LEAK
from tools.stem_region_repair_model import load_region_repair_checkpoint


def test_trainable_stem_refiner_cycle_changes_weights_and_improves_audio(tmp_path: Path) -> None:
    manifest = run_trainable_stem_refiner_cycle(
        output_dir=tmp_path / "cycle",
        stem="drums",
        issue_type=ISSUE_VOCAL_LEAK,
        epochs=80,
        hidden_channels=16,
        learning_rate=0.01,
        device="cpu",
        sample_rate=8_000,
        seconds=0.5,
    )

    before_after = json.loads(Path(manifest["before_after_json"]).read_text(encoding="utf-8"))
    model, checkpoint = load_region_repair_checkpoint(Path(manifest["checkpoint"]))
    weight_abs_sum = sum(float(torch.sum(torch.abs(value)).item()) for value in model.state_dict().values())

    assert manifest["weights_changed"] is True
    assert manifest["trainable_parameter_count"] > 0
    assert weight_abs_sum > 0.0
    assert before_after["target_region_mse_after"] < before_after["target_region_mse_before"] * 0.40
    assert before_after["outside_region_mse_delta"] < 1.0e-8
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False


def test_trainable_stem_refiner_cycle_never_marks_generated_audio_as_delivery(tmp_path: Path) -> None:
    manifest = run_trainable_stem_refiner_cycle(
        output_dir=tmp_path / "cycle",
        stem="minus",
        issue_type=ISSUE_VOCAL_LEAK,
        epochs=40,
        hidden_channels=12,
        learning_rate=0.01,
        device="cpu",
        sample_rate=8_000,
        seconds=0.35,
    )

    assert manifest["artifact_rag_write_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["contract"]["generated_audio_delivery_allowed"] is False
