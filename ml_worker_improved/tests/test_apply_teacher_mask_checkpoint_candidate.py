from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import tools.apply_teacher_mask_checkpoint_candidate as candidate_tool


def test_teacher_mask_checkpoint_candidate_is_route_only(tmp_path: Path, monkeypatch) -> None:
    sample_rate = 8_000
    source = tmp_path / "source.flac"
    checkpoint = tmp_path / "model.pt"
    output_dir = tmp_path / "route" / "stems"
    audio = np.zeros((sample_rate // 10, 2), dtype=np.float32)
    sf.write(source, audio, sample_rate)
    checkpoint.write_bytes(b"checkpoint")

    def fake_apply(**kwargs):
        apply_out = Path(kwargs["output_dir"])
        apply_out.mkdir(parents=True, exist_ok=True)
        stem_path = apply_out / "01_teacher_mask_checkpoint_minus.flac"
        sf.write(stem_path, audio + 0.1, sample_rate)
        return {
            "stems": {"minus": {"path": str(stem_path)}},
            "contract": {"application_reference_free": True},
        }

    monkeypatch.setattr(candidate_tool, "apply_teacher_mask_checkpoint_to_source", fake_apply)

    manifest = candidate_tool.apply_teacher_mask_checkpoint_candidate(
        source_mix_path=source,
        stems_dir=output_dir,
        checkpoint_path=checkpoint,
        stem="minus",
        device="cuda",
        command_args=[],
    )

    assert (output_dir / "minus.flac").is_file()
    assert (output_dir / "tm_candidate.json").is_file()
    assert manifest["contract"]["candidate_route_only"] is True
    assert manifest["contract"]["delivery_allowed_without_runner_gates"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "candidate_route_not_listen_pack"
