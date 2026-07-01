from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools import validate_mac_mel_refiner_chain as chain


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")
    return path


def _guard(root: Path) -> Path:
    audio = root / "audio"
    audio.mkdir(parents=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (audio / name).write_bytes(b"audio")
    _write_json(
        root / "manifest.json",
        {
            "audio": {
                "vocals": str(audio / "v.flac"),
                "instrumental": str(audio / "i.flac"),
                "raw_ref": str(audio / "raw.flac"),
            },
            "score": {"config": {"window_sec": 0.75}},
            "delivery_allowed": False,
            "listen_pack_created": False,
        },
    )
    return root


def _replay(status: str, verdict: str, output_dir: Path) -> dict[str, object]:
    audio = output_dir / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    for name in ("v.flac", "i.flac", "raw.flac"):
        (audio / name).write_bytes(b"audio")
    return {
        "status": status,
        "resolved_device": "mps",
        "after_score": {
            "verdict": verdict,
            "regions": [] if verdict == "pass" else [{"start": 1.0, "end": 2.0}],
            "config": {"window_sec": 0.75},
        },
        "metrics": {"before_region_count": 2, "after_region_count": 0 if verdict == "pass" else 1},
        "audio": {
            "vocals": str(audio / "v.flac"),
            "instrumental": str(audio / "i.flac"),
            "raw_ref": str(audio / "raw.flac"),
        },
        "delivery_allowed": False,
        "listen_pack_created": False,
    }


def test_chain_validation_passes_without_delivery_when_no_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        guard = _guard(root / "guard")
        first = _checkpoint(root / "ckpt" / "g1.5" / "model.pt")
        residual = _checkpoint(root / "ckpt" / "resid" / "model.pt")
        calls: list[Path] = []

        def fake_replay(**kwargs):
            out = Path(kwargs["output_dir"])
            calls.append(Path(kwargs["checkpoint_path"]))
            return _replay("repaired_passed_pitch_guard", "pass", out)

        original = chain.run_refiner_replay
        try:
            chain.run_refiner_replay = fake_replay
            manifest = chain.run_chain_validation(
                mel_guard_dir=guard,
                first_checkpoint=first,
                residual_checkpoint=residual,
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            chain.run_refiner_replay = original

        assert calls == [first, residual]
        assert manifest["status"] == "validation_passed"
        assert manifest["delivery_allowed"] is False
        assert manifest["listen_pack_created"] is False


def test_chain_validation_blocks_training_overlap_even_when_audio_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        guard = _guard(root / "guard")
        failed_stage = root / "failed"
        _write_json(failed_stage / "manifest.json", {"source_guard_dir": str(guard)})
        residual_manifest = root / "residual.json"
        _write_json(residual_manifest, {"source_failed_stage_dir": str(failed_stage), "delivery_allowed": False})
        first = _checkpoint(root / "ckpt" / "g1.5" / "model.pt")
        residual = _checkpoint(root / "ckpt" / "resid" / "model.pt")

        original = chain.run_refiner_replay
        try:
            chain.run_refiner_replay = lambda **kwargs: _replay("repaired_passed_pitch_guard", "pass", Path(kwargs["output_dir"]))
            manifest = chain.run_chain_validation(
                mel_guard_dir=guard,
                first_checkpoint=first,
                residual_checkpoint=residual,
                residual_training_manifest_path=residual_manifest,
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            chain.run_refiner_replay = original

        assert manifest["status"] == "validation_blocked_training_overlap"
        assert manifest["failed_checks"] == ["training_overlap"]
        assert manifest["delivery_allowed"] is False


def test_chain_validation_blocks_if_final_pitch_guard_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        guard = _guard(root / "guard")
        first = _checkpoint(root / "ckpt" / "g1.5" / "model.pt")
        residual = _checkpoint(root / "ckpt" / "resid" / "model.pt")

        def fake_replay(**kwargs):
            out = Path(kwargs["output_dir"])
            if out.name == "s1":
                return _replay("repaired_but_still_blocked", "fail", out)
            return _replay("repaired_but_still_blocked", "fail", out)

        original = chain.run_refiner_replay
        try:
            chain.run_refiner_replay = fake_replay
            manifest = chain.run_chain_validation(
                mel_guard_dir=guard,
                first_checkpoint=first,
                residual_checkpoint=residual,
                output_dir=root / "out",
                require_mps=True,
            )
        finally:
            chain.run_refiner_replay = original

        assert manifest["status"] == "validation_blocked_pitch_guard"
        assert manifest["failed_checks"] == ["pitch_guard_failed"]
        assert manifest["delivery_allowed"] is False
