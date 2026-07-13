from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "audit_minus_quality_ensemble.py"


def _load_module():
    tools_root = str(TOOLS_ROOT)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("audit_minus_quality_ensemble", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([audio, audio], axis=1), sample_rate, subtype="PCM_24")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> np.ndarray:
    time = np.arange(int(round(sample_rate * seconds)), dtype=np.float32) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _resized_tone(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> np.ndarray:
    return _tone(sample_rate, seconds, frequency, amplitude)


def _write_proxy_manifest(
    root: Path,
    *,
    model_id: str,
    source_path: Path,
    vocals: np.ndarray,
    sample_rate: int,
) -> Path:
    vocals_path = _write_audio(root / "vocals.flac", vocals, sample_rate)
    manifest = {
        "schema_version": 1,
        "source_audio_path": str(source_path),
        "model_id": model_id,
        "target_stem": "vocals",
        "stems": {
            "vocals": {
                "path": str(vocals_path),
                "role": "target",
                "sha256": _sha256(vocals_path),
            }
        },
    }
    manifest_path = root / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_multistem_proxy_manifest(
    root: Path,
    *,
    model_id: str,
    source_path: Path,
    vocals: np.ndarray,
    sample_rate: int,
) -> Path:
    vocals_path = _write_audio(root / "vocals.flac", vocals, sample_rate)
    source_reference_path = root / f"source_ref{source_path.suffix}"
    source_reference_path.write_bytes(source_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "source_audio_path": "/unavailable/mac/mount/source.flac",
        "model_id": model_id,
        "model": {"target_output": "multi_stem"},
        "source_reference": {
            "path": "/unavailable/mac/mount/source_ref.flac",
            "sha256": _sha256(source_reference_path),
            "exact_byte_copy": True,
        },
        "stems": {
            "vocals": {
                "path": str(vocals_path),
                "role": "target",
                "sha256": _sha256(vocals_path),
            }
        },
    }
    manifest_path = root / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


PITCH_CONFIG = {
    "probe_semitones": [0.0],
    "window_sec": 0.25,
    "hop_sec": 0.25,
    "score_threshold": 10.0,
    "projection_rel_db_threshold": 10.0,
    "strong_projection_rel_db_threshold": 10.0,
    "strong_projection_stft_cosine_threshold": 2.0,
}


class TestAuditMinusQualityEnsemble(unittest.TestCase):
    def test_mac_manifest_paths_map_to_runtime_root_before_reencoded_raw_ref(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_root = root / "runtime"
            producer_root = "/Users/test/AIMusicProject_win"
            sample_rate = 8000
            source_path = _write_audio(
                runtime_root / "artifacts" / "source" / "input.flac",
                _tone(sample_rate, 1.0, 220.0, 0.2),
                sample_rate,
            )
            proxy_root = runtime_root / "artifacts" / "proxy" / "run"
            vocals_path = _write_audio(
                proxy_root / "vocals.flac",
                _tone(sample_rate, 1.0, 660.0, 0.05),
                sample_rate,
            )
            raw_ref_path = _write_audio(
                proxy_root / "raw_ref.flac",
                _tone(sample_rate, 1.0, 330.0, 0.1),
                sample_rate,
            )
            self.assertNotEqual(_sha256(raw_ref_path), _sha256(source_path))
            manifest = {
                "schema_version": 1,
                "source_audio_path": f"{producer_root}/artifacts/source/input.flac",
                "model_id": "mac_vocal_proxy",
                "target_stem": "vocals",
                "stems": {
                    "vocals": {
                        "path": f"{producer_root}/artifacts/proxy/run/vocals.flac",
                        "role": "target",
                        "sha256": _sha256(vocals_path),
                    }
                },
                "artifacts": {
                    "raw_ref_flac": f"{producer_root}/artifacts/proxy/run/raw_ref.flac",
                },
            }
            manifest_path = proxy_root / "stem_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            proxy = module._load_proxy(
                manifest_path,
                _sha256(source_path),
                1,
                runtime_project_root=runtime_root,
                producer_project_roots=[producer_root],
            )

            self.assertTrue(proxy["valid"])
            self.assertEqual(proxy["invalid_reasons"], [])
            self.assertEqual(Path(proxy["source_audio_path"]), source_path.resolve())
            self.assertEqual(Path(proxy["audio_path"]), vocals_path.resolve())

    def test_multistem_manifest_with_exact_source_reference_is_valid_vocal_proxy(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            source_path = _write_audio(root / "source.flac", _tone(sample_rate, 1.0, 220.0, 0.2), sample_rate)
            proxy_manifest = _write_multistem_proxy_manifest(
                root / "proxy",
                model_id="scnet_multistem",
                source_path=source_path,
                vocals=_tone(sample_rate, 1.0, 660.0, 0.05),
                sample_rate=sample_rate,
            )

            proxy = module._load_proxy(proxy_manifest, _sha256(source_path), 1)

            self.assertTrue(proxy["valid"])
            self.assertEqual(proxy["invalid_reasons"], [])
            self.assertEqual(proxy["source_audio_sha256"], _sha256(source_path))

    def test_two_independent_proxies_confirm_vocal_leak(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            music = _tone(sample_rate, 1.0, 220.0, 0.35) + _tone(sample_rate, 1.0, 2100.0, 0.12)
            vocal = _tone(sample_rate, 1.0, 660.0, 0.22)
            source_path = _write_audio(root / "source.flac", music + vocal, sample_rate)
            candidate_path = _write_audio(root / "minus.flac", music + 0.55 * vocal, sample_rate)
            proxy_one = _write_proxy_manifest(
                root / "p1",
                model_id="vocal_model_one",
                source_path=source_path,
                vocals=vocal,
                sample_rate=sample_rate,
            )
            proxy_two = _write_proxy_manifest(
                root / "p2",
                model_id="vocal_model_two",
                source_path=source_path,
                vocals=0.97 * vocal,
                sample_rate=sample_rate,
            )

            manifest = module.audit_minus_quality(
                source_audio_path=source_path,
                candidate_minus_path=candidate_path,
                proxy_manifest_paths=[proxy_one, proxy_two],
                output_dir=root / "audit",
                segment_sec=0.5,
                pitch_config=PITCH_CONFIG,
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_blocked")
            self.assertFalse(manifest["overall"]["quality_gate_passed"])
            self.assertGreater(manifest["overall"]["confirmed_blocker_count"], 0)
            self.assertGreater(
                manifest["overall"]["issue_counts"].get("vocal_leak_proxy_consensus", 0),
                0,
            )
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])

    def test_clean_candidate_passes_proxy_gate_but_not_delivery(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            music = _tone(sample_rate, 1.0, 220.0, 0.35) + _tone(sample_rate, 1.0, 2100.0, 0.14)
            vocal = _tone(sample_rate, 1.0, 660.0, 0.05)
            source_path = _write_audio(root / "source.flac", music + vocal, sample_rate)
            candidate_path = _write_audio(root / "minus.flac", music, sample_rate)
            proxy_one = _write_proxy_manifest(
                root / "p1",
                model_id="vocal_model_one",
                source_path=source_path,
                vocals=vocal,
                sample_rate=sample_rate,
            )
            proxy_two_rate = 11025
            proxy_two = _write_proxy_manifest(
                root / "p2",
                model_id="vocal_model_two",
                source_path=source_path,
                vocals=_resized_tone(proxy_two_rate, 1.0, 660.0, 0.047),
                sample_rate=proxy_two_rate,
            )
            output_dir = root / "artifacts" / "audit"

            manifest = module.audit_minus_quality(
                source_audio_path=source_path,
                candidate_minus_path=candidate_path,
                proxy_manifest_paths=[proxy_one, proxy_two],
                output_dir=output_dir,
                segment_sec=0.5,
                pitch_config=PITCH_CONFIG,
                runtime_project_root=root,
                windows_project_root=r"D:\AIMusicProject",
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_pass")
            self.assertTrue(manifest["overall"]["quality_gate_passed"])
            self.assertFalse(manifest["delivery_contract"]["delivery_allowed"])
            matrix = json.loads((output_dir / "matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix["schema"], "minus_quality_ensemble.v2")
            self.assertEqual(matrix["detail_reference"]["mode"], "proxy_residual_consensus")
            self.assertTrue(matrix["pitch"]["p002"]["resampled_for_pitch"])
            self.assertNotEqual(matrix["pitch"]["p002"]["verdict"], "blocked_unknown")
            self.assertTrue(manifest["mount_contract"]["scratch_cleanup_verified"])
            self.assertEqual(
                manifest["mount_contract"]["windows_output_path"],
                r"D:\AIMusicProject\artifacts\audit",
            )
            for name in ("manifest.json", "rows.jsonl", "matrix.json", "progress.jsonl"):
                self.assertTrue((output_dir / name).is_file())

            with self.assertRaisesRegex(ValueError, "approved project artifact root"):
                module.audit_minus_quality(
                    source_audio_path=source_path,
                    candidate_minus_path=candidate_path,
                    proxy_manifest_paths=[proxy_one, proxy_two],
                    output_dir=root / "local_result",
                    segment_sec=0.5,
                    pitch_config=PITCH_CONFIG,
                    runtime_project_root=root,
                    windows_project_root=r"D:\AIMusicProject",
                )

    def test_duplicate_model_does_not_satisfy_independence(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            music = _tone(sample_rate, 1.0, 220.0, 0.35)
            vocal = _tone(sample_rate, 1.0, 660.0, 0.04)
            source_path = _write_audio(root / "source.flac", music + vocal, sample_rate)
            candidate_path = _write_audio(root / "minus.flac", music, sample_rate)
            proxy_one = _write_proxy_manifest(
                root / "p1",
                model_id="same_model",
                source_path=source_path,
                vocals=vocal,
                sample_rate=sample_rate,
            )
            proxy_two = _write_proxy_manifest(
                root / "p2",
                model_id="same_model",
                source_path=source_path,
                vocals=0.95 * vocal,
                sample_rate=sample_rate,
            )

            manifest = module.audit_minus_quality(
                source_audio_path=source_path,
                candidate_minus_path=candidate_path,
                proxy_manifest_paths=[proxy_one, proxy_two],
                output_dir=root / "audit",
                segment_sec=0.5,
                pitch_config=PITCH_CONFIG,
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_unknown")
            self.assertIn(
                "insufficient_independent_vocal_proxies",
                manifest["overall"]["quality_blocker_reasons"],
            )
            matrix = json.loads((root / "audit" / "matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix["independence"]["selected_proxy_ids"], ["p001"])
            self.assertIn("duplicate_model_identity", matrix["independence"]["excluded"][0]["reasons"])

    def test_source_conservation_hole_blocks_without_vocal_consensus(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            music = _tone(sample_rate, 2.0, 220.0, 0.35) + _tone(sample_rate, 2.0, 2100.0, 0.12)
            vocal = _tone(sample_rate, 2.0, 660.0, 0.03)
            source = music + vocal
            candidate = music.copy()
            candidate[sample_rate:] *= 0.30
            source_path = _write_audio(root / "source.flac", source, sample_rate)
            candidate_path = _write_audio(root / "minus.flac", candidate, sample_rate)
            proxy_one = _write_proxy_manifest(
                root / "p1",
                model_id="vocal_model_one",
                source_path=source_path,
                vocals=vocal,
                sample_rate=sample_rate,
            )
            proxy_two = _write_proxy_manifest(
                root / "p2",
                model_id="vocal_model_two",
                source_path=source_path,
                vocals=0.96 * vocal,
                sample_rate=sample_rate,
            )

            manifest = module.audit_minus_quality(
                source_audio_path=source_path,
                candidate_minus_path=candidate_path,
                proxy_manifest_paths=[proxy_one, proxy_two],
                output_dir=root / "audit",
                segment_sec=1.0,
                pitch_config=PITCH_CONFIG,
            )

            self.assertEqual(manifest["overall"]["status"], "proxy_blocked")
            self.assertGreater(
                manifest["overall"]["issue_counts"].get("low_rms_muffling_proxy", 0),
                0,
            )
            rows = [json.loads(line) for line in (root / "audit" / "rows.jsonl").read_text().splitlines()]
            self.assertFalse(rows[0]["blocker"])
            self.assertTrue(rows[1]["blocker"])

    def test_expected_vocal_removal_is_not_mislabeled_as_detail_loss(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 8000
            music = _tone(sample_rate, 1.0, 220.0, 0.32) + _tone(sample_rate, 1.0, 3000.0, 0.05)
            vocal = _tone(sample_rate, 1.0, 2100.0, 0.32)
            source_path = _write_audio(root / "source.flac", music + vocal, sample_rate)
            candidate_path = _write_audio(root / "minus.flac", music, sample_rate)
            proxy_one = _write_proxy_manifest(
                root / "p1",
                model_id="vocal_model_one",
                source_path=source_path,
                vocals=vocal,
                sample_rate=sample_rate,
            )
            proxy_two = _write_proxy_manifest(
                root / "p2",
                model_id="vocal_model_two",
                source_path=source_path,
                vocals=0.98 * vocal,
                sample_rate=sample_rate,
            )

            manifest = module.audit_minus_quality(
                source_audio_path=source_path,
                candidate_minus_path=candidate_path,
                proxy_manifest_paths=[proxy_one, proxy_two],
                output_dir=root / "audit",
                segment_sec=0.5,
                pitch_config=PITCH_CONFIG,
            )

            rows = [json.loads(line) for line in (root / "audit" / "rows.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["overall"]["status"], "proxy_pass")
            self.assertTrue(manifest["overall"]["quality_gate_passed"])
            self.assertTrue(
                any("presence_loss_proxy" in row["detail_reference"]["source_diagnostic_only_issues"] for row in rows)
            )
            self.assertTrue(
                all(not evidence["detail_issues"] for row in rows for evidence in row["detail_reference"]["proxy_residual_evidence"])
            )


if __name__ == "__main__":
    unittest.main()
