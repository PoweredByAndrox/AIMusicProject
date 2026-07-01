from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    sys.modules.setdefault("numpy", types.SimpleNamespace())
    sys.modules.setdefault("soundfile", types.SimpleNamespace())
    module_path = TOOLS_ROOT / "run_registered_model_candidate.py"
    spec = importlib.util.spec_from_file_location("run_registered_model_candidate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRegisteredModelCandidateRoleSafeManifests(unittest.TestCase):
    def test_minus_target_records_instrumental_alias_without_renaming_files(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            instrumental_path = tmp_path / "instrumental.flac"
            vocals_path = tmp_path / "vocals.flac"
            instrumental_path.write_bytes(b"fake instrumental")
            vocals_path.write_bytes(b"fake residual vocal companion")
            manifest_path = tmp_path / "stem_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": "becruily_instrumental_vocals",
                        "stems": {
                            "vocals": {
                                "path": str(vocals_path),
                                "role": "target",
                                "source": "residual_from_raw_minus_instrumental",
                                "model_output": False,
                                "residual_companion": True,
                            },
                            "instrumental": {
                                "path": str(instrumental_path),
                                "role": "companion",
                                "source": "model_output",
                                "model_output": True,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifact = module.ensure_target_stem_artifact(
                candidate_manifest_path=manifest_path,
                target_stem="minus",
                output_dir=output_dir,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(artifact["target_stem"], "minus")
            self.assertEqual(artifact["manifest_stem_name"], "instrumental")
            self.assertEqual(artifact["canonical_target_stem"], "minus")
            self.assertEqual(artifact["user_facing_role"], "minus")
            self.assertEqual(artifact["path"], str(instrumental_path))

            minus_entry = manifest["stems"]["minus"]
            self.assertEqual(minus_entry["path"], str(instrumental_path))
            self.assertEqual(minus_entry["role"], "target")
            self.assertEqual(minus_entry["canonical_stem"], "minus")
            self.assertEqual(minus_entry["user_facing_role"], "minus")
            self.assertEqual(minus_entry["manifest_stem_alias_of"], "instrumental")
            self.assertEqual(minus_entry["source_stems"], ["instrumental"])

            self.assertEqual(manifest["target_stem"], "minus")
            self.assertEqual(manifest["target_stem_artifact"]["manifest_stem_name"], "instrumental")
            self.assertEqual(manifest["target_stem_artifact"]["user_facing_role"], "minus")


if __name__ == "__main__":
    unittest.main()
