from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_ROOT / "virtualdj_baseline_manifest.py"


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("virtualdj_baseline_manifest", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestVirtualDjBaselineManifest(unittest.TestCase):
    def test_build_manifest_parses_settings_database_and_export_roles(self):
        module = _load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            virtualdj_root = temp_path / "VirtualDJ"
            export_root = temp_path / "exports"
            virtualdj_root.mkdir()
            (virtualdj_root / "Cache").mkdir()
            (virtualdj_root / "History").mkdir()
            export_root.mkdir()

            (virtualdj_root / "settings.xml").write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<settings>
  <automation>
    <autoBPMMatch>smart</autoBPMMatch>
    <resetStemsOnLoad>Pads</resetStemsOnLoad>
  </automation>
  <controls>
    <defaultStemsEqMode>EZRemix</defaultStemsEqMode>
    <stemsBleedMuteVocal>0.80</stemsBleedMuteVocal>
  </controls>
  <browser>
    <cdjExportStemsConfig>Acapella=vocals, Instrumental=instru+bass+kick+hihat</cdjExportStemsConfig>
    <cdjExportStems>Acapella</cdjExportStems>
    <cdjExportCompatibility>CDJ2000nexus</cdjExportCompatibility>
  </browser>
  <internet>
    <iRemoteDefaultPort>4243</iRemoteDefaultPort>
    <oscPort>0</oscPort>
  </internet>
  <performance>
    <stemsRealtimeSeparation>Always</stemsRealtimeSeparation>
    <stemsSavedStems>Manual</stemsSavedStems>
    <stemsGPU>Automatic</stemsGPU>
    <sampleRate>44100</sampleRate>
  </performance>
  <record>
    <recordFormat>mp3</recordFormat>
  </record>
  <options>
    <lastUsedBuild>9005</lastUsedBuild>
  </options>
</settings>
""",
                encoding="utf-8",
            )
            (virtualdj_root / "database.xml").write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<VirtualDJ_Database Version="2026">
  <Song FilePath="C:\\Music\\fixture.mp3" FileSize="123">
    <Infos SongLength="10.0" />
    <Scan Bpm="0.500000" Key="Am" />
    <Poi Pos="1.0" Type="beatgrid" />
  </Song>
</VirtualDJ_Database>
""",
                encoding="utf-8",
            )
            (virtualdj_root / "Cache" / "cache.db").write_bytes(b"SQLite format 3\x00fixture")
            (virtualdj_root / "History" / "tracklist.txt").write_text("fixture\n", encoding="utf-8")
            (export_root / "Nine Thou Acapella.wav").write_bytes(b"vocals")
            (export_root / "Nine Thou Instrumental.flac").write_bytes(b"instrumental")
            (export_root / "Nine Thou Beat.mp3").write_bytes(b"beat")

            manifest = module.build_manifest(virtualdj_data_root=virtualdj_root, export_root=export_root)

        self.assertEqual(manifest["virtualdj"]["settings"]["stems"]["stemsRealtimeSeparation"], "Always")
        self.assertEqual(manifest["virtualdj"]["settings"]["remote_and_transport"]["iRemoteDefaultPort"], "4243")
        self.assertEqual(manifest["virtualdj"]["database"]["song_count"], 1)
        self.assertEqual(manifest["virtualdj"]["database"]["songs"][0]["scan"]["Key"], "Am")
        self.assertEqual(manifest["virtualdj"]["cache"]["files"][0]["format_hint"], "sqlite")
        self.assertEqual(
            manifest["baseline_exports"]["detected_roles"],
            ["drums", "instrumental_mix", "vocals"],
        )
        self.assertEqual(
            manifest["baseline_exports"]["selected_reference_minus"]["docker_path"],
            "/virtualdj_export/Nine Thou Instrumental.flac",
        )
        self.assertTrue(manifest["quality_gate_hint"]["can_run_nine_thou_proxy_gate"])

    def test_single_unknown_audio_defaults_to_instrumental_mix(self):
        module = _load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = Path(temp_dir)
            (export_root / "baseline.flac").write_bytes(b"single")

            exports = module.classify_virtualdj_exports(export_root)

        self.assertEqual(exports["detected_roles"], ["instrumental_mix"])
        self.assertEqual(exports["items"][0]["role_inference"], "single_audio_file_defaulted_to_instrumental_mix")
        self.assertEqual(exports["selected_reference_minus"]["role"], "instrumental_mix")

    def test_missing_export_dir_is_reported_without_failure(self):
        module = _load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = Path(temp_dir) / "missing"

            exports = module.classify_virtualdj_exports(export_root)

        self.assertFalse(exports["exists"])
        self.assertEqual(exports["file_count"], 0)
        self.assertEqual(exports["missing_recommended_roles"], ["vocals", "instrumental_mix"])
        self.assertIsNone(exports["selected_reference_minus"])

    def test_cli_writes_manifest_json(self):
        module = _load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            virtualdj_root = temp_path / "VirtualDJ"
            export_root = temp_path / "exports"
            output_json = temp_path / "manifest.json"
            virtualdj_root.mkdir()
            export_root.mkdir()
            (virtualdj_root / "settings.xml").write_text("<settings />", encoding="utf-8")
            (virtualdj_root / "database.xml").write_text("<VirtualDJ_Database Version=\"2026\" />", encoding="utf-8")

            exit_code = module.main(
                [
                    "--virtualdj-data-root",
                    str(virtualdj_root),
                    "--export-dir",
                    str(export_root),
                    "--output-json",
                    str(output_json),
                ]
            )

            saved = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(saved["schema_version"], 1)
        self.assertEqual(saved["virtualdj"]["database"]["song_count"], 0)


if __name__ == "__main__":
    unittest.main()
