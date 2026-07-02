import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_DIR / "build_minus_hardcase_runtime_root.py"
MODULE_SPEC = importlib.util.spec_from_file_location("build_minus_hardcase_runtime_root", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"cannot load build_minus_hardcase_runtime_root from {MODULE_PATH}")
build_minus_hardcase_runtime_root_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(build_minus_hardcase_runtime_root_module)

build_minus_hardcase_runtime_root = build_minus_hardcase_runtime_root_module.build_minus_hardcase_runtime_root


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def touch_audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return str(path)


class BuildMinusHardcaseRuntimeRootTest(unittest.TestCase):
    def test_builds_runtime_root_from_stem_manifest_with_path_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            old_root = "/old/local"
            new_root = tmp_path / "mounted"
            mix = touch_audio(new_root / "artifacts" / "song" / "raw.flac")
            vocal = touch_audio(new_root / "artifacts" / "song" / "vocals.flac")
            minus = touch_audio(new_root / "artifacts" / "song" / "instrumental.flac")
            manifest = write_json(
                tmp_path / "src" / "stem_manifest.json",
                {
                    "source_audio_path": f"{old_root}/artifacts/song/raw.flac",
                    "model_id": "becruily_karaoke_vocals",
                    "model": {"family": "roformer"},
                    "stems": {
                        "vocals": {"path": f"{old_root}/artifacts/song/vocals.flac"},
                        "instrumental": {"path": f"{old_root}/artifacts/song/instrumental.flac"},
                    },
                    "delivery_allowed": False,
                    "release_ready": False,
                },
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[manifest],
                candidate_jsonls=[],
                output_dir=tmp_path / "out",
                path_rewrites=[(old_root, str(new_root))],
            )

            rows = read_jsonl(tmp_path / "out" / "candidate_pool.jsonl")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(rows[0]["mix_path"], mix)
            self.assertEqual(rows[0]["vocal_path"], vocal)
            self.assertEqual(rows[0]["minus_path"], minus)
            self.assertEqual(rows[0]["stem_paths"]["vocals"]["path"], vocal)
            self.assertEqual(rows[0]["stem_paths"]["instrumental"]["path"], minus)
            self.assertEqual(rows[0]["route_id"], "becruily_karaoke_vocals")
            self.assertIs(rows[0]["delivery_allowed"], False)
            self.assertIs(rows[0]["release_ready"], False)
            self.assertEqual(read_jsonl(tmp_path / "out" / "selected_candidates.jsonl"), [])
            self.assertEqual(read_jsonl(tmp_path / "out" / "pair_compatibility.jsonl"), [])
            self.assertIs(read_json(tmp_path / "out" / "preview_render_contract.json")["release_ready"], False)

    def test_builds_runtime_root_from_candidate_row_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            mix = touch_audio(tmp_path / "raw.flac")
            vocal = touch_audio(tmp_path / "vocals.flac")
            minus = touch_audio(tmp_path / "minus.flac")
            manifest = write_json(
                tmp_path / "lkg001" / "manifest.json",
                {
                    "candidate_row": {
                        "track_id": "track-a",
                        "candidate_id": "cand-a",
                        "route_id": "route-a",
                        "mix_path": mix,
                        "vocal_path": vocal,
                        "minus_path": minus,
                        "candidate_family": "judge_guided_minus_vocal_leakage_repair",
                        "delivery_allowed": False,
                        "release_ready": False,
                    }
                },
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[manifest],
                candidate_jsonls=[],
                output_dir=tmp_path / "out",
                path_rewrites=[],
            )

            rows = read_jsonl(tmp_path / "out" / "candidate_pool.jsonl")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(rows[0]["track_id"], "track-a")
            self.assertEqual(rows[0]["candidate_id"], "cand-a")
            self.assertEqual(rows[0]["selected_minus_path"], minus)
            self.assertEqual(rows[0]["source_candidate_family"], "judge_guided_minus_vocal_leakage_repair")

    def test_skips_manifest_with_missing_audio_without_synthetic_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            manifest = write_json(
                tmp_path / "bad" / "stem_manifest.json",
                {
                    "source_audio_path": str(tmp_path / "missing-raw.flac"),
                    "model_id": "bad-model",
                    "stems": {
                        "vocals": {"path": str(tmp_path / "missing-vocals.flac")},
                        "instrumental": {"path": str(tmp_path / "missing-minus.flac")},
                    },
                },
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[manifest],
                candidate_jsonls=[],
                output_dir=tmp_path / "out",
                path_rewrites=[],
            )

            self.assertEqual(result["candidate_count"], 0)
            skipped = read_jsonl(tmp_path / "out" / "skipped.jsonl")
            self.assertTrue(skipped)
            self.assertIn("missing_audio_file:mix_path", skipped[0]["skip_reason"])
            self.assertEqual(read_jsonl(tmp_path / "out" / "candidate_pool.jsonl"), [])

    def test_prefers_manifest_raw_ref_artifact_over_stale_source_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            old_root = "/old/local"
            new_root = tmp_path / "mounted"
            mix = touch_audio(new_root / "artifacts" / "mac_api" / "run" / "raw_ref.flac")
            vocal = touch_audio(new_root / "artifacts" / "mac_api" / "run" / "vocals.flac")
            minus = touch_audio(new_root / "artifacts" / "mac_api" / "run" / "instrumental.flac")
            manifest = write_json(
                tmp_path / "run" / "stem_manifest.json",
                {
                    "source_audio_path": f"{old_root}/source/t001.mp3",
                    "model_id": "model-a",
                    "stems": {
                        "vocals": {"path": f"{old_root}/artifacts/mac_api/run/vocals.flac"},
                        "instrumental": {"path": f"{old_root}/artifacts/mac_api/run/instrumental.flac"},
                    },
                    "artifacts": {
                        "raw_ref_flac": f"{old_root}/artifacts/mac_api/run/raw_ref.flac",
                    },
                },
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[manifest],
                candidate_jsonls=[],
                output_dir=tmp_path / "out",
                path_rewrites=[(old_root, str(new_root))],
            )

            rows = read_jsonl(tmp_path / "out" / "candidate_pool.jsonl")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(rows[0]["mix_path"], mix)
            self.assertEqual(rows[0]["vocal_path"], vocal)
            self.assertEqual(rows[0]["minus_path"], minus)
            self.assertEqual(rows[0]["stem_paths"]["vocals"]["path"], vocal)

    def test_residual_vocal_companion_from_instrumental_manifest_is_not_selected_vocal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            mix = touch_audio(tmp_path / "run" / "raw_ref.flac")
            vocal = touch_audio(tmp_path / "run" / "vocals.flac")
            minus = touch_audio(tmp_path / "run" / "instrumental.flac")
            manifest = write_json(
                tmp_path / "run" / "stem_manifest.json",
                {
                    "source_audio_path": mix,
                    "target_stem": "instrumental",
                    "model_id": "becruily_instrumental_vocals",
                    "stems": {
                        "vocals": {
                            "path": vocal,
                            "source": "residual_from_raw_minus_instrumental",
                            "model_output": False,
                            "residual_companion": True,
                        },
                        "instrumental": {"path": minus, "source": "model_output", "model_output": True},
                    },
                    "artifacts": {"raw_ref_flac": mix},
                },
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[manifest],
                candidate_jsonls=[],
                output_dir=tmp_path / "out",
                path_rewrites=[],
            )

            rows = read_jsonl(tmp_path / "out" / "candidate_pool.jsonl")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(rows[0]["minus_path"], minus)
            self.assertEqual(rows[0]["vocal_path"], "")
            self.assertEqual(rows[0]["selected_vocal_path"], "")
            self.assertEqual(rows[0]["vocal_companion_path"], vocal)
            self.assertEqual(rows[0]["vocal_companion_role"], "residual_companion")
            self.assertEqual(rows[0]["vocal_companion_source"], "residual_from_raw_minus_instrumental")

    def test_candidate_jsonl_rows_are_normalized_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            mix = touch_audio(tmp_path / "raw.flac")
            vocal = touch_audio(tmp_path / "vocals.flac")
            minus = touch_audio(tmp_path / "minus.flac")
            source = write_jsonl(
                tmp_path / "rows.jsonl",
                [
                    {"track_id": "t", "candidate_id": "c", "mix_path": mix, "vocal_path": vocal, "minus_path": minus},
                    {"track_id": "t", "candidate_id": "c", "mix_path": mix, "vocal_path": vocal, "minus_path": minus},
                ],
            )

            result = build_minus_hardcase_runtime_root(
                candidate_manifests=[],
                candidate_jsonls=[source],
                output_dir=tmp_path / "out",
                path_rewrites=[],
            )

            rows = read_jsonl(tmp_path / "out" / "candidate_pool.jsonl")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score_mode"], "minus")
            self.assertIs(rows[0]["release_ready"], False)


if __name__ == "__main__":
    unittest.main()
