from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_instrumental_from_stem_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_instrumental_from_stem_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch_audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return str(path)


def test_default_excluded_stems_skip_vocals_and_vocal_alias(tmp_path: Path) -> None:
    module = load_module()
    vocals_path = touch_audio(tmp_path / "vocals.flac")
    drums_path = touch_audio(tmp_path / "drums.flac")
    manifest = {
        "stems": {
            "vocals": {"path": vocals_path},
            "vocal": {"path": vocals_path},
            "drums": {"path": drums_path},
        }
    }

    selected = module.resolve_manifest_stems(manifest, set(module.DEFAULT_EXCLUDED_STEMS))

    assert selected == [("drums", Path(drums_path))]
