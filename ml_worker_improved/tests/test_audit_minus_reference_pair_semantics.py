from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audit_minus_reference_pair_semantics import main
from tools.build_full_pack_reference_corpus_intake import minus_reference_semantic_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "audit_minus_reference_pair_semantics.py"


def write_audio(path: Path, data: np.ndarray, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), sample_rate)
    return path


def sine(freq_hz: float, *, amplitude: float, duration_sec: float = 2.0, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    mono = amplitude * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    return np.column_stack([mono, mono])


def build_role_pair(
    root: Path,
    *,
    swapped: bool = False,
    reference_vocal_amplitude: float = 0.01,
    source_vocal_amplitudes: tuple[float, float] = (0.22, 0.20),
    reference_vocal_amplitudes: tuple[float, float] = (0.01, 0.008),
) -> dict[str, Path]:
    music = sine(110.0, amplitude=0.10)
    vocal = sine(440.0, amplitude=0.24)
    full_mix = music + vocal
    minus_reference = music + (reference_vocal_amplitude * vocal)
    source_mix_path = write_audio(root / "source_mix.flac", minus_reference if swapped else full_mix)
    minus_reference_path = write_audio(root / "minus_reference.flac", full_mix if swapped else minus_reference)

    d1_source = write_audio(root / "d1_source.flac", sine(440.0, amplitude=source_vocal_amplitudes[0]))
    d2_source = write_audio(root / "d2_source.flac", sine(440.0, amplitude=source_vocal_amplitudes[1]))
    d1_reference = write_audio(root / "d1_reference.flac", sine(440.0, amplitude=reference_vocal_amplitudes[0]))
    d2_reference = write_audio(root / "d2_reference.flac", sine(440.0, amplitude=reference_vocal_amplitudes[1]))
    if swapped:
        d1_source, d1_reference = d1_reference, d1_source
        d2_source, d2_reference = d2_reference, d2_source
    return {
        "source_mix": source_mix_path,
        "minus_reference": minus_reference_path,
        "d1_source": d1_source,
        "d1_reference": d1_reference,
        "d2_source": d2_source,
        "d2_reference": d2_reference,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_accepted_audit_writes_short_schema_exact_outputs(tmp_path: Path) -> None:
    files = build_role_pair(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
            "--detector",
            "detector_b",
            str(files["d2_source"]),
            str(files["d2_reference"]),
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output_dir.iterdir()} == {"sem.json", "progress.jsonl"}
    sem = read_json(output_dir / "sem.json")
    progress = read_jsonl(output_dir / "progress.jsonl")

    assert sem["schema"] == "minus_reference_semantic_audit.v1"
    assert sem["decision"] == "accepted"
    assert sem["training_allowed"] is True
    assert sem["delivery_allowed"] is False
    assert sem["detector_count"] == 2
    assert sem["source_mix"]["path"] == str(files["source_mix"])
    assert sem["minus_reference"]["path"] == str(files["minus_reference"])
    assert sorted(sem["evidence"]["accepted_support_detector_ids"]) == ["detector_a", "detector_b"]
    assert sem["evidence"]["reference_median_vocal_to_mix_rms_db"] <= -20.0
    assert sem["evidence"]["median_source_reference_gap_db"] >= 6.0
    assert sem["asr_dependency"] is False
    assert sem["audio_copied"] is False
    assert sem["audio_exported"] is False
    assert [row["event"] for row in progress] == ["start", "detector_scored", "detector_scored", "finish"]
    assert progress[-1]["decision"] == "accepted"
    assert progress[-1]["training_allowed"] is True


def test_roles_reversed_when_reference_gap_is_reversed(tmp_path: Path) -> None:
    files = build_role_pair(tmp_path, swapped=True)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
            "--detector",
            "detector_b",
            str(files["d2_source"]),
            str(files["d2_reference"]),
        ]
    )

    assert exit_code == 2
    sem = read_json(output_dir / "sem.json")
    assert sem["decision"] == "roles_reversed"
    assert sem["training_allowed"] is False
    assert sem["delivery_allowed"] is False
    assert sem["evidence"]["median_source_reference_gap_db"] <= -6.0
    assert sorted(sem["evidence"]["roles_reversed_support_detector_ids"]) == ["detector_a", "detector_b"]


def test_unresolved_when_reference_vocal_share_stays_too_high(tmp_path: Path) -> None:
    files = build_role_pair(
        tmp_path,
        reference_vocal_amplitude=0.80,
        source_vocal_amplitudes=(0.22, 0.21),
        reference_vocal_amplitudes=(0.16, 0.15),
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
            "--detector",
            "detector_b",
            str(files["d2_source"]),
            str(files["d2_reference"]),
        ]
    )

    assert exit_code == 2
    sem = read_json(output_dir / "sem.json")
    assert sem["decision"] == "unresolved"
    assert sem["training_allowed"] is False
    assert "reference_vocal_share_above_max" in sem["reasons"]
    assert sem["evidence"]["reference_median_vocal_to_mix_rms_db"] > -20.0


def test_blocks_when_unique_detector_quorum_is_missing(tmp_path: Path) -> None:
    files = build_role_pair(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
        ]
    )

    assert exit_code == 2
    sem = read_json(output_dir / "sem.json")
    assert sem["decision"] == "blocked"
    assert sem["training_allowed"] is False
    assert sem["delivery_allowed"] is False
    assert "insufficient_unique_detectors" in sem["reasons"]
    assert sem["detector_count"] == 1


def test_output_matches_existing_intake_consumer_contract(tmp_path: Path) -> None:
    files = build_role_pair(tmp_path)
    output_dir = tmp_path / "bundle"
    manifest_path = output_dir / "ground_truth_manifest.json"

    exit_code = main(
        [
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
            "--detector",
            "detector_b",
            str(files["d2_source"]),
            str(files["d2_reference"]),
        ]
    )

    assert exit_code == 0
    manifest_path.write_text("{}", encoding="utf-8")
    summary, blockers = minus_reference_semantic_audit(
        manifest_path=manifest_path,
        source_audio_path=str(files["source_mix"]),
        target_audio_path=str(files["minus_reference"]),
        stem="minus",
    )

    assert blockers == []
    assert summary["status"] == "accepted"
    assert summary["decision"] == "accepted"
    assert summary["training_allowed"] is True
    assert summary["detector_count"] == 2


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    files = build_role_pair(tmp_path)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_minus_reference_pair_semantics.py",
            str(files["source_mix"]),
            str(files["minus_reference"]),
            str(output_dir),
            "--detector",
            "detector_a",
            str(files["d1_source"]),
            str(files["d1_reference"]),
            "--detector",
            "detector_b",
            str(files["d2_source"]),
            str(files["d2_reference"]),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "sem.json").exists()
    payload = json.loads(result.stdout.strip())
    assert payload["decision"] == "accepted"


def test_main_starts_with_shared_docker_runtime_guard() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    import_names = set()
    main_function = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "tools.docker_runtime":
            import_names.update(alias.name for alias in node.names)
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_function = node

    assert "require_docker_runtime" in import_names
    assert main_function is not None

    body = list(main_function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    assert body
    first = body[0]
    assert isinstance(first, ast.Expr)
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == "require_docker_runtime"
