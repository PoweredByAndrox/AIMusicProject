"""Apply STFT-domain gain masking on original instrumental — removes leak without holes."""
from __future__ import annotations
import sys, json, shutil, time
from pathlib import Path

# Accept optional alias arg
target_alias = sys.argv[1] if len(sys.argv) > 1 else None

sys.path.insert(0, "/Users/andrejviluha/AIMusicProject_win/ml_worker_improved")
from tools.apply_actual_masked_residue_minus_cleanup import apply_actual_masked_residue_cleanup
from tools.mac_melband_pitch_guard import score_pitch_shifted_vocal_leak
from tools.stem_region_repair_common import read_audio_stereo

PROJECT = Path("/Users/andrejviluha/AIMusicProject_win")

tracks = [
    ("armin-sound-of-you", "armin"),
    ("gradusi_final", "gradusi"),
    ("leps-ona-ne-tvoya", "leps"),
    ("nen-hot-tiktok-douyin-2025", "nhot"),
    ("titus-let-it-rip-chipcat", "titus"),
]

strengths = [0.1, 0.2, 0.3]

for track_id, alias in tracks:
    if target_alias and alias != target_alias:
        continue
    macq = PROJECT / "artifacts/mac_quick" / track_id
    instr_path = macq / "instrumental.flac"
    vocals_path = macq / "vocals.flac"

    if not instr_path.exists() or not vocals_path.exists():
        print(f"SKIP {alias}: missing audio")
        continue

    instr_audio, sr = read_audio_stereo(instr_path)
    vocals_audio, sr2 = read_audio_stereo(vocals_path)
    assert sr == sr2

    n = min(instr_audio.shape[0], vocals_audio.shape[0])
    instr_audio = instr_audio[:n]
    vocals_audio = vocals_audio[:n]

    for s in strengths:
        out_dir = PROJECT / "artifacts/mac_quick_metrics/stft_cleanup" / alias / f"s{s}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        refined, report = apply_actual_masked_residue_cleanup(
            base_minus=instr_audio,
            primary_vocals=vocals_audio,
            sample_rate=sr,
            strength=s,
            mask_power=1.35,
            smoothing_sec=0.10,
            window_sec=5.0,
            include_low_priority_windows=False,
            min_low_priority_score=0.0,
            base_overlap_power=0.0,
        )
        elapsed = time.time() - t0

        from tools.stem_region_repair_common import write_audio
        refined_path = out_dir / "m001.flac"
        write_audio(refined_path, refined, sr)

        pg_dir = out_dir / "pitch_guard"
        pg_dir.mkdir(parents=True)
        pg = score_pitch_shifted_vocal_leak(
            vocals_path=vocals_path,
            instrumental_path=refined_path,
            output_dir=pg_dir,
        )
        r_count = pg.get("total_fail_regions", len(pg.get("regions", [])))
        status = report.get("cleanup_status", "?")
        res_windows = report.get("review_window_count", 0)

        out_dir.joinpath("result.json").write_text(json.dumps({"strength": s, "refined_regions": r_count, "status": status}))

        print(f"{alias} s={s}: {r_count} regions | {status} | {elapsed:.0f}s | windows={res_windows}")

print("\nDone")
