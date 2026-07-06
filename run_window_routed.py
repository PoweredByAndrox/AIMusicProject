"""Per-window routing: for windows WITH leak regions → apply refiner; for clean → keep original."""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, time
from tools.mac_melband_pitch_guard import score_pitch_shifted_vocal_leak
from tools.stem_region_repair_common import read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint, resolve_region_repair_training_device

ISSUE_VOCAL_LEAK = "vocal_leak"

def run_window_routed(
    track_alias: str,
    vocals_path: Path,
    instrumental_path: Path,
    raw_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    win_sec: float = 5.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_region_repair_training_device("mps")
    
    vocals, sr = read_audio_stereo(vocals_path)
    instrumental, sr2 = read_audio_stereo(instrumental_path)
    raw, sr3 = read_audio_stereo(raw_path)
    assert sr == sr2 == sr3
    
    total_frames = min(vocals.shape[0], instrumental.shape[0], raw.shape[0])
    win_frames = int(round(win_sec * sr))
    
    # Get full-track leak regions from original score
    full_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_path, instrumental_path=instrumental_path,
        output_dir=output_dir / "full_score",
    )
    all_regions = full_score.get("regions", [])
    print(f"Full track: {full_score.get('total_fail_regions', len(all_regions))} leak regions")
    
    # Load model once
    model, _ = load_region_repair_checkpoint(checkpoint_path)
    
    stitched_voc = vocals[:total_frames].copy()
    stitched_instr = instrumental[:total_frames].copy()
    decisions = []
    
    for start in range(0, total_frames, win_frames):
        end = min(start + win_frames, total_frames)
        if end - start < sr // 2:
            continue
        
        w_sec_start = start / sr
        w_sec_end = end / sr
        
        # Count leak regions in this window
        region_count = sum(1 for r in all_regions if r.get("start", 0) < w_sec_end and r.get("end", 0) > w_sec_start)
        
        if region_count == 0:
            decisions.append({"win": start//win_frames, "s": round(w_sec_start,1), "e": round(w_sec_end,1), "regions": 0, "use_ref": False})
            continue
        
        # Apply refiner to this window
        w_v = vocals[start:end]
        w_i = instrumental[start:end]
        w_r = raw[start:end]
        w_mask = np.ones(end - start, dtype=np.float32)
        
        repaired, diag = apply_region_repair_model(
            model=model, mix_audio=w_r, bad_stem_audio=w_i,
            source_condition_audio=w_v.copy(), region_mask=w_mask,
            sample_rate=sr, issue_type=ISSUE_VOCAL_LEAK, device=device,
        )
        delta = repaired[:end-start] - w_i
        delta_rms = float(np.sqrt(np.mean(np.square(delta.astype(np.float64)))))
        
        use_ref = delta_rms < 0.05  # safety: don't apply if changes are too large
        
        if use_ref:
            stitched_instr[start:end] = np.clip(w_i + delta, -1.0, 1.0)
            stitched_voc[start:end] = np.clip(w_v - delta, -1.0, 1.0)
        
        decisions.append({"win": start//win_frames, "s": round(w_sec_start,1), "e": round(w_sec_end,1), "regions": region_count, "delta_rms": round(delta_rms, 6), "use_ref": use_ref})
    
    n_ref = sum(1 for d in decisions if d["use_ref"])
    print(f"Windows: {len(decisions)}, refined: {n_ref}/{len(decisions)}")
    
    ad = output_dir / "audio"
    write_audio(ad / "v.flac", stitched_voc, sr)
    write_audio(ad / "i.flac", stitched_instr, sr)
    write_audio(ad / "raw.flac", raw[:total_frames], sr)
    
    final_score = score_pitch_shifted_vocal_leak(
        vocals_path=ad / "v.flac", instrumental_path=ad / "i.flac",
        output_dir=output_dir / "score_final",
    )
    
    result = {
        "track": track_alias,
        "total_windows": len(decisions),
        "refined_windows": n_ref,
        "original_regions": full_score.get("total_fail_regions", len(all_regions)),
        "final_regions": final_score.get("total_fail_regions", 0),
        "final_verdict": final_score.get("verdict"),
    }
    write_json(output_dir / "result.json", result)
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    
    base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics")
    pg = base / "pitch_guard" / args.track
    
    if args.checkpoint:
        cp = Path(args.checkpoint)
    else:
        # Use best available checkpoint
        for candidate in [
            base / "training" / args.track / "model.pt",
            base / "training" / "nen-hot-tiktok-douyin-2025_v5_heavy" / "model.pt",
        ]:
            if candidate.exists():
                cp = candidate
                break
        else:
            raise FileNotFoundError("No checkpoint found")
    
    result = run_window_routed(
        track_alias=args.track,
        vocals_path=pg / "audio" / "v.flac",
        instrumental_path=pg / "audio" / "i.flac",
        raw_path=pg / "audio" / "raw.flac",
        checkpoint_path=cp,
        output_dir=base / "window_routed" / args.track,
    )
    print(json.dumps(result, indent=2))
