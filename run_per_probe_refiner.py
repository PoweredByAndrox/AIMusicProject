"""Per-probe refiner training + region-routed application."""
from __future__ import annotations
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
from tools.mac_melband_pitch_guard import score_pitch_shifted_vocal_leak
from tools.stem_region_repair_common import read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import apply_region_repair_model, load_region_repair_checkpoint, resolve_region_repair_training_device
import json, numpy as np, time, shutil

ISSUE_VOCAL_LEAK = "vocal_leak"

def train_per_probe_refiners(
    *,
    track_alias: str,
    vocals_path: Path,
    instrumental_path: Path,
    raw_path: Path,
    base_output: Path,
    probe_values: list[float] | None = None,
    epochs: int = 400,
):
    if probe_values is None:
        probe_values = [0.0, -12.0, -7.0, -5.0]
    
    out = base_output / track_alias
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Score original
    original_score = score_pitch_shifted_vocal_leak(
        vocals_path=vocals_path, instrumental_path=instrumental_path,
        output_dir=out / "score_original",
    )
    all_regions = original_score.get("regions", [])
    total = original_score.get("total_fail_regions", 0)
    tax = original_score.get("failure_taxonomy", {}).get("by_probe", {})
    print(f"{track_alias}: {total} regions, by_probe={tax}")
    
    if total == 0:
        print("  No leaks, skipping")
        return None
    
    # 2. Train per-probe refiners
    trained = {}
    for pv in probe_values:
        ps = f"{pv:.1f}"
        pr = [r for r in all_regions if abs(float(r.get("probe_semitones", r.get("semitones", 0))) - pv) < 0.01]
        if len(pr) < 2:
            continue
        
        gd = out / "guards" / f"p{ps.replace('.','_').replace('-','n')}"
        gd.mkdir(parents=True, exist_ok=True)
        ad = gd / "audio"; ad.mkdir(exist_ok=True)
        shutil.copy2(vocals_path, ad / "v.flac")
        shutil.copy2(instrumental_path, ad / "i.flac")
        shutil.copy2(raw_path, ad / "raw.flac")
        
        sm = dict(original_score)
        sm["regions"] = pr[:16]
        sm["total_fail_regions"] = len(pr)
        sm["selected_region_count"] = min(len(pr), 16)
        write_json(gd / "manifest.json", {
            "schema": "mac_melband_pitch_guard_run.v1",
            "score": sm,
            "audio": {"vocals": str(ad/"v.flac"), "instrumental": str(ad/"i.flac"), "raw_ref": str(ad/"raw.flac")},
        })
        
        print(f"  Training refiner for probe {ps} ({len(pr)} regions)...")
        t0 = time.time()
        m = run_training(
            mel_guard_dir=gd,
            output_dir=out / "training" / f"p{ps.replace('.','_').replace('-','n')}",
            target_gain=2.5, epochs=epochs, require_mps=True,
            max_regions=min(len(pr), 16),
            objective_profile="projection_teacher",
        )
        cp = m.get("checkpoint")
        if cp and Path(str(cp)).exists():
            trained[ps] = {"checkpoint": str(cp), "regions": pr,
                           "before": m["metrics"]["before_region_count"],
                           "after": m["metrics"]["after_region_count"]}
            print(f"    done {time.time()-t0:.0f}s: {m['metrics']['before_region_count']}->{m['metrics']['after_region_count']}")
    
    return trained

def apply_probe_refiners_region_routed(
    *,
    vocals_path: Path,
    instrumental_path: Path,
    raw_path: Path,
    trained_refiners: dict,
    all_regions: list,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_region_repair_training_device("mps")
    
    vocals, sr = read_audio_stereo(vocals_path)
    instrumental, sr2 = read_audio_stereo(instrumental_path)
    raw, sr3 = read_audio_stereo(raw_path)
    assert sr == sr2 == sr3
    
    total_frames = min(vocals.shape[0], instrumental.shape[0], raw.shape[0])
    
    # Group regions by probe, build per-probe masks
    probe_masks = {}
    for ps, info in trained_refiners.items():
        pv = float(ps)
        mask = np.zeros(total_frames, dtype=np.float32)
        pad = int(round(0.05 * sr))
        for r in info["regions"]:
            s = max(0, int(round(r.get("start",0)*sr)) - pad)
            e = min(total_frames, int(round(r.get("end",0)*sr)) + pad)
            mask[s:e] = 1.0
        probe_masks[ps] = mask
    
    # Load models
    models = {}
    for ps, info in trained_refiners.items():
        cp = Path(str(info["checkpoint"]))
        if cp.exists():
            m, _ = load_region_repair_checkpoint(cp)
            models[ps] = m
    
    # Apply each model to its mask regions
    stitched_instr = instrumental[:total_frames].copy()
    stitched_voc = vocals[:total_frames].copy()
    applied = {}
    
    for ps, model in models.items():
        mask = probe_masks[ps]
        if np.sum(mask) < sr:  # less than 1 sec of active frames
            continue
        print(f"  Applying probe {ps} model ({int(np.sum(mask))} frames)...")
        repaired, diag = apply_region_repair_model(
            model=model, mix_audio=raw, bad_stem_audio=instrumental,
            source_condition_audio=vocals, region_mask=mask,
            sample_rate=sr, issue_type=ISSUE_VOCAL_LEAK, device=device,
        )
        delta = (repaired - instrumental)[:total_frames]
        # Apply only where THIS probe's mask is active and no other probe was applied
        mask_2d = mask[:total_frames, None]
        stitched_instr = np.where(mask_2d > 0.5, np.clip(stitched_instr + delta - (instrumental[:total_frames] - stitched_instr), -1.0, 1.0), stitched_instr)
        # Actually, simpler: just add delta where mask is active
        applied_in_this_probe = np.where(mask_2d > 0.5)[0]
        if len(applied_in_this_probe) > 0:
            stitched_instr[applied_in_this_probe] = np.clip(instrumental[:total_frames][applied_in_this_probe] + delta[applied_in_this_probe], -1.0, 1.0)
            stitched_voc[applied_in_this_probe] = np.clip(vocals[:total_frames][applied_in_this_probe] - delta[applied_in_this_probe], -1.0, 1.0)
        applied[ps] = int(np.sum(mask > 0.5))
    
    # Write output
    ad = output_dir / "audio"
    write_audio(ad / "v.flac", stitched_voc, sr)
    write_audio(ad / "i.flac", stitched_instr, sr)
    
    # Score
    fs = score_pitch_shifted_vocal_leak(
        vocals_path=ad / "v.flac", instrumental_path=ad / "i.flac",
        output_dir=output_dir / "score_final",
    )
    
    result = {
        "original_regions": len(all_regions),
        "final_regions": fs.get("total_fail_regions"),
        "final_verdict": fs.get("verdict"),
        "applied_probes": applied,
        "trained_probes": list(trained_refiners.keys()),
    }
    write_json(output_dir / "result.json", result)
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    args = p.parse_args()
    
    base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics")
    # First: train per-probe refiners
    trained = train_per_probe_refiners(
        track_alias=args.track,
        vocals_path=base / f"pitch_guard/{args.track}/audio/v.flac",
        instrumental_path=base / f"pitch_guard/{args.track}/audio/i.flac",
        raw_path=base / f"pitch_guard/{args.track}/audio/raw.flac",
        base_output=base / "per_probe",
        epochs=400,
    )
    if trained:
        # Get all regions from original score
        orig_score = score_pitch_shifted_vocal_leak(
            vocals_path=base / f"pitch_guard/{args.track}/audio/v.flac",
            instrumental_path=base / f"pitch_guard/{args.track}/audio/i.flac",
            output_dir=base / "per_probe" / args.track / "score_ref",
        )
        result = apply_probe_refiners_region_routed(
            vocals_path=base / f"pitch_guard/{args.track}/audio/v.flac",
            instrumental_path=base / f"pitch_guard/{args.track}/audio/i.flac",
            raw_path=base / f"pitch_guard/{args.track}/audio/raw.flac",
            trained_refiners=trained,
            all_regions=orig_score.get("regions", []),
            output_dir=base / "per_probe" / args.track / "applied",
        )
        print(json.dumps(result, indent=2))
