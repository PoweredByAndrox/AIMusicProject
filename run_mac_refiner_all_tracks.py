"""Run mac_refiner training on remaining 4 tracks with best config."""
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
import json, time

tracks = [
    "armin-sound-of-you",
    "gradusi_final",
    "leps-ona-ne-tvoya",
    "titus-let-it-rip-chipcat",
]
results = []
for track in tracks:
    mrd = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/{track}")
    od = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{track}")
    print(f"\n[{time.strftime('%H:%M:%S')}] Training {track}...")
    t0 = time.time()
    manifest = run_training(
        mel_guard_dir=mrd,
        output_dir=od,
        target_gain=3.0,
        epochs=800,
        require_mps=True,
        max_regions=32,
        objective_profile="projection_teacher",
    )
    elapsed = time.time() - t0
    mm = manifest.get("metrics", {})
    ad = mm.get("apply_diagnostics", {})
    results.append({
        "track": track,
        "elapsed_sec": round(elapsed, 1),
        "status": manifest.get("status"),
        "before_regions": mm.get("before_region_count"),
        "after_regions": mm.get("after_region_count"),
        "region_reduction_pct": round((1 - mm.get("after_region_count",0)/max(mm.get("before_region_count",1),1))*100, 1),
        "before_score": round(mm.get("before_best_score",0), 4),
        "after_score": round(mm.get("after_best_score",0), 4),
        "delta_rms": round(mm.get("instrumental_delta_rms",0), 6),
        "changed_ratio": round(ad.get("changed_region_ratio",0), 4),
        "next_action": manifest.get("next_training_action"),
    })
    print(f"  {track}: {results[-1]['before_regions']}->{results[-1]['after_regions']} ({results[-1]['region_reduction_pct']}%) in {elapsed:.0f}s")

print("\n\n=== SUMMARY ===")
print(json.dumps(results, indent=2))
