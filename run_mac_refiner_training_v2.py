"""Run mac_refiner training with more epochs on a specific track."""
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
import json

track = "nen-hot-tiktok-douyin-2025"
mel_guard_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/{track}")
output_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{track}_v2")

print(f"Training refiner on {track} (v2: epochs=400, max_regions=16)...")
manifest = run_training(
    mel_guard_dir=mel_guard_dir,
    output_dir=output_dir,
    target_gain=2.5,
    epochs=400,
    require_mps=True,
    max_regions=16,
    objective_profile="projection_teacher",
)
print(json.dumps({
    "status": manifest.get("status"),
    "resolved_device": manifest.get("resolved_device"),
    "target_issue_improved": manifest.get("metrics", {}).get("target_issue_improved"),
    "before_region_count": manifest.get("metrics", {}).get("before_region_count"),
    "after_region_count": manifest.get("metrics", {}).get("after_region_count"),
    "before_best_score": manifest.get("metrics", {}).get("before_best_score"),
    "after_best_score": manifest.get("metrics", {}).get("after_best_score"),
    "instrumental_delta_rms": manifest.get("metrics", {}).get("instrumental_delta_rms"),
    "changed_region_ratio": manifest.get("metrics", {}).get("apply_diagnostics", {}).get("changed_region_ratio"),
    "next_training_action": manifest.get("next_training_action"),
}, indent=2))
