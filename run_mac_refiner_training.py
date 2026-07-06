"""Run mac_refiner training on a specific track."""
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
import json

track = "nen-hot-tiktok-douyin-2025"
mel_guard_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/{track}")
output_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{track}")

print(f"Training refiner on {track}...")
manifest = run_training(
    mel_guard_dir=mel_guard_dir,
    output_dir=output_dir,
    target_gain=2.5,
    epochs=160,
    require_mps=True,
    max_regions=8,
    objective_profile="projection_teacher",
)
print(json.dumps({
    "status": manifest.get("status"),
    "resolved_device": manifest.get("resolved_device"),
    "target_issue_improved": manifest.get("metrics", {}).get("target_issue_improved"),
    "before_region_count": manifest.get("metrics", {}).get("before_region_count"),
    "after_region_count": manifest.get("metrics", {}).get("after_region_count"),
}, indent=2))
