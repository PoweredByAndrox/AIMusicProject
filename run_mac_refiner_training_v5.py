"""Run mac_refiner training with more epochs and regions."""
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
import json

track = "nen-hot-tiktok-douyin-2025"
mel_guard_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/{track}")

manifest = run_training(
    mel_guard_dir=mel_guard_dir,
    output_dir=Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{track}_v5_heavy"),
    target_gain=3.0,
    epochs=800,
    require_mps=True,
    max_regions=32,
    objective_profile="projection_teacher",
)
m = manifest.get("metrics", {})
ad = m.get("apply_diagnostics", {})
print(json.dumps({
    "profile": "projection_teacher_heavy",
    "status": manifest.get("status"),
    "improved": m.get("target_issue_improved"),
    "regions": [m.get("before_region_count"), m.get("after_region_count")],
    "score": [m.get("before_best_score"), m.get("after_best_score")],
    "delta_rms": m.get("instrumental_delta_rms"),
    "changed_ratio": ad.get("changed_region_ratio"),
    "recon_rms": m.get("reconstruction_rms"),
    "next": manifest.get("next_training_action"),
}, indent=2))
