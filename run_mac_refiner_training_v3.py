"""Run mac_refiner training with damage_aware_v2 profile."""
from pathlib import Path
from tools.train_mac_refiner_from_mel_guard import run_training
import json

track = "nen-hot-tiktok-douyin-2025"
mel_guard_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/{track}")

for label, profile in [("v3_damage_aware", "pitch_region_detector_damage_aware_v2"),
                        ("v4_attack", "pitch_region_count_attack_v2")]:
    output_dir = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{track}_{label}")
    print(f"\n=== Training {profile} ===")
    manifest = run_training(
        mel_guard_dir=mel_guard_dir,
        output_dir=output_dir,
        target_gain=2.5,
        epochs=400,
        require_mps=True,
        max_regions=16,
        objective_profile=profile,
    )
    m = manifest.get("metrics", {})
    ad = m.get("apply_diagnostics", {})
    print(json.dumps({
        "profile": profile,
        "status": manifest.get("status"),
        "target_issue_improved": m.get("target_issue_improved"),
        "region_count_before_after": [m.get("before_region_count"), m.get("after_region_count")],
        "best_score_before_after": [m.get("before_best_score"), m.get("after_best_score")],
        "instrumental_delta_rms": m.get("instrumental_delta_rms"),
        "changed_region_ratio": ad.get("changed_region_ratio"),
        "next_action": manifest.get("next_training_action"),
    }, indent=2))
