"""Show results for attack_v2 training."""
from pathlib import Path
import json

for label in [
    "nen-hot-tiktok-douyin-2025_v2",
    "nen-hot-tiktok-douyin-2025_v3_damage_aware",
    "nen-hot-tiktok-douyin-2025_v4_attack",
]:
    path = Path(f"/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/training/{label}/manifest.json")
    if not path.exists():
        print(f"\n=== {label} === NO_MANIFEST")
        continue
    m = json.loads(path.read_text())
    mm = m.get("metrics", {})
    ad = mm.get("apply_diagnostics", {})
    print(f"\n=== {label} ===")
    print(f"  profile: {m.get('objective_profile', '?')}")
    print(f"  status: {m.get('status')}")
    print(f"  improved: {mm.get('target_issue_improved')}")
    print(f"  regions: {mm.get('before_region_count')} -> {mm.get('after_region_count')}")
    print(f"  score: {mm.get('before_best_score',0):.4f} -> {mm.get('after_best_score',0):.4f}")
    print(f"  delta_rms: {mm.get('instrumental_delta_rms',0):.6f}")
    print(f"  changed_ratio: {ad.get('changed_region_ratio',0):.6f}")
    print(f"  next_action: {m.get('next_training_action')}")
