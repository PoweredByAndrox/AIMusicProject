"""Collect pitch-guard results summary for all 5 tracks."""
from pathlib import Path
import json

tracks = [
    "armin-sound-of-you",
    "gradusi_final",
    "leps-ona-ne-tvoya",
    "nen-hot-tiktok-douyin-2025",
    "titus-let-it-rip-chipcat",
]
base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard")
summary = {}
for t in tracks:
    mf = base / t / "manifest.json"
    if not mf.exists():
        summary[t] = {"error": "no_manifest"}
        continue
    m = json.loads(mf.read_text())
    s = m["score"]
    a = m.get("after_repair_score") or {}
    summary[t] = {
        "verdict": s["verdict"],
        "blocker_reasons": s["blocker_reasons"],
        "all_region_count": s["all_region_count"],
        "total_fail_regions": s["total_fail_regions"],
        "best_probe": s["best_probe"],
        "failure_taxonomy": s["failure_taxonomy"],
        "scanned_windows": s["scanned_windows"],
        "repair_applied": m.get("repair") is not None,
    }
    if a:
        summary[t]["after_verdict"] = a.get("verdict", None)
        summary[t]["after_total_fail"] = a.get("total_fail_regions", None)
        summary[t]["after_failure_taxonomy"] = a.get("failure_taxonomy", None)
print(json.dumps(summary, indent=2))
