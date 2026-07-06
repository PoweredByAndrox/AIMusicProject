"""Create final listen pack from best per-track training."""
from pathlib import Path
import json, shutil

base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics")
lp = base / "listen_pack_final"
lp.mkdir(parents=True, exist_ok=True)

tracks = {
    "armin-sound-of-you": ("armin_best", 311, 293),
    "gradusi_final": ("gradusi_best", 116, 93),
    "leps-ona-ne-tvoya": ("leps_best", 118, 103),
    "nen-hot-tiktok-douyin-2025": ("nhot_best", 240, 212),
    "titus-let-it-rip-chipcat": ("titus_best", 105, 90),
}

entries = []
for track, (alias, before, after) in tracks.items():
    tdir = base / "training" / track
    mf = tdir / "manifest.json"
    if not mf.exists():
        entries.append({"track": track, "error": "no_training"})
        continue
    m = json.loads(mf.read_text())
    audio = m.get("audio", {})
    
    for stem, suffix in [("vocals", "v"), ("instrumental", "i")]:
        src = audio.get(stem, "")
        if src and Path(str(src)).exists():
            shutil.copy2(str(src), str(lp / f"{alias}_{suffix}.flac"))
    
    entries.append({
        "label": alias,
        "regions_before": before,
        "regions_after": after,
        "reduction_pct": round((1-after/before)*100, 1),
    })

# Also copy original mac_quick raw_ref and instrumental for comparison
macq = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick")
for track, (alias, _, _) in tracks.items():
    src = macq / track / "instrumental.flac"
    if src.exists():
        shutil.copy2(str(src), str(lp / f"{alias}_original_i.flac"))

(lp / "manifest.json").write_text(json.dumps({
    "schema": "listen_pack_final",
    "entries": entries,
}, indent=2))
print("Created:", lp)
for e in entries:
    print(f"  {e['label']}: {e['regions_before']} -> {e['regions_after']} ({e['reduction_pct']}%)")
