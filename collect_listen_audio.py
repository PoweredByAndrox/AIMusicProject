"""Collect best repaired audio from training into listen pack."""
from pathlib import Path
import json, shutil

base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics")

listen_root = base / "listen_pack"
listen_root.mkdir(parents=True, exist_ok=True)

# Training results
trainings = [
    ("nen-hot-tiktok-douyin-2025_v5_heavy", "projection_teacher_heavy", 240, 212),
    ("armin-sound-of-you", "projection_teacher_heavy", 311, 293),
    ("gradusi_final", "projection_teacher_heavy", 116, 93),
    ("leps-ona-ne-tvoya", "projection_teacher_heavy", 118, 103),
    ("titus-let-it-rip-chipcat", "projection_teacher_heavy", 105, 90),
]

entries = []
for tname, profile, before, after in trainings:
    tdir = base / "training" / tname
    mf = tdir / "manifest.json"
    if not mf.exists():
        entries.append({"track": tname, "error": "no_manifest"})
        continue
    m = json.loads(mf.read_text())
    audio = m.get("audio", {})
    vpath = audio.get("vocals", "")
    ipath = audio.get("instrumental", "")
    # Copy to listen pack
    alias = tname.replace("-douyin-2025", "").replace("-chipcat", "").replace("-sound-of-you", "").replace("-final", "").replace("-ne-tvoya", "").replace("-it-rip", "")
    if len(alias) > 20:
        alias = alias[:20]
    vdest = listen_root / f"{alias}_v.flac"
    idest = listen_root / f"{alias}_i.flac"
    if vpath and Path(vpath).exists():
        shutil.copy2(vpath, vdest)
    if ipath and Path(ipath).exists():
        shutil.copy2(ipath, idest)
    entries.append({
        "track": tname,
        "alias": alias,
        "profile": profile,
        "regions_before": before,
        "regions_after": after,
        "reduction_pct": round((1 - after/max(before,1))*100, 1),
        "vocals": str(vdest) if vdest.exists() else None,
        "instrumental": str(idest) if idest.exists() else None,
    })

manifest = {
    "schema": "mac_quick_listen_pack.v1",
    "delivery_allowed": False,
    "presentation_status": "user_evaluation_only",
    "entries": entries,
}
(listen_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(entries, indent=2))
