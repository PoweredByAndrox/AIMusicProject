cat > /tmp/run_inpaint_pg.py << 'PYEOF'
import sys, json, time, subprocess, os
from pathlib import Path

sys.path.insert(0, "/tmp/mac_tools/tools")
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak

TRACKS = ["armin-sound-of-you", "gradusi_final", "leps-ona-ne-tvoya",
          "nen-hot-tiktok-douyin-2025", "titus-let-it-rip-chipcat"]
CONFIGS = ["0.3_0.6", "0.3_0.5", "0.4_0.7"]

for t in TRACKS:
    for pair in CONFIGS:
        instr = Path(f"/tmp/mac_quick_inpaint/{t}/{pair}/m001.flac")
        vocals = Path(f"/tmp/mac_quick_inputs/{t}/vocals.flac")
        outdir = Path(f"/tmp/mac_quick_inpaint/{t}/{pair}/pitch_guard")
        if not instr.exists():
            print(f"SKIP {t} {pair}: no instrument")
            continue
        t0 = time.time()
        pg = score_pitch_shifted_vocal_leak(
            vocals_path=vocals,
            instrumental_path=instr,
            output_dir=outdir,
        )
        rc = pg.get("total_fail_regions", len(pg.get("regions", [])))
        print(json.dumps({"track": t, "config": pair, "regions": rc, "elapsed_s": round(time.time()-t0, 1)}))
print("ALL DONE")
PYEOF
PYTHONPATH=/tmp/mac_tools/tools /Users/andrejviluha/aimusic_mps_venv/bin/python /tmp/run_inpaint_pg.py 2>&1
