import sys, json
from pathlib import Path
sys.path.insert(0, "/tmp/mac_tools/tools")
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak

r = score_pitch_shifted_vocal_leak(
    vocals_path=Path("/tmp/mac_quick_inputs/armin-sound-of-you/vocals.flac"),
    instrumental_path=Path("/tmp/mac_quick_inputs/armin-sound-of-you/instrumental.flac"),
    output_dir=Path("/tmp/mac_quick_inputs/armin-sound-of-you/pg_ground"),
)
json.dump(r, open("/tmp/ground_truth_vgw.json", "w"), indent=2, default=str)
print("Ground truth:", r.get("total_fail_regions", r.get("selected_region_count", 0)))
