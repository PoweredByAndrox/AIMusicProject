"""Run pitch-guard on all 5 mac_quick tracks."""
from tools.mac_melband_pitch_guard import run_pitch_guard_on_existing_melband_run
from pathlib import Path

tracks = [
    "armin-sound-of-you",
    "gradusi_final",
    "leps-ona-ne-tvoya",
    "nen-hot-tiktok-douyin-2025",
    "titus-let-it-rip-chipcat",
]

base_artifacts = "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick/"
base_output = "/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics/pitch_guard/"

for track in tracks:
    melband_run_dir = Path(base_artifacts + track + "/run")
    output_dir = Path(base_output + track)
    print("Processing " + track + "...")
    manifest = run_pitch_guard_on_existing_melband_run(
        melband_run_dir=melband_run_dir, output_dir=output_dir
    )
    verdict = manifest["score"]["verdict"]
    blockers = manifest["blocker_reasons"]
    print("  " + track + ": verdict=" + str(verdict) + ", blockers=" + str(blockers))
