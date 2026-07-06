"""Train ONE universal refiner on all regions from ALL 5 tracks combined."""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, time

from tools.mac_melband_pitch_guard import (
    score_pitch_shifted_vocal_leak, pitch_shift_resample_same_length,
    score_region_count, score_best_value,
)
from tools.stem_region_repair_common import read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import (
    StemRegionRepairTrainConfig, apply_region_repair_model,
    resolve_region_repair_training_device, save_region_repair_checkpoint,
    train_region_repair_model_from_arrays, write_training_history,
)

ISSUE_VOCAL_LEAK = "vocal_leak"

train_config = StemRegionRepairTrainConfig(
    hidden_channels=32, kernel_size=31, epochs=800,
    learning_rate=0.003, region_loss_weight=1.0, outside_loss_weight=0.45,
    mix_conservation_loss_weight=0.5,
    forensic_source_conditioning=True,
    forensic_source_decorrelation_loss_weight=0.2,
    forensic_source_envelope_loss_weight=0.08, max_delta=0.8,
    device="mps",
)

tracks = {
    "armin-sound-of-you": {"label": "armin"},
    "gradusi_final": {"label": "gradusi"},
    "leps-ona-ne-tvoya": {"label": "leps"},
    "nen-hot-tiktok-douyin-2025": {"label": "nhot"},
    "titus-let-it-rip-chipcat": {"label": "titus"},
}

base = Path("/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick_metrics")
out = base / "universal_refiner"
out.mkdir(parents=True, exist_ok=True)

all_raw, all_bad, all_target, all_src, all_mask = [], [], [], [], []
sample_rate = None
total_regions = 0

for track in tracks:
    pg = base / "pitch_guard" / track
    m = json.loads((pg / "manifest.json").read_text())
    score = m.get("score", {})
    regions = list(score.get("regions", []) or [])
    if not regions:
        print(f"SKIP {track}: no regions")
        continue
    
    vocals, sr = read_audio_stereo(pg / "audio" / "v.flac")
    instrumental, _ = read_audio_stereo(pg / "audio" / "i.flac")
    raw, _ = read_audio_stereo(pg / "audio" / "raw.flac")
    
    if sample_rate is None:
        sample_rate = sr
    if sr != sample_rate:
        print(f"SKIP {track}: sample rate mismatch")
        continue
    
    tf = min(vocals.shape[0], instrumental.shape[0], raw.shape[0])
    vocals = vocals[:tf]; instrumental = instrumental[:tf]; raw = raw[:tf]
    
    # Process each region individually
    ctx = int(round(0.05 * sr))
    track_raw, track_bad, track_target, track_src, track_mask = [], [], [], [], []
    
    for r in regions[:16]:  # cap per track
        s = max(0, int(round(r.get("start",0)*sr)) - ctx)
        e = min(tf, int(round(r.get("end",0)*sr)) + ctx)
        if e - s < sr // 4:
            continue
        
        crop_v = vocals[s:e]; crop_i = instrumental[s:e]; crop_r = raw[s:e]
        crop_mask = np.ones(e - s, dtype=np.float32)
        
        # Source = vocals at dominant probe for this region
        probe = float(r.get("probe_semitones", r.get("semitones", 0)))
        source = pitch_shift_resample_same_length(crop_v, probe)
        
        coeff = float(np.sum((crop_i * source).astype(np.float64)) / (np.sum(np.square(source.astype(np.float64))) + 1e-10))
        coeff = max(-2.5, min(2.5, coeff * 2.5))
        pseudo = np.clip(crop_i - coeff * source * crop_mask[:, None], -1.0, 1.0)
        
        track_raw.append(crop_r); track_bad.append(crop_i)
        track_target.append(pseudo); track_src.append(source)
        track_mask.append(crop_mask)
    
    if not track_raw:
        continue
    
    all_raw.append(np.concatenate(track_raw, axis=0))
    all_bad.append(np.concatenate(track_bad, axis=0))
    all_target.append(np.concatenate(track_target, axis=0))
    all_src.append(np.concatenate(track_src, axis=0))
    all_mask.append(np.concatenate(track_mask, axis=0))
    total_regions += len(track_raw)
    print(f"{track}: {len(track_raw)} regions, {sum(m.shape[0] for m in track_raw)/sr:.1f}s")

if not all_raw:
    print("No training data!"); exit()

train_raw = np.concatenate(all_raw, axis=0).astype(np.float32)
train_bad = np.concatenate(all_bad, axis=0).astype(np.float32)
train_target = np.concatenate(all_target, axis=0).astype(np.float32)
train_src = np.concatenate(all_src, axis=0).astype(np.float32)
train_mask = np.concatenate(all_mask, axis=0).astype(np.float32)

print(f"\nTotal: {total_regions} regions, {train_raw.shape[0]/sample_rate:.1f}s training audio ({train_raw.shape[0]} frames)")

# Train
model, history = train_region_repair_model_from_arrays(
    mix_audio=train_raw, bad_stem_audio=train_bad,
    target_stem_audio=train_target,
    forensic_source_audio=train_src, region_mask=train_mask,
    region_constraint_masks=[],
    sample_rate=sample_rate, issue_type=ISSUE_VOCAL_LEAK,
    config=train_config,
)
cp = save_region_repair_checkpoint(model=model, path=out / "model.pt", training_history=history)
write_training_history(out / "history.json", history)

# Apply to each track
device = resolve_region_repair_training_device("mps")
results = {}
for track in tracks:
    pg = base / "pitch_guard" / track
    vocals, sr = read_audio_stereo(pg / "audio" / "v.flac")
    instrumental, _ = read_audio_stereo(pg / "audio" / "i.flac")
    raw, _ = read_audio_stereo(pg / "audio" / "raw.flac")
    tf = min(vocals.shape[0], instrumental.shape[0], raw.shape[0])
    
    m2 = json.loads((pg / "manifest.json").read_text())
    score2 = m2.get("score", {})
    regions = score2.get("regions", [])
    
    # Build mask from original regions
    mask = np.zeros(tf, dtype=np.float32)
    ctx = int(round(0.05 * sr))
    for r in regions:
        s = max(0, int(round(r.get("start",0)*sr)) - ctx)
        e = min(tf, int(round(r.get("end",0)*sr)) + ctx)
        mask[s:e] = 1.0
    
    probe = float(score2.get("best_probe", {}).get("semitones", 0))
    src = pitch_shift_resample_same_length(vocals[:tf], probe)
    
    repaired, _ = apply_region_repair_model(
        model=model, mix_audio=raw[:tf], bad_stem_audio=instrumental[:tf],
        source_condition_audio=src, region_mask=mask,
        sample_rate=sr, issue_type=ISSUE_VOCAL_LEAK, device=device,
    )
    delta = repaired - instrumental[:tf]
    ref_i = np.clip(instrumental[:tf] + delta, -1.0, 1.0)
    ref_v = np.clip(vocals[:tf] - delta, -1.0, 1.0)
    
    td = out / f"apply_{track}"
    write_audio(td / "v.flac", ref_v, sr)
    write_audio(td / "i.flac", ref_i, sr)
    
    fs = score_pitch_shifted_vocal_leak(
        vocals_path=td / "v.flac", instrumental_path=td / "i.flac",
        output_dir=td / "score",
    )
    orig_regions = score2.get("total_fail_regions", len(regions))
    final_regions = fs.get("total_fail_regions", len(fs.get("regions", [])))
    results[track] = {"original": orig_regions, "final": final_regions, "verdict": fs.get("verdict")}
    print(f"{track}: {orig_regions} -> {final_regions} ({fs.get('verdict')})")

write_json(out / "results.json", results)
print(f"\nCheckpoint: {cp}")
print(json.dumps(results, indent=2))
