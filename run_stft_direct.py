"""Direct STFT-domain gain mask: attenuate vocal-dominant bins, keep minimum gain floor."""
from __future__ import annotations
import sys, json, shutil, time
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, "/Users/andrejviluha/AIMusicProject_win/ml_worker_improved")
from tools.mac_melband_pitch_guard import score_pitch_shifted_vocal_leak
from tools.stem_region_repair_common import read_audio_stereo, write_audio

PROJECT = Path("/Users/andrejviluha/AIMusicProject_win")

N_FFT = 4096
HOP = 1024
EPS = 1e-12

def build_gain_mask(instr_mono, vocal_mono, sr, strength=0.25, voice_band=(160, 5200),
                    low_band_max=140, min_gain_voice=0.78, min_gain_presence=0.86, min_gain_low=0.995):
    """Build per-bin gain mask: attenuate where vocals dominate."""
    freqs, _, instr_stft = signal.stft(instr_mono, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
    _, _, vocal_stft = signal.stft(vocal_mono, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")

    instr_mag = np.abs(instr_stft) + EPS
    vocal_mag = np.abs(vocal_stft) + EPS

    # Residue ratio: how much vocals dominate over instrumental at each bin
    residue_ratio = vocal_mag / (instr_mag + vocal_mag)  # 0-1, close to 1 = vocal dominates
    mean_residue = float(np.mean(residue_ratio[voice_band_mask(freqs, voice_band), :]))

    # Gain: 1.0 where no vocal, reduced where vocal dominates
    gain = 1.0 - strength * residue_ratio

    # Frequency band weights
    vb_mask = (freqs >= voice_band[0]) & (freqs <= voice_band[1])
    pb_mask = (freqs >= 1800) & (freqs <= 4500)
    lb_mask = freqs < low_band_max

    # Apply minimum gain per band
    min_gain = np.ones_like(freqs)
    min_gain[vb_mask] = min_gain_voice
    min_gain[pb_mask] = min_gain_presence
    min_gain[lb_mask] = min_gain_low
    min_gain = min_gain[:, None]

    gain = np.maximum(gain, min_gain)
    return gain, {"mean_residue_ratio": round(mean_residue, 6)}

def voice_band_mask(freqs, band):
    return (freqs >= band[0]) & (freqs <= band[1])

tracks = [
    ("armin-sound-of-you", "armin"),
    ("gradusi_final", "gradusi"),
    ("leps-ona-ne-tvoya", "leps"),
    ("nen-hot-tiktok-douyin-2025", "nhot"),
    ("titus-let-it-rip-chipcat", "titus"),
]

strengths = [0.15, 0.25, 0.35]

for track_id, alias in tracks:
    macq = PROJECT / "artifacts/mac_quick" / track_id
    instr_path = macq / "instrumental.flac"
    vocals_path = macq / "vocals.flac"
    if not instr_path.exists() or not vocals_path.exists():
        print(f"SKIP {alias}")
        continue

    instr_audio, sr = read_audio_stereo(instr_path)
    vocals_audio, sr2 = read_audio_stereo(vocals_path)
    assert sr == sr2
    n = min(instr_audio.shape[0], vocals_audio.shape[0])
    instr_audio = instr_audio[:n]
    vocals_audio = vocals_audio[:n]

    for s in strengths:
        out_dir = PROJECT / "artifacts/mac_quick_metrics/stft_direct" / alias / f"s{s}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        refined_channels = []
        all_metrics = []
        for ch in range(instr_audio.shape[1]):
            gain, metrics = build_gain_mask(
                instr_audio[:, ch], vocals_audio[:, min(ch, vocals_audio.shape[1] - 1)],
                sr, strength=s,
            )
            all_metrics.append(metrics)
            _, _, ch_stft = signal.stft(instr_audio[:, ch], fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
            _, refined_ch = signal.istft(ch_stft * gain, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, input_onesided=True)
            refined_channels.append(refined_ch[:instr_audio.shape[0]].astype(np.float32))

        refined = np.column_stack(refined_channels).astype(np.float32)
        elapsed = time.time() - t0
        write_audio(out_dir / "m001.flac", refined, sr)

        pg_dir = out_dir / "pitch_guard"
        pg_dir.mkdir(parents=True)
        pg = score_pitch_shifted_vocal_leak(
            vocals_path=vocals_path,
            instrumental_path=out_dir / "m001.flac",
            output_dir=pg_dir,
        )
        r_count = pg.get("total_fail_regions", len(pg.get("regions", [])))
        mean_res = all_metrics[0]["mean_residue_ratio"] if all_metrics else 0

        delta_rms = np.sqrt(np.mean((refined - instr_audio) ** 2))
        instr_rms = np.sqrt(np.mean(instr_audio ** 2))
        delta_db = 20 * np.log10(delta_rms / (instr_rms + EPS))

        (out_dir / "result.json").write_text(json.dumps({
            "strength": s, "regions": r_count, "mean_residue": float(mean_res), "delta_db": float(delta_db)
        }))
        print(f"{alias} s={s}: {r_count} regions | residue={mean_res:.3f} | delta={delta_db:.1f}dB | {elapsed:.0f}s")

print("\nDone")
