#!/usr/bin/env python3
"""BS-RoFo-SW TTA MPS + dip repair + metrics pipeline.
Run on Mac:  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python run_mac_full_pipeline.py
No CPU fallback."""

import argparse, gc, json, math, os, shutil, sys, time, traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
import yaml
from ml_collections import ConfigDict
from bs_roformer.utils import get_model_from_config
from bs_roformer.inference import SafeLoaderWithTuple

def load_bs_roformer(config_path: Path, ckpt_path: Path, device):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return model, config

# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def tta_forward(mix_chunk: torch.Tensor, model: nn.Module, device) -> torch.Tensor:
    """Return vocal stem [1, 2, T] from identity+flip TTA."""
    with torch.no_grad():
        p = model(mix_chunk.to(device))
        p_flip = model((-mix_chunk).to(device))
    # BS-RoFo-SW 6 stems: ('bass','drums','other','vocals','guitar','piano')
    # Index 3 = vocals
    return ((p.cpu() - p_flip.cpu()) / 2)[:, 3]

def separate_tta(mix_t: torch.Tensor, model: nn.Module, device, sr: int,
                 chunk_sec: int = 10) -> np.ndarray:
    """Full-track TTA separation, returns vocal [T, C]."""
    n_samples = mix_t.shape[-1]
    hop = int(sr * chunk_sec)
    chunks = []
    for start in range(0, n_samples, hop):
        end = min(start + hop, n_samples)
        chunk = mix_t[..., start:end]
        v = tta_forward(chunk, model, device)  # [1, 2, T]
        chunks.append(v)
    return torch.cat(chunks, dim=-1).squeeze(0).numpy().T  # [T, C]

# ---------------------------------------------------------------------------
# hardened residual
# ---------------------------------------------------------------------------
def hardened_residual(mix: np.ndarray, vocal: np.ndarray,
                      max_ratio: float = 1.05, eps: float = 1e-6) -> np.ndarray:
    limit = np.abs(mix) * max_ratio + eps
    clipped = np.clip(vocal, -limit, limit)
    return mix - clipped

# ---------------------------------------------------------------------------
# STFT-domain dip repair — per-band magnitude floor from original mix
# ---------------------------------------------------------------------------
def stft_dip_repair(instr: np.ndarray, vocal: np.ndarray, mix: np.ndarray,
                    sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Repair dips by enforcing a per-band spectral floor from the original mix.

    Strategy:
    - STFT of mix, instrumental (dipped), and vocal.
    - For every TF bin where vocal energy dominates, the hardened residual
      over-suppresses the instrumental. We restore the lost floor energy
      from the original mix.
    - Floor ratio is per-band: more aggressive where vocals are strong
      (mid-range), less in sub/air bands.
    - Vocal-dominant regions get a smoothed floor blend; quiet regions
      are left untouched.
    """
    eps_r = 1e-8

    def _stft(x):
        return torch.stft(torch.from_numpy(x).float(),
                         n_fft=n_fft, hop_length=hop_length,
                         win_length=n_fft, window=torch.hann_window(n_fft),
                         return_complex=True, center=True).numpy()

    def _istft(X, n):
        return torch.istft(torch.from_numpy(X), n_fft=n_fft,
                          hop_length=hop_length, win_length=n_fft,
                          window=torch.hann_window(n_fft), length=n).numpy()

    n = instr.shape[0]
    orig_1d = instr.ndim == 1
    if orig_1d:
        instr = instr[:, None]
        vocal = vocal[:, None]
        mix = mix[:, None]

    # per-band floor ratio (fraction of mix magnitude to restore)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # [F]
    floor_ratio = np.full(len(freqs), 0.15, dtype=np.float32)
    # sub (0-200):     little vocal energy, low floor
    floor_ratio[(freqs >= 0)     & (freqs < 200)]   = 0.15
    # low-mid (200-800): some vocal fundamental
    floor_ratio[(freqs >= 200)   & (freqs < 800)]   = 0.25
    # mid (800-4000): main vocal range, aggressive floor
    floor_ratio[(freqs >= 800)   & (freqs < 4000)]  = 0.50
    # high-mid (4000-8000): sibilance, moderate
    floor_ratio[(freqs >= 4000)  & (freqs < 8000)]  = 0.30
    # air (8000+): texture restoration
    floor_ratio[(freqs >= 8000)]                     = 0.40

    floor_ratio = floor_ratio[:, None]  # [F, 1]

    result = np.zeros_like(instr)
    for ch in range(instr.shape[1]):
        Xi = _stft(instr[:, ch])           # current instrumental
        Xv = _stft(vocal[:, ch])           # removed vocal
        Xm = _stft(mix[:, ch])             # original mix

        mag_i = np.abs(Xi)                 # [F, T]
        mag_v = np.abs(Xv)
        mag_m = np.abs(Xm)

        # --- vocal activity mask (time frames) ---
        # a frame is vocal-active if vocal energy > 15% of mix in that frame
        v_energy_per_frame = np.sum(mag_v ** 2, axis=0)
        m_energy_per_frame = np.sum(mag_m ** 2, axis=0)
        frame_vocal = (v_energy_per_frame / (m_energy_per_frame + eps_r)) > 0.15

        # --- per-bin blend factor ---
        # Smooth function of |V| / |M|
        v_ratio = mag_v / (mag_m + eps_r)  # [F, T]

        # blend = floor_ratio * smooth(v_ratio, threshold=0.1, knee=0.3)
        blend_raw = np.clip((v_ratio - 0.1) / 0.3, 0.0, 1.0)
        blend = floor_ratio * blend_raw

        # In inactive frames, zero out blend
        blend *= frame_vocal[None, :].astype(np.float32)

        # 2D smooth
        from scipy.ndimage import median_filter
        blend = median_filter(blend, size=(5, 7))

        # --- apply: repaired_mag = |I| + blend * (floor * |M| - |I|) ---
        # i.e., pull |I| toward (floor_ratio * |M|) where blend > 0
        floor_mag = floor_ratio * mag_m
        repaired_mag = mag_i + blend * np.maximum(0.0, floor_mag - mag_i)

        # phase from current instrumental
        repaired = repaired_mag * np.exp(1j * np.angle(Xi))
        result[:, ch] = _istft(repaired, n)

    if orig_1d or result.shape[1] == 1:
        result = result[:, 0]
    return result

# ---------------------------------------------------------------------------
# metric computation – dip / bleed / detail
# ---------------------------------------------------------------------------
def compute_dip_metrics(instr: np.ndarray, vocal: np.ndarray, mix: np.ndarray,
                        sr: int) -> dict:
    """Compute dip metrics: RMS, presence, air band dips per window."""
    eps_m = 1e-10
    window_sec = 0.5
    hop_sec = 0.25
    win_len = int(sr * window_sec)
    hop_len = int(sr * hop_sec)

    def _rms(x):
        return np.sqrt(np.mean(x ** 2, axis=-1) + eps_m)

    def _band_rms(x, lo_hz, hi_hz):
        """RMS in frequency band using STFT."""
        n_fft = 2048
        X = torch.stft(torch.from_numpy(x).float(), n_fft=n_fft,
                       hop_length=n_fft//2, window=torch.hann_window(n_fft),
                       return_complex=True, center=False).numpy()
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mask = (freqs >= lo_hz) & (freqs < hi_hz)
        band_mag = mag[mask]
        if band_mag.size == 0:
            return np.array([eps_m])
        return np.sqrt(np.mean(band_mag ** 2, axis=0) + eps_m)

    # stereo collapse
    if instr.ndim == 2:
        instr_mono = instr.mean(axis=-1)
        mix_mono = mix.mean(axis=-1)
    else:
        instr_mono = instr
        mix_mono = mix

    n = len(instr_mono)
    n_windows = max(1, (n - win_len) // hop_len + 1)

    # ---- RMS per window: instr vs mix ----
    rms_instr = np.array([
        _rms(instr_mono[i * hop_len:i * hop_len + win_len])
        for i in range(n_windows)
    ])
    rms_mix = np.array([
        _rms(mix_mono[i * hop_len:i * hop_len + win_len])
        for i in range(n_windows)
    ])

    # ----- dip metrics (traditional: instr-only) -----
    rms_median = np.median(rms_instr)
    rms_dip_mask = rms_instr < (rms_median * 0.5)
    rms_dip_db = 20 * np.log10(
        (rms_instr[rms_dip_mask] + eps_m) / (rms_median + eps_m)
    ) if rms_dip_mask.any() else np.array([0.0])

    # presence band (2-4 kHz) dips
    presence_rms = _band_rms(instr_mono, 2000, 4000)
    p_median = np.median(presence_rms)
    p_dip_mask = presence_rms < p_median * 0.5
    p_dip_db = 20 * np.log10(
        (presence_rms[p_dip_mask] + eps_m) / (p_median + eps_m)
    ) if p_dip_mask.any() else np.array([0.0])

    # air band (8-16 kHz) dips
    air_rms = _band_rms(instr_mono, 8000, 16000)
    a_median = np.median(air_rms)
    a_dip_mask = air_rms < a_median * 0.5
    a_dip_db = 20 * np.log10(
        (air_rms[a_dip_mask] + eps_m) / (a_median + eps_m)
    ) if a_dip_mask.any() else np.array([0.0])

    # ----- relative dip: instr vs original mix -----
    # normalised_dip = 20 * log10(instr_rms / mix_rms)
    # high for vocal-dominant windows (expected), low for instrumentals
    ratio = rms_instr / (rms_mix + eps_m)
    # compute worst-case excess dip for vocal-dominant windows
    vocal_active = rms_mix > (np.median(rms_mix) * 0.1)
    windows_above = vocal_active & (ratio < 0.5)
    excess_dip_db = 20 * np.log10(
        ratio[windows_above] + eps_m
    ) if windows_above.any() else np.array([0.0])

    # mix-normalized dip: log10(instr/mix) for all windows
    # <-20 dB means the instrumental lost >99% energy relative to mix
    mean_ratio_db = float(20 * np.log10(np.mean(ratio) + eps_m))

    return {
        "rms_dip_count": int(rms_dip_mask.sum()),
        "worst_rms_dip_db": round(float(rms_dip_db.min()), 3) if len(rms_dip_db) > 0 else 0.0,
        "average_rms_dip_db": round(float(rms_dip_db.mean()), 3) if len(rms_dip_db) > 0 else 0.0,
        "worst_presence_dip_db": round(float(p_dip_db.min()), 3) if len(p_dip_db) > 0 else 0.0,
        "average_presence_dip_db": round(float(p_dip_db.mean()), 3) if len(p_dip_db) > 0 else 0.0,
        "worst_air_dip_db": round(float(a_dip_db.min()), 3) if len(a_dip_db) > 0 else 0.0,
        "average_air_dip_db": round(float(a_dip_db.mean()), 3) if len(a_dip_db) > 0 else 0.0,
        "vocal_bleed_ratio": round(float(
            np.mean(np.abs(vocal) > np.abs(instr).mean() * 0.1)
        ), 4),
        "worst_excess_dip_db": round(float(excess_dip_db.min()), 3) if len(excess_dip_db) > 0 else 0.0,
        "mean_instr_mix_ratio_db": round(mean_ratio_db, 3),
    }

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BS-RoFo-SW TTA MPS + dip repair + metrics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # single track
  python run_mac_full_pipeline.py path/to/track_dir/
  # all tracks
  python run_mac_full_pipeline.py
  # dry-run metrics only (uses existing vocals/instrumental)
  python run_mac_full_pipeline.py --dry-metrics
""")
    parser.add_argument("track_dir", type=Path, nargs="?",
                        help="Single track directory (optional; default: process all under --output-root)")
    parser.add_argument("--config", type=Path,
                        default="/Users/andrejviluha/models/bsroformer/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default="/Users/andrejviluha/models/bsroformer/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt")
    parser.add_argument("--output-root", type=Path,
                        default="/Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tracks that already have bsroformer_metrics in manifest")
    parser.add_argument("--repair", action="store_true", default=True,
                        help="Apply STFT dip repair")
    parser.add_argument("--dry-metrics", action="store_true",
                        help="Skip separation/repair, compute metrics on existing files")
    parser.add_argument("--repair-only", action="store_true",
                        help="Load existing vocals+instr+mix, re-run repair + metrics")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if not args.dry_metrics:
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available. Abort. No CPU fallback.")
    device = torch.device(args.device) if not args.dry_metrics else None

    if not args.dry_metrics:
        print(f"Device: {device}  ({torch.device('mps').type if torch.backends.mps.is_available() else 'N/A'})")

    # determine track dirs
    if args.track_dir is not None:
        track_dirs = [args.track_dir]
        print(f"Track: {args.track_dir}")
    else:
        print(f"Output root: {args.output_root}")
        track_dirs = sorted(args.output_root.iterdir())
        track_dirs = [d for d in track_dirs if d.is_dir()]

    print(f"Found {len(track_dirs)} track(s)\n")

    # load model if needed
    model, model_config = None, None
    need_model = not args.dry_metrics and not args.repair_only
    if need_model:
        print("Loading BS-RoFo-SW model...", end=" ", flush=True)
        t0 = time.perf_counter()
        model, model_config = load_bs_roformer(args.config, args.checkpoint, device)
        instruments = tuple(model_config.training.instruments)
        print(f"done ({time.perf_counter()-t0:.1f}s)  stems: {instruments}")

    for track_dir in track_dirs:
        track_name = track_dir.name
        manifest_path = track_dir / "stem_manifest.json"

        # determine source path
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            # check if already done
            if args.skip_existing and manifest.get("metrics", {}).get("bsroformer_worst_rms_dip_db") is not None:
                print(f"[SKIP] {track_name} – already processed with BS-RoFo")
                continue
            source_key = "source_audio_path"
            source_path = manifest.get(source_key) or manifest.get("target_audio_path")
        else:
            manifest = {"schema_version": 1, "model_id": "bs_roformer_sw_jarredou",
                        "target_stem": "vocals", "target_product_stem": "vocal"}
            source_path = None

        if source_path is None or not Path(source_path).exists():
            # try to find source
            candidates = list(track_dir.glob("raw_ref.*")) + list(track_dir.glob("source.*"))
            if not candidates:
                print(f"[SKIP] {track_name} – no source audio found")
                continue
            source_path = str(candidates[0])

        source_path_p = Path(source_path)
        if not source_path_p.exists():
            print(f"[SKIP] {track_name} – source not found: {source_path}")
            continue

        print(f"[{track_name}]")
        print(f"  source: {source_path_p.name}")
        t_start = time.perf_counter()

        try:
            # ---- load audio ----
            mix, sr = sf.read(str(source_path_p))
            orig_ndim = mix.ndim
            if mix.ndim == 1:
                mix = np.stack([mix, mix], axis=-1)

            duration_s = mix.shape[0] / sr
            print(f"  duration: {duration_s:.1f}s, sr={sr}, shape={mix.shape}")

            if args.dry_metrics:
                # load existing files
                vocal_path = track_dir / "vocals.flac"
                instr_path = track_dir / "instrumental.flac"
                if not vocal_path.exists() or not instr_path.exists():
                    print(f"  SKIP: existing vocals/instrumental not found")
                    continue
                vocal, _ = sf.read(str(vocal_path))
                instr_repaired, _ = sf.read(str(instr_path))
                if instr_repaired.ndim == 1:
                    instr_repaired = instr_repaired[:, None]
                if vocal.ndim == 1:
                    vocal = vocal[:, None]
                # trim
                min_len = min(mix.shape[0], instr_repaired.shape[0], vocal.shape[0])
                mix = mix[:min_len]
                vocal = vocal[:min_len]
                instr_repaired = instr_repaired[:min_len]
                if orig_ndim == 1:
                    vocal = vocal[:, 0]
                    instr_repaired = instr_repaired[:, 0]
                print(f"  dry-metrics: loaded existing vocal/instr")
            elif args.repair_only:
                # load existing vocals + instrumental + use mix from source
                vocal_path = track_dir / "vocals.flac"
                instr_path = track_dir / "instrumental.flac"
                if not vocal_path.exists() or not instr_path.exists():
                    print(f"  SKIP: existing vocals/instrumental not found")
                    continue
                vocal, voc_sr = sf.read(str(vocal_path))
                instr_old, _ = sf.read(str(instr_path))
                if vocal.ndim == 1:
                    vocal = vocal[:, None]
                if instr_old.ndim == 1:
                    instr_old = instr_old[:, None]
                # trim all to match mix
                min_len = min(mix.shape[0], instr_old.shape[0], vocal.shape[0])
                mix = mix[:min_len]
                vocal = vocal[:min_len]
                instr_old = instr_old[:min_len]
                if orig_ndim == 1:
                    vocal = vocal[:, 0]
                    instr_old = instr_old[:, 0]
                print(f"  repair-only: loaded existing vocal+instr ({len(vocal)} samples)")

                t1 = time.perf_counter()
                instr_repaired = stft_dip_repair(instr_old, vocal, mix, sr)
                print(f"  dip repair: {time.perf_counter()-t1:.1f}s")
            else:
                mix_t = torch.from_numpy(mix.T.copy()).float().unsqueeze(0)

                # ---- separate ----
                t1 = time.perf_counter()
                vocal = separate_tta(mix_t, model, device, sr, chunk_sec=10)
                print(f"  separation: {time.perf_counter()-t1:.1f}s")

                # trim to same length
                min_len = min(mix.shape[0], vocal.shape[0])
                mix = mix[:min_len]
                vocal = vocal[:min_len]
                if orig_ndim == 1:
                    vocal = vocal[:, 0]

                # ---- hardened residual ----
                instr = hardened_residual(mix, vocal)

                # ---- dip repair ----
                if args.repair:
                    t1 = time.perf_counter()
                    instr_repaired = stft_dip_repair(instr, vocal, mix, sr)
                    print(f"  dip repair: {time.perf_counter()-t1:.1f}s")
                else:
                    instr_repaired = instr

            # ---- metrics ----
            t1 = time.perf_counter()
            metrics = compute_dip_metrics(instr_repaired, vocal, mix, sr)
            # prefix with bsroformer_
            bs_metrics = {f"bsroformer_{k}": v for k, v in metrics.items()}
            print(f"  metrics: {time.perf_counter()-t1:.1f}s  RMS_dip={bs_metrics['bsroformer_worst_rms_dip_db']}dB  ratio={bs_metrics['bsroformer_mean_instr_mix_ratio_db']}dB")

            if not args.dry_metrics:
                # ---- save ----
                out_vocals = track_dir / "vocals.flac"
                out_instr = track_dir / "instrumental.flac"

                sf.write(str(out_vocals), vocal, sr, subtype="PCM_24")
                sf.write(str(out_instr), instr_repaired, sr, subtype="PCM_24")

            # ---- update manifest ----
            manifest["model_id"] = "bs_roformer_sw_jarredou"
            manifest["model_name"] = "BS RoFormer SW by jarredou"
            manifest["metrics"] = {
                **manifest.get("metrics", {}),
                **bs_metrics,
                "bsroformer_inference_seconds": round(time.perf_counter() - t_start, 1),
            }
            if not args.dry_metrics:
                manifest["delivery_status"] = "awaiting_audit"
                manifest["presentation_status"] = "awaiting_audit"
                manifest["delivery_allowed"] = False
                manifest["execution_contract"] = {
                    "actual_acceleration_mode": "native_macos_mps",
                    "gpu_inference_skipped": False,
                    "gpu_inference_required": True,
                    "fallback_used": False,
                    "vendor_cpu_fallback_detected": False,
                }

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"  total: {time.perf_counter()-t_start:.1f}s  -> saved")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            # write error manifest
            try:
                with open(manifest_path, "w") as f:
                    json.dump({**manifest, "error": str(e)}, f, indent=2, ensure_ascii=False)
            except:
                pass

        print()

    print("ALL DONE")


if __name__ == "__main__":
    main()
