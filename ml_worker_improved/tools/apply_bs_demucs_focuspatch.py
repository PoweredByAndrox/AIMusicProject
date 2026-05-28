from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from docker_runtime import require_docker_runtime  # noqa: E402

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return audio.astype(np.float32), int(sample_rate)


def _match_length(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(arr.shape[0] for arr in arrays)
    return [arr[:length] for arr in arrays]


def _lr_to_ms(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = stereo[:, 0]
    right = stereo[:, 1]
    return (left + right) * 0.5, (left - right) * 0.5


def _ms_to_lr(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=1)


def _band_energy(y: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, spec = stft(y, fs=sample_rate, nperseg=2048, noverlap=1024, nfft=2048, boundary="zeros")
    return freqs, np.abs(spec) ** 2


def _band_db_ratio(freqs: np.ndarray, source_energy: np.ndarray, ref_energy: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    num = float(source_energy[mask].sum())
    den = float(ref_energy[mask].sum())
    return 10.0 * np.log10((num + 1e-18) / (den + 1e-18))


def _stereo_width(stereo: np.ndarray) -> float:
    mid, side = _lr_to_ms(stereo)
    mid_rms = float(np.sqrt(np.mean(mid**2)))
    side_rms = float(np.sqrt(np.mean(side**2)))
    return side_rms / max(mid_rms, 1e-12)


def _residual_rms(raw: np.ndarray, candidate: np.ndarray) -> float:
    residual = raw.mean(axis=1) - candidate.mean(axis=1)
    return float(np.sqrt(np.mean(residual**2)))


def _make_mid_weights(freqs: np.ndarray, profile: str) -> np.ndarray:
    weights = np.zeros_like(freqs, dtype=np.float32)
    if profile in {"balanced", "mid_only"}:
        weights[freqs < 250.0] = 0.0
        weights[(freqs >= 250.0) & (freqs < 700.0)] = 0.30
        weights[(freqs >= 700.0) & (freqs < 2500.0)] = 0.70

        upper_mid = (freqs >= 2500.0) & (freqs < 6000.0)
        if np.any(upper_mid):
            weights[upper_mid] = np.linspace(0.20, 0.08, int(np.sum(upper_mid)), dtype=np.float32)

        weights[freqs >= 6000.0] = 0.03
        return weights

    if profile == "vocal_band":
        weights[freqs < 350.0] = 0.0
        weights[(freqs >= 350.0) & (freqs < 900.0)] = 0.20
        weights[(freqs >= 900.0) & (freqs < 2800.0)] = 0.62

        upper_mid = (freqs >= 2800.0) & (freqs < 4500.0)
        if np.any(upper_mid):
            weights[upper_mid] = np.linspace(0.18, 0.06, int(np.sum(upper_mid)), dtype=np.float32)

        weights[freqs >= 4500.0] = 0.0
        return weights

    raise ValueError(f"Unsupported profile: {profile}")


def _make_side_weights(freqs: np.ndarray, profile: str) -> np.ndarray:
    if profile in {"mid_only", "vocal_band"}:
        return np.zeros_like(freqs, dtype=np.float32)

    if profile != "balanced":
        raise ValueError(f"Unsupported profile: {profile}")

    weights = np.zeros_like(freqs, dtype=np.float32)
    weights[freqs < 250.0] = 0.02
    weights[(freqs >= 250.0) & (freqs < 700.0)] = 0.02
    weights[(freqs >= 700.0) & (freqs < 2500.0)] = 0.01
    weights[freqs >= 2500.0] = 0.0
    return weights


def _make_time_mask(*, length: int, sample_rate: int, patch_windows: list[tuple[float, float]], xfade_sec: float) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for index in range(length):
        t = index / float(sample_rate)
        value = 0.0
        for patch_start, patch_end in patch_windows:
            if patch_start - xfade_sec <= t < patch_start:
                value = max(value, (t - (patch_start - xfade_sec)) / max(xfade_sec, 1e-8))
            elif patch_start <= t <= patch_end:
                value = max(value, 1.0)
            elif patch_end < t <= patch_end + xfade_sec:
                value = max(value, 1.0 - ((t - patch_end) / max(xfade_sec, 1e-8)))
        mask[index] = value
    return mask


def main() -> None:
    require_docker_runtime(
        tool_name="apply_bs_demucs_focuspatch.py",
        entrypoint_path="/app/tools/apply_bs_demucs_focuspatch.py",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_audio_path")
    parser.add_argument("bs_minus_path")
    parser.add_argument("demucs_minus_path")
    parser.add_argument("output_dir")
    parser.add_argument("--listen-dir")
    parser.add_argument("--listen-label", default="02_bs_demucs_focuspatch.flac")
    parser.add_argument("--patch-start-sec", type=float, default=50.0)
    parser.add_argument("--patch-end-sec", type=float, default=70.0)
    parser.add_argument("--xfade-sec", type=float, default=1.0)
    parser.add_argument("--profile", choices=["balanced", "mid_only", "vocal_band"], default="balanced")
    parser.add_argument("--patch-window", nargs=2, action="append", metavar=("START_SEC", "END_SEC"), help="Repeatable focused patch window.")
    args = parser.parse_args()

    raw_audio_path = Path(args.raw_audio_path)
    bs_minus_path = Path(args.bs_minus_path)
    demucs_minus_path = Path(args.demucs_minus_path)
    output_dir = Path(args.output_dir)
    listen_dir = Path(args.listen_dir) if args.listen_dir else None

    raw_audio, raw_sr = _load_audio(raw_audio_path)
    bs_minus, bs_sr = _load_audio(bs_minus_path)
    demucs_minus, demucs_sr = _load_audio(demucs_minus_path)
    if len({raw_sr, bs_sr, demucs_sr}) != 1:
        raise ValueError("All inputs must have the same sample rate")

    raw_audio, bs_minus, demucs_minus = _match_length(raw_audio, bs_minus, demucs_minus)
    sample_rate = raw_sr

    patch_start = float(args.patch_start_sec)
    patch_end = float(args.patch_end_sec)
    xfade_sec = float(args.xfade_sec)
    profile = str(args.profile)
    patch_windows = [(patch_start, patch_end)]
    if args.patch_window:
        patch_windows = [(float(start), float(end)) for start, end in args.patch_window]

    focus_start_sec = min(start for start, _ in patch_windows)
    focus_end_sec = max(end for _, end in patch_windows)

    start_frame = max(0, int(round((focus_start_sec - xfade_sec) * sample_rate)))
    end_frame = min(bs_minus.shape[0], int(round((focus_end_sec + xfade_sec) * sample_rate)))

    bs_seg = bs_minus[start_frame:end_frame]
    demucs_seg = demucs_minus[start_frame:end_frame]

    mid_bs, side_bs = _lr_to_ms(bs_seg)
    mid_demucs, side_demucs = _lr_to_ms(demucs_seg)

    n_fft = 4096
    hop = 1024
    noverlap = n_fft - hop

    freqs, _, mid_bs_z = stft(mid_bs, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, boundary="zeros", padded=True)
    _, _, mid_demucs_z = stft(mid_demucs, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, boundary="zeros", padded=True)
    _, _, side_bs_z = stft(side_bs, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, boundary="zeros", padded=True)
    _, _, side_demucs_z = stft(side_demucs, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, boundary="zeros", padded=True)

    mid_weights = _make_mid_weights(freqs, profile)[:, None]
    side_weights = _make_side_weights(freqs, profile)[:, None]

    mid_hybrid_z = (1.0 - mid_weights) * mid_bs_z + mid_weights * mid_demucs_z
    side_hybrid_z = (1.0 - side_weights) * side_bs_z + side_weights * side_demucs_z

    _, mid_hybrid = istft(mid_hybrid_z, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, input_onesided=True, boundary=True)
    _, side_hybrid = istft(side_hybrid_z, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap, input_onesided=True, boundary=True)

    patch = _ms_to_lr(mid_hybrid[: min(len(mid_hybrid), len(side_hybrid))], side_hybrid[: min(len(mid_hybrid), len(side_hybrid))])

    refined = bs_minus.copy()
    patch_length = min(patch.shape[0], end_frame - start_frame)
    patch = patch[:patch_length]
    current = refined[start_frame:start_frame + patch_length]
    local_patch_windows = [
        (start - (start_frame / sample_rate), end - (start_frame / sample_rate))
        for start, end in patch_windows
    ]
    mask = _make_time_mask(length=patch_length, sample_rate=sample_rate, patch_windows=local_patch_windows, xfade_sec=xfade_sec)[:, None]
    refined[start_frame:start_frame + patch_length] = (1.0 - mask) * current + mask * patch

    peak = float(np.max(np.abs(refined)))
    if peak > 0.9999:
        refined = refined / peak * 0.9999

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_ref_path = output_dir / "raw_ref.flac"
    bs_base_path = output_dir / "bs_minus_base.flac"
    demucs_ref_path = output_dir / "demucs_minus_ref.flac"
    hybrid_path = output_dir / "minus_hybrid.flac"
    sf.write(str(raw_ref_path), raw_audio, sample_rate, subtype="PCM_24")
    sf.write(str(bs_base_path), bs_minus, sample_rate, subtype="PCM_24")
    sf.write(str(demucs_ref_path), demucs_minus, sample_rate, subtype="PCM_24")
    sf.write(str(hybrid_path), refined, sample_rate, subtype="PCM_24")

    focus_start = int(round(focus_start_sec * sample_rate))
    focus_end = int(round(focus_end_sec * sample_rate))
    raw_focus = raw_audio[focus_start:focus_end]
    bs_focus = bs_minus[focus_start:focus_end]
    demucs_focus = demucs_minus[focus_start:focus_end]
    hybrid_focus = refined[focus_start:focus_end]

    freqs_bs, bs_focus_energy = _band_energy(bs_focus.mean(axis=1), sample_rate)
    _, demucs_focus_energy = _band_energy(demucs_focus.mean(axis=1), sample_rate)
    _, hybrid_focus_energy = _band_energy(hybrid_focus.mean(axis=1), sample_rate)

    manifest = {
        "raw_audio_path": str(raw_audio_path),
        "bs_minus_path": str(bs_minus_path),
        "demucs_minus_path": str(demucs_minus_path),
        "artifacts": {
            "raw_ref_flac": str(raw_ref_path),
            "bs_minus_base_flac": str(bs_base_path),
            "demucs_minus_ref_flac": str(demucs_ref_path),
            "minus_hybrid_flac": str(hybrid_path),
        },
        "patch": {
            "name": "bs_demucs_focuspatch_v1",
            "profile": profile,
            "patch_start_sec": focus_start_sec,
            "patch_end_sec": focus_end_sec,
            "patch_windows_sec": [[start, end] for start, end in patch_windows],
            "xfade_sec": xfade_sec,
        },
        "focus_metrics": {
            "bs_width": _stereo_width(bs_focus),
            "demucs_width": _stereo_width(demucs_focus),
            "hybrid_width": _stereo_width(hybrid_focus),
            "raw_bs_residual_rms": _residual_rms(raw_focus, bs_focus),
            "raw_demucs_residual_rms": _residual_rms(raw_focus, demucs_focus),
            "raw_hybrid_residual_rms": _residual_rms(raw_focus, hybrid_focus),
            "hybrid_vs_bs_db_20_200": _band_db_ratio(freqs_bs, hybrid_focus_energy, bs_focus_energy, 20.0, 200.0),
            "hybrid_vs_bs_db_200_800": _band_db_ratio(freqs_bs, hybrid_focus_energy, bs_focus_energy, 200.0, 800.0),
            "hybrid_vs_bs_db_800_3000": _band_db_ratio(freqs_bs, hybrid_focus_energy, bs_focus_energy, 800.0, 3000.0),
            "hybrid_vs_bs_db_3000_6000": _band_db_ratio(freqs_bs, hybrid_focus_energy, bs_focus_energy, 3000.0, 6000.0),
            "hybrid_vs_bs_db_6000_16000": _band_db_ratio(freqs_bs, hybrid_focus_energy, bs_focus_energy, 6000.0, 16000.0),
            "hybrid_vs_demucs_db_20_200": _band_db_ratio(freqs_bs, hybrid_focus_energy, demucs_focus_energy, 20.0, 200.0),
            "hybrid_vs_demucs_db_200_800": _band_db_ratio(freqs_bs, hybrid_focus_energy, demucs_focus_energy, 200.0, 800.0),
            "hybrid_vs_demucs_db_800_3000": _band_db_ratio(freqs_bs, hybrid_focus_energy, demucs_focus_energy, 800.0, 3000.0),
            "hybrid_vs_demucs_db_3000_6000": _band_db_ratio(freqs_bs, hybrid_focus_energy, demucs_focus_energy, 3000.0, 6000.0),
            "hybrid_vs_demucs_db_6000_16000": _band_db_ratio(freqs_bs, hybrid_focus_energy, demucs_focus_energy, 6000.0, 16000.0),
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "frames": int(refined.shape[0]),
            "duration_sec": round(float(refined.shape[0]) / float(sample_rate), 6),
        },
    }
    manifest_path = output_dir / "stem_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if listen_dir is not None:
        listen_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(listen_dir / "00_raw_ref.flac"), raw_audio, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / "01_bs_mamba2_minus_refined.flac"), bs_minus, sample_rate, subtype="PCM_24")
        sf.write(str(listen_dir / args.listen_label), refined, sample_rate, subtype="PCM_24")
        (listen_dir / "00_README.txt").write_text(
            f"Track 15 hybrid focus patch.\nListen only here.\n00 = raw reference\n01 = BS Mamba2 minus refined\n02 = BS/Demucs focus patch ({profile})\n",
            encoding="utf-8",
        )
        (listen_dir / "candidate_map.json").write_text(
            json.dumps(
                {
                    "00_raw_ref.flac": str(raw_ref_path),
                    "01_bs_mamba2_minus_refined.flac": str(bs_base_path),
                    args.listen_label: str(hybrid_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"output_dir": str(output_dir), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
