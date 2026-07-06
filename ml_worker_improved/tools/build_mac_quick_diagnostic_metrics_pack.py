#!/usr/bin/env python3
"""Diagnostic no-reference metrics pack for mac_quick separation artifacts.

This tool is diagnostic-only. It does not mark any artifact as accepted,
released, or listen-ready. It computes objective relative metrics between
the source raw mix (raw_ref) and the separated vocal+instrumental outputs
produced by the diagnostic Mac MPS MelBand RoFormer route.

Metrics computed per track (all diagnostic-only):
  - waveform_reconstruction_residual_rms_db
      residual = raw_ref - (vocal + instrumental); RMS (dBFS), per channel + max.
      Lower magnitude indicates the (vocal+instr) pair approximates the mix.
  - vocal_energy_fraction_db
      10*log10(vocal_energy / (vocal_energy + instrumental_energy)).
  - instrumental_energy_retention_db
      10*log10(instrumental_energy / raw_ref_energy). Negative => attenuation.
  - per_band_rms_db
      RMS (dB) per band for raw_ref/vocal/instrumental on 5 bands
      (sub, low, lowmid, highmid, air) using scipy butterworth split.
  - vocal_dip_count / instrumental_dip_count
      Short-term RMS dips per stem (>= dip_depth_db drop over > dip_min_sec).
      Detects "muting" / presence loss.
  - spectral_flatness_mean_raw / vocal / instrumental
      Mean spectral flatness over time frames (Wiener_entropy style).
  - spectral_centroid_mean Hz per stem.
  - stereo_width_mean per stem (side/mid energy ratio, dB).

Output:
  - per-track JSON manifest under <out_dir>/<alias>/metrics.json
  - aggregate JSON manifest under <out_dir>/metrics_pack.json
    with provenance (source manifest path, sha256 of each audio, computed_at).

All artifact directories use Windows-safe short aliases; raw track titles stay
only in provenance fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


BAND_NAMES = ("sub", "low", "lowmid", "highmid", "air")
BAND_EDGES_HZ = (20.0, 200.0, 800.0, 4000.0, 12000.0, 20000.0)


def _now_unix() -> float:
    return time.time()


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(
            json.dumps(
                {"time_unix": _now_unix(), "phase": phase, "status": status, **extra},
                sort_keys=True,
            )
            + "\n"
        )
        h.flush()


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    return audio, int(sr)


def _rms_db(audio: np.ndarray, eps: float = 1e-12) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < eps:
        rms = eps
    return float(20.0 * math.log10(rms))


def _energy(audio: np.ndarray) -> float:
    return float(np.sum(audio.astype(np.float64) ** 2))


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _residual_rms_db(raw: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    raw_a, recon_a = _align(raw, recon)
    residual = raw_a - recon_a
    out: dict[str, float] = {"overall_db": _rms_db(residual)}
    ch = residual.shape[1] if residual.ndim == 2 else 1
    if residual.ndim == 1:
        residual = residual[:, None]
        ch = 1
    per_ch = []
    for c in range(ch):
        per_ch.append(_rms_db(residual[:, c]))
    out["per_channel_db"] = per_ch
    out["max_channel_db"] = float(max(per_ch))
    return out


def _band_split(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    if audio.ndim == 1:
        audio = audio[:, None]
    out: dict[str, np.ndarray] = {}
    prev_edge = BAND_EDGES_HZ[0]
    for name, hi in zip(BAND_NAMES, BAND_EDGES_HZ[1:]):
        lo = prev_edge
        if hi >= sr / 2:
            hi = sr / 2 - 1.0
        if lo >= hi:
            out[name] = np.zeros_like(audio)
            prev_edge = hi
            continue
        nyq = sr / 2.0
        low = lo / nyq
        high = hi / nyq
        high = min(high, 0.999)
        low = max(low, 1e-5)
        if high <= low:
            out[name] = np.zeros_like(audio)
            prev_edge = hi
            continue
        sos = butter(4, [low, high], btype="bandpass", output="sos")
        out[name] = sosfiltfilt(sos, audio.astype(np.float64), axis=0).astype(np.float32)
        prev_edge = hi
    return out


def _per_band_rms_db(audio: np.ndarray, sr: int) -> dict[str, float]:
    split = _band_split(audio, sr)
    return {name: _rms_db(b) for name, b in split.items()}


def _spectral_flatness_mean(audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> float:
    import librosa

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float32)
    sf_flat = librosa.feature.spectral_flatness(y=mono, n_fft=n_fft, hop_length=hop)
    return float(np.mean(sf_flat))


def _spectral_centroid_mean(audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> float:
    import librosa

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float32)
    sc = librosa.feature.spectral_centroid(y=mono, sr=sr, n_fft=n_fft, hop_length=hop)
    return float(np.mean(sc))


def _stereo_width_db(audio: np.ndarray, eps: float = 1e-12) -> float:
    if audio.ndim == 1 or audio.shape[1] < 2:
        return float(0.0)
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5
    em = float(np.sum(mid.astype(np.float64) ** 2))
    es = float(np.sum(side.astype(np.float64) ** 2))
    if em < eps:
        em = eps
    if es < eps:
        return float(-120.0)
    return float(10.0 * math.log10(es / em))


def _envelope_rms(audio: np.ndarray, sr: int, frame_ms: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    n = mono.shape[0]
    frame = max(1, int(sr * frame_ms / 1000))
    if n % frame:
        trimmed = n - (n % frame)
        mono = mono[:trimmed]
        n = trimmed
    env = mono.reshape(-1, frame).astype(np.float64)
    rms = np.sqrt(np.mean(env**2, axis=1))
    times = (np.arange(rms.shape[0]) * frame + frame / 2) / sr
    return rms, times


def _detect_dips(
    rms_env: np.ndarray,
    times: np.ndarray,
    panel_db: float,
    min_depth_db: float = 6.0,
    min_duration_s: float = 0.3,
) -> list[dict[str, float]]:
    if rms_env.size == 0:
        return []
    rms_db = 20.0 * np.log10(np.maximum(rms_env, 1e-12))
    dips: list[dict[str, float]] = []
    i = 0
    n = rms_db.size
    while i < n:
        if rms_db[i] <= panel_db - min_depth_db:
            start = i
            while i < n and rms_db[i] <= panel_db - min_depth_db + 3.0:
                i += 1
            end = i - 1
            dur = float(times[end] - times[start]) if end >= start else 0.0
            if dur >= min_duration_s:
                dips.append(
                    {
                        "start_sec": float(times[start]),
                        "end_sec": float(times[end]),
                        "duration_sec": dur,
                        "floor_dip_db": float(panel_db - float(np.min(rms_db[start : end + 1]))),
                    }
                )
        else:
            i += 1
    return dips


def _panel_db(rms_env: np.ndarray) -> float:
    if rms_env.size == 0:
        return -120.0
    p = float(np.percentile(rms_env, 85))
    return float(20.0 * math.log10(max(p, 1e-12)))


def _compute_track_metrics(
    alias: str,
    raw_path: Path,
    vocal_path: Path,
    instr_path: Path,
    source_manifest_path: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    if progress_path is not None:
        _append_progress(progress_path, "track_start", "ok", alias=alias)
    raw, sr_r = _load_audio(raw_path)
    voc, sr_v = _load_audio(vocal_path)
    ins, sr_i = _load_audio(instr_path)
    sr = sr_r or sr_v or sr_i
    if not (sr_r == sr_v == sr_i):
        if progress_path is not None:
            _append_progress(
                progress_path,
                "sample_rate_drift",
                "warning",
                alias=alias,
                raw_sr=sr_r,
                vocal_sr=sr_v,
                instr_sr=sr_i,
            )

    voc, ins = _align(voc, ins)
    recon = voc + ins
    raw_a, recon_a = _align(raw, recon)

    residual_report = _residual_rms_db(raw_a, recon_a)
    e_raw = _energy(raw_a)
    e_voc = _energy(voc)
    e_ins = _energy(ins)
    total = e_voc + e_ins
    vocal_frac = e_voc / total if total > 0 else 0.0
    instr_retention = e_ins / e_raw if e_raw > 0 else 0.0
    vocal_frac_db = float(10.0 * math.log10(max(vocal_frac, 1e-12)))
    instr_ret_db = float(10.0 * math.log10(max(instr_retention, 1e-12)))

    # energy dip detection on raw and stems
    rms_raw, t_raw = _envelope_rms(raw_a, sr)
    rms_voc, t_voc = _envelope_rms(voc, sr)
    rms_ins, t_ins = _envelope_rms(ins, sr)

    raw_panel = _panel_db(rms_raw)
    raw_dips = _detect_dips(rms_raw, t_raw, raw_panel)
    voc_dips = _detect_dips(rms_voc, t_voc, _panel_db(rms_voc))
    ins_dips = _detect_dips(rms_ins, t_ins, _panel_db(rms_ins))

    sf_raw = _spectral_flatness_mean(raw_a, sr)
    sf_voc = _spectral_flatness_mean(voc, sr)
    sf_ins = _spectral_flatness_mean(ins, sr)
    sc_raw = _spectral_centroid_mean(raw_a, sr)
    sc_voc = _spectral_centroid_mean(voc, sr)
    sc_ins = _spectral_centroid_mean(ins, sr)

    out: dict[str, Any] = {
        "alias": alias,
        "diagnostic_only": True,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_metrics_only_blocked_from_listen",
        "sample_rate_hz": sr,
        "duration_sec": float(raw_a.shape[0] / sr) if sr else 0.0,
        "audio_sha256": {
            "raw_ref": _sha256(raw_path),
            "vocal": _sha256(vocal_path),
            "instrumental": _sha256(instr_path),
        },
        "waveform_reconstruction_residual_rms_db": residual_report,
        "vocal_energy_fraction_db": vocal_frac_db,
        "vocal_energy_ratio": vocal_frac,
        "instrumental_energy_retention_db": instr_ret_db,
        "instrumental_energy_ratio": instr_retention,
        "per_band_rms_db": {
            "raw_ref": _per_band_rms_db(raw_a, sr),
            "vocal": _per_band_rms_db(voc, sr),
            "instrumental": _per_band_rms_db(ins, sr),
        },
        "dip_detection": {
            "raw_ref_panel_db": raw_panel,
            "raw_ref_dip_count": len(raw_dips),
            "raw_ref_dips": raw_dips[:20],
            "vocal_dip_count": len(voc_dips),
            "vocal_dips": voc_dips[:20],
            "instrumental_dip_count": len(ins_dips),
            "instrumental_dips": ins_dips[:20],
        },
        "spectral_flatness_mean": {
            "raw_ref": sf_raw,
            "vocal": sf_voc,
            "instrumental": sf_ins,
        },
        "spectral_centroid_mean_hz": {
            "raw_ref": sc_raw,
            "vocal": sc_voc,
            "instrumental": sc_ins,
        },
        "stereo_width_db": {
            "raw_ref": _stereo_width_db(raw_a),
            "vocal": _stereo_width_db(voc),
            "instrumental": _stereo_width_db(ins),
        },
        "provenance": {
            "source_manifest_path": str(source_manifest_path)
            if source_manifest_path is not None
            else None,
            "raw_ref_audio_path": str(raw_path),
            "vocal_audio_path": str(vocal_path),
            "instrumental_audio_path": str(instr_path),
        },
    }
    if progress_path is not None:
        _append_progress(
            progress_path,
            "track_end",
            "ok",
            alias=alias,
            residual_db=residual_report.get("overall_db"),
            vocal_dip_count=len(voc_dips),
            instr_dip_count=len(ins_dips),
            instr_retention_db=instr_ret_db,
        )
    return out


def _iter_track_dirs(mac_quick_root: Path) -> list[tuple[str, Path, Path, Path, Path]]:
    rows: list[tuple[str, Path, Path, Path, Path]] = []
    for d in sorted(mac_quick_root.iterdir()):
        if not d.is_dir():
            continue
        raw = d / "raw_ref.flac"
        voc = d / "vocals.flac"
        ins = d / "instrumental.flac"
        man = d / "stem_manifest.json"
        if raw.exists() and voc.exists() and ins.exists():
            rows.append((d.name, raw, voc, ins, man))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mac-quick-root",
        required=True,
        help="Directory containing per-track mac_quick separation artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for metrics pack (Windows-safe short aliases).",
    )
    parser.add_argument("--alias", default=None, help="Compute metrics only for this alias.")
    args = parser.parse_args(argv)

    mac_quick_root = Path(args.mac_quick_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "metrics_progress.jsonl"
    pack_path = out_dir / "metrics_pack.json"

    _append_progress(progress_path, "pack_start", "ok", mac_quick_root=str(mac_quick_root))

    track_rows = _iter_track_dirs(mac_quick_root)
    if args.alias is not None:
        track_rows = [r for r in track_rows if r[0] == args.alias]
    if not track_rows:
        _append_progress(progress_path, "no_tracks_found", "failed", mac_quick_root=str(mac_quick_root))
        print("[diagnostic_metrics] no matching track directories found", file=sys.stderr)
        return 2

    per_track: list[dict[str, Any]] = []
    for alias, raw, voc, ins, man in track_rows:
        track_out_dir = out_dir / alias
        track_out_dir.mkdir(parents=True, exist_ok=True)
        _append_progress(progress_path, "track_compute_start", "ok", alias=alias)
        rec = _compute_track_metrics(alias, raw, voc, ins, man, progress_path)
        rec_path = track_out_dir / "metrics.json"
        rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        per_track.append(rec)

    aggregate = {
        "schema": "mac_quick_diagnostic_metrics_pack.v1",
        "diagnostic_only": True,
        "delivery_allowed": False,
        "release_ready": False,
        "presentation_status": "diagnostic_metrics_only_blocked_from_listen",
        "computed_at_unix": _now_unix(),
        "tool": "build_mac_quick_diagnostic_metrics_pack.py",
        "track_count": len(per_track),
        "tracks": per_track,
    }
    pack_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _append_progress(progress_path, "pack_end", "ok", track_count=len(per_track))

    # short console summary
    for rec in per_track:
        print(
            f"{rec['alias']}: residual={rec['waveform_reconstruction_residual_rms_db']['overall_db']:.2f}dB "
            f"vocal_frac={rec['vocal_energy_fraction_db']:.2f}dB "
            f"instr_ret={rec['instrumental_energy_retention_db']:.2f}dB "
            f"raw_dips={rec['dip_detection']['raw_ref_dip_count']} "
            f"voc_dips={rec['dip_detection']['vocal_dip_count']} "
            f"instr_dips={rec['dip_detection']['instrumental_dip_count']}"
        )
    print(f"[diagnostic_metrics] wrote {pack_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
