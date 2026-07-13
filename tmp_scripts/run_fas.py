import numpy as np, soundfile as sf, time, sys
from scipy import signal
from pathlib import Path
sys.path.insert(0, "/tmp/mac_tools/tools")
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak

def freq_adaptive_subtract(instr, voc, sr, alpha=1.0, floor_scale=0.5):
    """
    Frequency-domain adaptive subtraction.
    Estimates bleed path response H[f] = E[I * conj(V)] / E[|V|^2]
    Then subtracts: clean = I - H * V

    Additional: apply spectral floor to prevent excessive subtraction
    """
    if instr.ndim > 1: instr = instr[:,0]
    if voc.ndim > 1: voc = voc[:,0]
    n = min(len(instr), len(voc))
    instr = instr[:n]; voc = voc[:n]

    nf = 4096; hp = nf//4
    f,t,Zi = signal.stft(instr, fs=sr, nperseg=nf, noverlap=nf-hp)
    f,t,Zv = signal.stft(voc, fs=sr, nperseg=nf, noverlap=nf-hp)

    # Estimate bleed path response
    # S_iv = E[I * conj(V)], S_vv = E[|V|^2]
    S_iv = np.mean(Zi * np.conj(Zv), axis=1)  # per-frequency, averaged over time
    S_vv = np.mean(np.abs(Zv)**2, axis=1)

    # Smooth for stability
    S_iv = signal.medfilt(np.real(S_iv), kernel_size=5) + 1j * signal.medfilt(np.imag(S_iv), kernel_size=5)
    S_vv = signal.medfilt(S_vv, kernel_size=5)

    H = S_iv / (S_vv + 1e-10)  # per-frequency complex response

    # Estimated bleed in time-frequency
    bleed_est = H[:, None] * Zv  # [freq, time]

    # Subtract
    clean = Zi - alpha * bleed_est

    # Apply per-frequency spectral floor: don't remove more than (1 - floor_scale) of original magnitude
    mag_orig = np.abs(Zi)
    mag_clean = np.abs(clean)
    scale = np.maximum(mag_clean, floor_scale * mag_orig) / (mag_clean + 1e-10)
    clean = clean * scale

    _, x = signal.istft(clean, fs=sr, nperseg=nf, noverlap=nf-hp)
    if len(x) < n: x = np.pad(x, (0, n-len(x)))
    else: x = x[:n]
    return x

for a,fs in [(1.0,0.5),(1.5,0.5),(2.0,0.7),(1.0,0.8)]:
    print("=== alpha=%s floor_scale=%s ===" % (a,fs))
    for tn in ["armin-sound-of-you","leps-ona-ne-tvoya","nen-hot-tiktok-douyin-2025","titus-let-it-rip-chipcat"]:
        t0 = time.time()
        d = Path("/tmp/mac_quick_inputs/%s" % tn)
        i,s = sf.read(str(d/"instrumental.flac"))
        v,_ = sf.read(str(d/"vocals.flac"))
        c = freq_adaptive_subtract(i,v,s,alpha=a,floor_scale=fs)
        od = Path("/tmp/fas/a%s_fs%s/%s" % (a,fs,tn))
        od.mkdir(parents=True,exist_ok=True)
        sf.write(str(od/"no_vocals.flac"),c.astype(np.float32),s)
        pg = score_pitch_shifted_vocal_leak(
            vocals_path=str(d/"vocals.flac"),
            instrumental_path=str(od/"no_vocals.flac"),
            output_dir=od/"pitch_guard")
        t = pg.get("total_fail_regions",pg.get("selected_region_count",0))
        st = "PASS" if t<70 else "FAIL(%d)"%t
        print("  %s: %s (%.0fs)"%(tn,st,time.time()-t0))
