import numpy as np, soundfile as sf, time, sys
from scipy import signal
from pathlib import Path
sys.path.insert(0, "/tmp/mac_tools/tools")
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak

def vgw(instr, voc, sr, alpha=2.0, floor=0.1):
    if instr.ndim > 1: instr = instr[:,0]
    if voc.ndim > 1: voc = voc[:,0]
    n = min(len(instr), len(voc))
    instr = instr[:n]; voc = voc[:n]
    nf = 4096; hp = nf//4
    _,_,Zx = signal.stft(instr,fs=sr,nperseg=nf,noverlap=nf-hp)
    _,_,Zn = signal.stft(voc,fs=sr,nperseg=nf,noverlap=nf-hp)
    ms = np.abs(Zx); mn = np.abs(Zn)
    mn = signal.medfilt(mn, kernel_size=(3,3))
    g = ms**2 / (ms**2 + alpha*mn**2 + 1e-10)
    g = np.maximum(g, floor)
    _,x = signal.istft(Zx*g, fs=sr, nperseg=nf, noverlap=nf-hp)
    if len(x)<n: x = np.pad(x,(0,n-len(x)))
    else: x = x[:n]
    return x

# Two-pass: aggressive first, gentle second
# Also try: aggressive third pass
PASSES = [
    "1pass_a5_f0.2",
    "2pass_a5_f0.2_a2_f0.3",
]

tn = "armin-sound-of-you"
print("=== Multi-pass on %s ===" % tn)
t0 = time.time()
d = Path("/tmp/mac_quick_inputs/%s" % tn)
i,s = sf.read(str(d/"instrumental.flac"))
v,_ = sf.read(str(d/"vocals.flac"))

# Pass 1: aggressive
c1 = vgw(i, v, s, alpha=5.0, floor=0.2)
od1 = Path("/tmp/mp_vgw/1pass/%s" % tn)
od1.mkdir(parents=True, exist_ok=True)
sf.write(str(od1/"no_vocals.flac"), c1.astype(np.float32), s)
pg1 = score_pitch_shifted_vocal_leak(
    vocals_path=str(d/"vocals.flac"), instrumental_path=str(od1/"no_vocals.flac"),
    output_dir=od1/"pitch_guard")
total1 = pg1.get("total_fail_regions", pg1.get("selected_region_count", 0))
print("  Pass 1 (a5 f0.2): %d (%.0fs)" % (total1, time.time()-t0))

# Pass 2: gentle on pass 1 output
c2 = vgw(c1.astype(np.float64), v, s, alpha=2.0, floor=0.3)
od2 = Path("/tmp/mp_vgw/2pass/%s" % tn)
od2.mkdir(parents=True, exist_ok=True)
sf.write(str(od2/"no_vocals.flac"), c2.astype(np.float32), s)
pg2 = score_pitch_shifted_vocal_leak(
    vocals_path=str(d/"vocals.flac"), instrumental_path=str(od2/"no_vocals.flac"),
    output_dir=od2/"pitch_guard")
total2 = pg2.get("total_fail_regions", pg2.get("selected_region_count", 0))
print("  Pass 2 (a2 f0.3 on pass1): %d (%.0fs)" % (total2, time.time()-t0))

# Pass 3: even more gentle
c3 = vgw(c2.astype(np.float64), v, s, alpha=1.0, floor=0.5)
od3 = Path("/tmp/mp_vgw/3pass/%s" % tn)
od3.mkdir(parents=True, exist_ok=True)
sf.write(str(od3/"no_vocals.flac"), c3.astype(np.float32), s)
pg3 = score_pitch_shifted_vocal_leak(
    vocals_path=str(d/"vocals.flac"), instrumental_path=str(od3/"no_vocals.flac"),
    output_dir=od3/"pitch_guard")
total3 = pg3.get("total_fail_regions", pg3.get("selected_region_count", 0))
print("  Pass 3 (a1 f0.5 on pass2): %d (%.0fs)" % (total3, time.time()-t0))

# Also try: VG Wiener using model's vocal (neural-guided)
# If we use the BS-RoFo-SW vocal stem as noise profile instead of original vocal
# First, we need model vocal at 48000. Let's try what we have.
# Actually skip this for now.

print("\n=== Now on all tracks with 2-pass ===")
for tn in ["leps-ona-ne-tvoya","nen-hot-tiktok-douyin-2025","titus-let-it-rip-chipcat"]:
    t0 = time.time()
    d = Path("/tmp/mac_quick_inputs/%s" % tn)
    i,s = sf.read(str(d/"instrumental.flac"))
    v,_ = sf.read(str(d/"vocals.flac"))
    c1 = vgw(i, v, s, alpha=5.0, floor=0.2)
    c2 = vgw(c1.astype(np.float64), v, s, alpha=2.0, floor=0.3)
    od = Path("/tmp/mp_vgw/2pass/%s" % tn)
    od.mkdir(parents=True, exist_ok=True)
    sf.write(str(od/"no_vocals.flac"), c2.astype(np.float32), s)
    pg = score_pitch_shifted_vocal_leak(
        vocals_path=str(d/"vocals.flac"), instrumental_path=str(od/"no_vocals.flac"),
        output_dir=od/"pitch_guard")
    total = pg.get("total_fail_regions", pg.get("selected_region_count", 0))
    st = "PASS" if total<70 else "FAIL(%d)"%total
    print("  %s: %s (%.0fs)"%(tn,st,time.time()-t0))
