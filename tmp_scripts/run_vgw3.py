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
    mns = signal.medfilt(mn, kernel_size=(3,3))
    g = ms**2 / (ms**2 + alpha*mns**2 + 1e-10)
    g = np.maximum(g, floor)
    _,x = signal.istft(Zx*g, fs=sr, nperseg=nf, noverlap=nf-hp)
    if len(x)<n: x = np.pad(x,(0,n-len(x)))
    else: x = x[:n]
    return x

for a,f in [(3.0,0.15),(4.0,0.15),(5.0,0.2)]:
    print("=== alpha=%s floor=%s ===" % (a,f))
    for tn in ["armin-sound-of-you","leps-ona-ne-tvoya","nen-hot-tiktok-douyin-2025","titus-let-it-rip-chipcat"]:
        t0 = time.time()
        d = Path("/tmp/mac_quick_inputs/%s" % tn)
        i,s = sf.read(str(d/"instrumental.flac"))
        v,_ = sf.read(str(d/"vocals.flac"))
        c = vgw(i,v,s,alpha=a,floor=f)
        od = Path("/tmp/vgw3/a%s_f%s/%s" % (a,f,tn))
        od.mkdir(parents=True,exist_ok=True)
        sf.write(str(od/"no_vocals.flac"),c.astype(np.float32),s)
        pg = score_pitch_shifted_vocal_leak(
            vocals_path=str(d/"vocals.flac"),
            instrumental_path=str(od/"no_vocals.flac"),
            output_dir=od/"pitch_guard")
        t = pg.get("total_fail_regions",pg.get("selected_region_count",0))
        st = "PASS" if t<70 else "FAIL(%d)"%t
        print("  %s: %s (%.0fs)"%(tn,st,time.time()-t0))
