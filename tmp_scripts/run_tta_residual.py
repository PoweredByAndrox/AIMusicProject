import torch, yaml, time, sys, os
from pathlib import Path
from functools import wraps

if torch.backends.mps.is_available():
    _orig_amp_ac = torch.amp.autocast
    class _MPSA:
        def __init__(self, device_type="mps", enabled=True, dtype=None, cache_enabled=None):
            d = "mps" if (device_type is None or device_type == "cuda") else device_type
            self._ac = _orig_amp_ac(d, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        def __enter__(self): return self._ac.__enter__()
        def __exit__(self, *a): return self._ac.__exit__(*a)
        def __call__(self, f):
            @wraps(f)
            def w(*a, **kw):
                with self: return f(*a, **kw)
            return w
    torch.amp.autocast = _MPSA
    torch.cuda.amp.autocast = _MPSA
    torch.cuda.is_available = lambda: False
    from yaml.constructor import SafeConstructor
    SafeConstructor.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))

from bs_roformer import get_model_from_config, demix_track
from ml_collections import ConfigDict
import soundfile as sf
import numpy as np
from scipy import signal as ss

sys.path.insert(0, "/tmp/mac_tools/tools")
from mac_melband_pitch_guard import score_pitch_shifted_vocal_leak

device = torch.device("mps")

# Load BS-RoFo-SW config+model (6-stem)
with open("/tmp/bs_model_cache/BS-Rofo-SW-Fixed.yaml") as f:
    raw = yaml.load(f, Loader=yaml.SafeLoader)
raw["training"]["model_type"] = "bs_roformer"
config = ConfigDict(raw)

model = get_model_from_config("bs_roformer", config)
state = torch.load("/tmp/bs_model_cache/BS-Rofo-SW-Fixed.ckpt", map_location="cpu", weights_only=True)
state = state["state_dict"] if "state_dict" in state else state
model.load_state_dict(state)
model = model.to(device)
model.eval()
print("Model loaded\n")

def hardened_residual(mix, vocals):
    """instrumental = mix - vocals, clamped by |mix| * 1.05"""
    n = min(mix.shape[-1], vocals.shape[-1])
    mix = mix[:, :n]
    vocals = vocals[:, :n]
    inst = mix - vocals
    max_abs = np.abs(mix) * 1.05 + 1e-6
    np.clip(inst, -max_abs, max_abs, out=inst)
    np.nan_to_num(inst, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return inst

PASS_THRESH = 70
TRACKS = ["armin-sound-of-you"]

for tn in TRACKS:
    t0 = time.time()
    print(f"\n=== {tn} ===")
    d = Path(f"/tmp/mac_quick_inputs/{tn}")
    instr, sr = sf.read(str(d / "instrumental.flac"))
    voc, _ = sf.read(str(d / "vocals.flac"))
    n = min(len(instr), len(voc))
    mix = instr[:n] + voc[:n]

    # Resample to 44100 if needed
    if sr != 44100:
        new_len = int(len(mix) * 44100 / sr)
        mix_r = ss.resample(mix, new_len)
    else:
        mix_r = mix

    if mix_r.ndim == 1:
        mix_r = np.column_stack([mix_r, mix_r])

    # TTA: identity + polarity flip, average vocals
    vocal_acc = None

    for aug, label in [(1.0, "identity"), (-1.0, "polarity")]:
        mix_t = torch.from_numpy((mix_r.T * aug).astype(np.float32)).float()
        with torch.no_grad():
            result = demix_track(config, model, mix_t, device)
        result_dict = result[0]
        voc_stem = result_dict["vocals"]  # [ch, samples]

        # Invert augmentation
        if aug < 0:
            voc_stem = -voc_stem

        if vocal_acc is None:
            vocal_acc = voc_stem.astype(np.float32)
        else:
            vocal_acc += voc_stem.astype(np.float32)

    # Average TTA
    vocal_acc *= 0.5

    # Residual: instrumental = mix - vocal
    # mix_r.T is [ch, samples], vocal_acc is [ch, samples]
    mix_ch = np.ascontiguousarray(mix_r.T).astype(np.float32)
    inst_arr = hardened_residual(mix_ch, vocal_acc)
    inst_final = inst_arr.T  # [samples, ch]

    # Resample back to original SR if needed
    if sr != 44100:
        new_len2 = int(len(inst_final) * sr / 44100)
        if inst_final.ndim == 1:
            inst_final_r = ss.resample(inst_final, new_len2)
        else:
            chs = [ss.resample(inst_final[:,c], new_len2) for c in range(inst_final.shape[1])]
            inst_final_r = np.column_stack(chs)
    else:
        inst_final_r = inst_final

    # Save
    out_dir = Path(f"/tmp/tta_residual/{tn}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "no_vocals.flac"), inst_final_r.astype(np.float32), sr)

    # Pitch-guard at original SR
    pg = score_pitch_shifted_vocal_leak(
        vocals_path=str(d / "vocals.flac"),
        instrumental_path=str(out_dir / "no_vocals.flac"),
        output_dir=out_dir / "pitch_guard",
    )
    total = pg.get("total_fail_regions", pg.get("selected_region_count", 0))
    s = "PASS" if total < PASS_THRESH else f"FAIL({total})"
    print(f"  TTA+Residual: {s} ({time.time()-t0:.0f}s)")
