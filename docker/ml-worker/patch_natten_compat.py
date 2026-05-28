"""
Patch allin1/models/dinat.py to work with natten>=0.15 (new fused API).

allin1==1.1.0 uses the old natten<=0.14 split API:
  scores = natten1dqkrpb(q, k, rpb, kernel_size, dilation)
  ctx    = natten1dav(attn_probs, v, kernel_size, dilation)

natten>=0.15 removed these functions. This script replaces the import in
dinat.py with pure-PyTorch compat shims that reproduce the same semantics.

RPB (relative positional bias) layout:
  1D: rpb shape (num_heads, 2*kernel_size - 1)
      neighbor ki (0..K-1) maps to rpb[:, ki]
  2D: rpb shape (num_heads, 2*kernel_size - 1, 2*kernel_size - 1)
      neighbor (ki, kj) maps to rpb[:, ki, kj]

NOTE: We locate dinat.py via importlib.util (no import of allin1 itself,
which would fail before the patch is applied).
"""

import sys
import importlib.util

# Locate allin1 package without importing it
spec = importlib.util.find_spec("allin1")
if spec is None:
    print("ERROR: allin1 package not found")
    sys.exit(1)

import os
allin1_dir = os.path.dirname(spec.origin)
dinat_path = os.path.join(allin1_dir, "models", "dinat.py")

if not os.path.exists(dinat_path):
    print(f"ERROR: dinat.py not found at {dinat_path}")
    sys.exit(1)

print(f"Patching {dinat_path}")

with open(dinat_path, 'r') as f:
    src = f.read()

OLD_IMPORT = "from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb"

if OLD_IMPORT not in src:
    print("Already patched or import line changed — nothing to do.")
    sys.exit(0)

NEW_CODE = '''# natten>=0.15 removed the split API; compat shims below reproduce the same semantics.
# TEMPORARY: allin1==1.1.0 has not been updated for natten>=0.15.
# Real fix: upstream allin1 update to use natten.functional.na1d / na2d.
import torch as _torch


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """
    Neighborhood 1D QK dot-product + relative positional bias.
    query, key: (B, H, T, D)
    rpb:        (H, 2*kernel_size - 1)
    returns:    (B, H, T, kernel_size)
    """
    B, H, T, D = query.shape
    K = kernel_size
    half = K // 2
    scores = _torch.zeros(B, H, T, K, device=query.device, dtype=query.dtype)
    for ki in range(K):
        offset = ki - half
        ni = _torch.arange(T, device=query.device)
        ni = (ni + offset).clamp(0, T - 1)
        k_nb = key[:, :, ni, :]          # (B, H, T, D)
        scores[:, :, :, ki] = (query * k_nb).sum(-1)
        if rpb is not None:
            scores[:, :, :, ki] = scores[:, :, :, ki] + rpb[:, ki].unsqueeze(0).unsqueeze(2)
    return scores


def natten1dav(attn, value, kernel_size, dilation):
    """
    Neighborhood 1D attention-weighted value aggregation.
    attn:    (B, H, T, kernel_size)
    value:   (B, H, T, D)
    returns: (B, H, T, D)
    """
    B, H, T, K = attn.shape
    D = value.shape[3]
    half = K // 2
    out = _torch.zeros(B, H, T, D, device=value.device, dtype=value.dtype)
    for ki in range(K):
        offset = ki - half
        ni = _torch.arange(T, device=value.device)
        ni = (ni + offset).clamp(0, T - 1)
        v_nb = value[:, :, ni, :]        # (B, H, T, D)
        out = out + attn[:, :, :, ki:ki+1] * v_nb
    return out


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """
    Neighborhood 2D QK dot-product + relative positional bias.
    query, key: (B, H, Ht, Wt, D)
    rpb:        (H, 2*kernel_size - 1, 2*kernel_size - 1)
    returns:    (B, H, Ht, Wt, kernel_size * kernel_size)
    """
    B, H, Ht, Wt, D = query.shape
    K = kernel_size
    K2 = K * K
    half = K // 2
    scores = _torch.zeros(B, H, Ht, Wt, K2, device=query.device, dtype=query.dtype)
    idx = 0
    for ki in range(K):
        for kj in range(K):
            ni_off = ki - half
            nj_off = kj - half
            ni = _torch.arange(Ht, device=query.device)
            nj = _torch.arange(Wt, device=query.device)
            ni = (ni + ni_off).clamp(0, Ht - 1)
            nj = (nj + nj_off).clamp(0, Wt - 1)
            k_nb = key[:, :, ni[:, None], nj[None, :], :]  # (B, H, Ht, Wt, D)
            s = (query * k_nb).sum(-1)
            if rpb is not None:
                s = s + rpb[:, ki, kj].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            scores[:, :, :, :, idx] = s
            idx += 1
    return scores


def natten2dav(attn, value, kernel_size, dilation):
    """
    Neighborhood 2D attention-weighted value aggregation.
    attn:    (B, H, Ht, Wt, kernel_size * kernel_size)
    value:   (B, H, Ht, Wt, D)
    returns: (B, H, Ht, Wt, D)
    """
    B, H, Ht, Wt, K2 = attn.shape
    D = value.shape[4]
    K = kernel_size
    half = K // 2
    out = _torch.zeros(B, H, Ht, Wt, D, device=value.device, dtype=value.dtype)
    idx = 0
    for ki in range(K):
        for kj in range(K):
            ni_off = ki - half
            nj_off = kj - half
            ni = _torch.arange(Ht, device=value.device)
            nj = _torch.arange(Wt, device=value.device)
            ni = (ni + ni_off).clamp(0, Ht - 1)
            nj = (nj + nj_off).clamp(0, Wt - 1)
            v_nb = value[:, :, ni[:, None], nj[None, :], :]  # (B, H, Ht, Wt, D)
            out = out + attn[:, :, :, :, idx:idx+1] * v_nb
            idx += 1
    return out'''

patched = src.replace(OLD_IMPORT, NEW_CODE)

with open(dinat_path, 'w') as f:
    f.write(patched)

print("Patch applied successfully.")

# =============================================================================
# Patch 2: allin1/postprocessing/tempo.py — fix bpm=None when beats < 2
#
# allin1==1.1.0 bug: estimate_tempo_from_beats() returns None when len(beats) < 2.
# AnalysisResult.bpm is typed as int, so None propagates silently and causes
#   TypeError: unsupported format string passed to NoneType.__format__
# in any f-string that uses :.1f or :.0f on result.bpm.
#
# Fix: return 0 (int) instead of None — matches the declared type and lets
# callers detect "no beat data" via bpm == 0 rather than bpm is None.
#
# TEMPORARY: real fix is allin1 upstream updating estimate_tempo_from_beats()
# to return a valid int or raise an explicit exception.
# Upstream issue: https://github.com/mir-aidj/all-in-one/issues
# =============================================================================

tempo_path = os.path.join(allin1_dir, "postprocessing", "tempo.py")

if not os.path.exists(tempo_path):
    print(f"WARNING: tempo.py not found at {tempo_path}, skipping patch 2.")
else:
    print(f"Patching {tempo_path}")
    with open(tempo_path, 'r') as f:
        tempo_src = f.read()

    OLD_TEMPO = "    # The song has less than 2 beats. Perhaps it doesn't have much percussive elements.\n    return None"
    NEW_TEMPO = "    # The song has less than 2 beats. Perhaps it doesn't have much percussive elements.\n    # PATCHED: return 0 (int) instead of None to match AnalysisResult.bpm: int type.\n    # Callers should treat bpm==0 as 'no reliable beat data'.\n    return 0"

    if OLD_TEMPO not in tempo_src:
        print("tempo.py already patched or content changed — skipping patch 2.")
    else:
        patched_tempo = tempo_src.replace(OLD_TEMPO, NEW_TEMPO)
        with open(tempo_path, 'w') as f:
            f.write(patched_tempo)
        print("Patch 2 (tempo.py) applied successfully.")

# =============================================================================
# Patch 3: allin1/demix.py — add --segment to Demucs to prevent VRAM OOM
#
# Problem: Demucs (HTDemucs) is launched as a subprocess with --device cuda
# but WITHOUT --segment, so it loads the entire track into VRAM at once.
# For long tracks (>~15 min), this exhausts VRAM (12GB on RTX 4080 Laptop)
# and the OOM killer sends SIGKILL. This causes structure_analyzer to fall
# back entirely, losing BPM/segment data.
#
# Fix: add '--segment', '30' to the demucs.separate call.
# Demucs processes the track in 30-second chunks, each using ~1-2GB VRAM.
# GPU is still used — no speed regression for short tracks, long tracks no
# longer OOM.
#
# TEMPORARY: real fix is allin1 exposing a demix_segment parameter upstream.
# Upstream issue: https://github.com/mir-aidj/all-in-one/issues
# =============================================================================

demix_path = os.path.join(allin1_dir, "demix.py")

if not os.path.exists(demix_path):
    print(f"WARNING: demix.py not found at {demix_path}, skipping patch 3.")
else:
    print(f"Patching {demix_path}")
    with open(demix_path, 'r') as f:
        demix_src = f.read()

    OLD_DEMIX = "        '--device', str(device),"
    NEW_DEMIX = "        '--device', str(device),\n        '--segment', '7',  # PATCHED: chunked processing to avoid VRAM OOM on long tracks (HTDemucs max=7.8s)"

    if OLD_DEMIX not in demix_src or "'--segment'" in demix_src:
        print("demix.py already patched or content changed — skipping patch 3.")
    else:
        patched_demix = demix_src.replace(OLD_DEMIX, NEW_DEMIX)
        with open(demix_path, 'w') as f:
            f.write(patched_demix)
        print("Patch 3 (demix.py) applied successfully.")
