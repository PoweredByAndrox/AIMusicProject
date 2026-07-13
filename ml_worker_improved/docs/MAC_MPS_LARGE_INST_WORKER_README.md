# Mac MPS Large-Inst Worker

`tools/mac_mps_large_inst_worker.py` is a persistent, single-model native Mac
MPS process for `bs_roformer_large_inst`. It loads the exact checkpoint once,
serializes inference requests over a Unix socket, and keeps the model-specific
autocast/STFT patches isolated from the API process and other model lanes.

This worker is an execution optimization. It does not improve separation
quality and cannot accept an output.

## Safety Contract

- native PyTorch MPS only;
- `PYTORCH_ENABLE_MPS_FALLBACK=0` is mandatory;
- `AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1` is mandatory;
- one exact model implementation and one strict checkpoint per process;
- requests are serialized; no concurrent MPS inference inside one worker;
- an incomplete Unix-socket request is disconnected after a finite server-side
  read timeout so one client cannot stall the serialized worker;
- inference failure poisons the worker and requires restart;
- no cold-export fallback when the configured socket is unavailable;
- every inference request must match the loaded config, checkpoint, and exact
  implementation SHA-256 fingerprint;
- result manifests and canonical audio are written directly to the mounted
  Windows workspace;
- source/output bytes under the system temporary directory are allowed only
  when explicitly declared `disposable_scratch` and the exporter verifies
  cleanup;
- every output remains `delivery_allowed=false` and
  `presentation_status=blocked_until_quality_audit`.

## Start

Run from the Mac host after the Windows project share is mounted:

```bash
cd /Users/andrejviluha/AIMusicProject_win/ml_worker_improved
PYTORCH_ENABLE_MPS_FALLBACK=0 \
AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1 \
PYTHONPATH=/Users/andrejviluha/AIMusicProject_win/ml_worker_improved:/Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/music_source_separation_training \
/Users/andrejviluha/aimusic_mps_venv/bin/python \
  tools/mac_mps_large_inst_worker.py \
  --socket-path /tmp/aimusic_li.sock \
  --project-root /Users/andrejviluha/AIMusicProject_win \
  --artifacts-root /Users/andrejviluha/AIMusicProject_win/artifacts \
  --state-dir /Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/pf001/wrk/state3 \
  --windows-project-root D:/AIMusicProject \
  --runtime-repo /Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/music_source_separation_training \
  --model-implementation-path /Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/bs_roformer_large_inst_diag/bs_roformer.py \
  --config-path /Users/andrejviluha/AIMusicProject_win/artifacts/mac_api/pf001/sm2/run/cfg.yaml \
  --checkpoint-path /Users/andrejviluha/AIMusicProject_win/ml_worker_improved/vendor/bs_roformer_large_inst_diag/bs_large_v2_inst.ckpt
```

The state directory is diagnostic and must use a short path under the mounted
Windows `artifacts` root. The socket and temporary input/output staging are
Mac-local control/scratch surfaces, not result roots.

## Control

```bash
python tools/mac_mps_worker_client.py --socket-path /tmp/aimusic_li.sock --action health
python tools/mac_mps_worker_client.py --socket-path /tmp/aimusic_li.sock --action release_cache
python tools/mac_mps_worker_client.py --socket-path /tmp/aimusic_li.sock --action shutdown
```

`release_cache` synchronizes MPS and releases cached inference buffers without
unloading model weights. Use it between model-lane waves, not between every
track in a homogeneous Large-Inst batch. A poisoned worker rejects this action
with `restart_required=true`; cache release cannot change poisoned state back
to ready.

The CLI exposes control actions only. Separation requests must go through the
exporter/API so the expected model fingerprint, output hash verification,
canonical artifact materialization, and scratch cleanup contract are enforced.

## API Integration

Start `tools.mac_mps_separator_api` with:

```text
--large-inst-worker-socket /tmp/aimusic_li.sock
--large-inst-worker-timeout-sec 600
```

The API validates the socket at startup. Large-Inst exporter requests then use
the resident worker; an unavailable or contract-invalid worker blocks the job
instead of silently running the cold model path. Successful diagnostic API
responses report:

- `resident_backend_used=true`
- `resident_capable=true`
- `resident_worker_evidence.model_load_count=1`
- `resident_worker_evidence.scratch_cleanup_verified=true`

`/health` also reports the worker state and loaded fingerprint. A configured
but unavailable or poisoned worker changes API health to `degraded`.

`/separate` requires a previously nonexistent output root and reserves it
atomically. It does not read stale `run/stem_manifest.json` files, concurrent
requests cannot share an output root, and a failed mount preflight does not
write to the rejected path.

The dedicated minus can satisfy an API request for the `instrumental` alias
only when its source manifest explicitly declares `target_stem=minus`. The API
copies that same dedicated model output; it never builds minus by summing stem
model outputs.

## Measured Result

The 12-second `pf001/sm2` source produced this evidence on Mac MPS:

- model load: `2.098 s`;
- warm worker inference in the hardened API integration run: `7.892 s`;
- resident exporter: `9.040 s`;
- prior cold exporter: `10.846 s`;
- exporter time reduction: about `16.7%`;
- repeated raw worker output SHA-256 remained identical;
- old cold output versus resident canonical output: `130.43 dB` SDR,
  `99.99999999%` cosine, residual peak `-138.47 dBFS`;
- explicit cache release reduced Metal driver allocation from about `2.66 GB`
  to about `0.29 GB` while preserving `model_load_count=1`.

Evidence roots:

- `artifacts/mac_api/pf001/wrk`
- `artifacts/mac_api/pf001/wx1`
- `artifacts/mac_api/pf001/ax3`

Execution parity is not clean-minus quality. The corresponding ensemble audit
at `artifacts/mac_api/qa001/ax3/manifest.json` remains `proxy_blocked` with
three confirmed blocker windows, so no listen/delivery result was promoted.
