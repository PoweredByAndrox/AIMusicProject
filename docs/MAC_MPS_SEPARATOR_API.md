# Mac MPS Separator API

This is a diagnostic-only native macOS MPS runner for local separator checks on
Apple Silicon machines. It does not replace the production delivery validation
contract and it cannot mark listen or delivery artifacts as accepted.

Before starting it from Windows/Codex, bootstrap the current Mac access and
mounted Windows workspace:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1
```

## Contract

- The server runs registered local separator routes with
  `AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1`.
- Supported route ids:
  - `htdemucs_6s`
  - `mdx23c_vocals`
  - `melband_roformer_vocals`
  - `becruily_instrumental_vocals`
  - `scnet_xl_ihf_musdb18`
  - `bs_roformer_large_inst`
- Outputs must stay under the configured shared artifacts root:
  `/Users/andrejviluha/AIMusicProject_win/artifacts`.
- The API defaults to `/Users/andrejviluha/AIMusicProject_win` and
  `/Users/andrejviluha/AIMusicProject_win/artifacts` when those root options are
  omitted, and startup fails before accepting jobs if `--artifacts-root` points
  anywhere else.
- Forbidden future result roots are
  `/Users/andrejviluha/aimusic_local_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/host_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/mac_api`,
  `/Users/andrejviluha/win_artifacts`, `/Users/andrejviluha/artifacts`, and
  `/tmp/artifacts`.
- Job ids are short Windows-safe aliases and must not include raw track names.
- Every job writes `manifest.json`, `progress.jsonl`, `run.log`, and model
  output under `run/`.
- Every API artifact is diagnostic-only:
  - `delivery_allowed=false`
  - `listen_pack_created=false`
  - `presentation_status=diagnostic_mps_api_only`
- Accepted/listen output still requires the normal production delivery contract.
- `status=ok` means that the registered inference command completed. It is not a
  clean-stem verdict. Dedicated minus candidates must subsequently pass the
  multi-proxy gate documented in
  `ml_worker_improved/docs/MINUS_QUALITY_ENSEMBLE_README.md`; that proxy pass
  still cannot replace an exact-reference or accepted production policy gate.

## Start

For resident Large-Inst execution, start the single-model worker first and add:

```text
--large-inst-worker-socket /tmp/aimusic_li.sock
--large-inst-worker-timeout-sec 600
--route-timeout-sec 1800
```

The API fails at startup if the configured socket is unavailable. It never
silently falls back to the cold Large-Inst exporter. Worker startup, cache
release, shutdown, timing, memory, and quality-lock evidence are documented in
`ml_worker_improved/docs/MAC_MPS_LARGE_INST_WORKER_README.md`.

`/health` includes resident worker status and the loaded config/checkpoint/
implementation fingerprint. `/separate` atomically reserves a new output root;
any pre-existing path is rejected, including an empty directory, so concurrent
requests cannot cross-write. Missing or zero request timeouts use the finite
`--route-timeout-sec` value, and values above that configured maximum are
rejected. The effective request timeout is also passed to the inner
`mdx23c_vocals` MSST subprocess instead of falling back to its shorter
standalone default. Mount-preflight failures do not write into the requested
path.
HTTP access logs are written to the mounted Windows artifact path
`artifacts/mac_api/server.log`; request delivery does not depend on the SSH
session's stdout/stderr descriptors. `/health` exposes access-log write status
and failure evidence.

```bash
/Users/andrejviluha/aimusic_mps_venv/bin/python \
  ml_worker_improved/tools/mac_mps_separator_api.py \
  --project-root /Users/andrejviluha/AIMusicProject_win \
  --artifacts-root /Users/andrejviluha/AIMusicProject_win/artifacts \
  --mac-windows-project-mount /Users/andrejviluha/AIMusicProject_win \
  --windows-project-root 'D:\AIMusicProject' \
  --python-bin /Users/andrejviluha/aimusic_mps_venv/bin/python \
  --host 0.0.0.0 \
  --port 8765
```

## Health

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://192.168.0.138:8765/health"
```

## Submit Job

```powershell
$body = @{
  source_audio_path = "/Users/andrejviluha/AIMusicProject_win/testing_stuff_music/source.mp3"
  output_subdir = "mac_api/api_full001"
  route_id = "htdemucs_6s"
} | ConvertTo-Json -Compress

Invoke-WebRequest `
  -UseBasicParsing `
  -Uri "http://192.168.0.138:8765/jobs" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Poll:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://192.168.0.138:8765/jobs/<job_id>"
```

## Verified Routes

- `htdemucs_6s`: verified through the API on a full-track diagnostic run under
  `artifacts/mac_api/api_full001`.
- `mdx23c_vocals`: verified through the API on a 12 second slice under
  `artifacts/mac_api/api_mdx003`.
- `melband_roformer_vocals`: verified through the API on a 12 second slice under
  `artifacts/mac_api/api_mel002`.
- `becruily_instrumental_vocals`: dedicated instrumental/minus checkpoint route
  through the shared MelBand exporter. It must expose `instrumental.flac`
  directly from the Becruily instrumental checkpoint; it must not materialize
  minus from stem-model outputs.
- `scnet_xl_ihf_musdb18`: verified through the API on a 12 second slice under
  `artifacts/mac_api/api_scnet002`; this route is slow on 8GB MPS and uses a
  longer exporter timeout.
- `bs_roformer_large_inst`: verified through the API on short and full-track
  inputs with strict checkpoint loading, exact isolated architecture,
  `PYTORCH_ENABLE_MPS_FALLBACK=0`, and a device-aware MPS float16 autocast
  compatibility layer. Its source license metadata is still incomplete, so
  outputs remain diagnostic candidates. The precision contract explicitly
  requires a post-inference quality-equivalence gate; accelerator success alone
  cannot promote the artifact. Its exporter records the observed native Mac MPS
  runtime rather than claiming Docker execution, and candidate-only output is not
  copied into a listen pack.
  The resident Mac worker path is also verified: responses expose
  `resident_backend_used=true`, retain `model_load_count=1`, verify temporary
  scratch cleanup, and map the API `instrumental` alias to the same dedicated
  `target_stem=minus` output without summing stem-model files.

These artifacts are diagnostic evidence only. They do not satisfy the
production listen/delivery contract by themselves.

The current `bs_roformer_large_inst` audit at
`artifacts/mac_api/qa001/ens2/manifest.json` is explicitly blocked: independent
MelBand and MDX23C vocal proxies agree on a `0-1 s` leak/full-mix-like segment,
and source conservation blocks `4-5 s` and `8-9 s` for detail loss.

## MelBand Pitch Guard

`melband_roformer_vocals` can leave vocal-like material in `instrumental.flac`
after pitch/formant damage. The `melband_pitch_guard_v1` diagnostic profile
adds a post-pass for that defect class:

```powershell
docker compose run --rm --no-deps -T --volume D:/AIMusicProject:/repo --workdir /repo/ml_worker_improved ml-worker `
  python tools/run_mac_melband_pitch_guard_pipeline.py `
    --mac-api-url "http://192.168.0.138:8765" `
    --source-audio-path "/Users/andrejviluha/AIMusicProject_win/testing_stuff_music/source.mp3" `
    --api-output-subdir "mac_api/mel_guard001/api" `
    --out "/repo/artifacts/mac_api/mel_guard001"
```

The profile:

- starts a Mac API `melband_roformer_vocals` job;
- scans `vocals.flac` against `instrumental.flac` with pitch-shifted vocal
  probes over the whole audio, not a hardcoded timestamp;
- blocks with `pitch_shifted_vocal_in_instrumental` when vocal content is
  still present in the instrumental after pitch/formant-like transformation;
- can write a mixture-consistent diagnostic repair under short paths:
  `audio/v.flac`, `audio/i.flac`, `audio/raw.flac`, `score/pitch.json`,
  `repair/v.flac`, `repair/i.flac`;
- always remains diagnostic-only:
  `delivery_allowed=false`, `listen_pack_created=false`,
  `presentation_status=diagnostic_melband_pitch_guard_only`.

`artifacts/mac_api/api_mel002/run` is the regression seed for this guard,
including the user-reported tail region near 11-12 seconds. That timing is not
a production special case; the detector must scan every window and only store
the timing as evidence in `score/pitch.json`.
