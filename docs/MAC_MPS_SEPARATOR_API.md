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
  - `scnet_xl_ihf_musdb18`
- Outputs must stay under the configured shared artifacts root:
  `/Users/andrejviluha/AIMusicProject_win/artifacts`.
- Forbidden future result roots are
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

## Start

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
- `scnet_xl_ihf_musdb18`: verified through the API on a 12 second slice under
  `artifacts/mac_api/api_scnet002`; this route is slow on 8GB MPS and uses a
  longer exporter timeout.

These artifacts are diagnostic evidence only. They do not satisfy the
production listen/delivery contract by themselves.

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
