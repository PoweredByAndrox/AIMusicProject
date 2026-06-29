# Mac MPS Separator API

This is a diagnostic-only native macOS MPS runner for local separator checks on
Apple Silicon machines. It does not replace Docker production validation and it
cannot mark listen or delivery artifacts as accepted.

## Contract

- The server runs `ml_worker_improved/tools/export_hq_stems_full.py` with
  `--device mps` and `AIMUSIC_ALLOW_NATIVE_MAC_DIAGNOSTIC=1`.
- Outputs must stay under the configured shared artifacts root.
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
  --project-root /Users/andrejviluha/Desktop/work_git/AIMusicProject \
  --artifacts-root /Users/andrejviluha/win_artifacts \
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
  source_audio_path = "/Users/andrejviluha/win_project/testing_stuff_music/source.mp3"
  output_subdir = "mac_api/api_full001"
  model_name = "htdemucs_6s"
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

## Current Limit

The verified native MPS route is `htdemucs_6s`. CUDA-only exporters such as
MDX23C, MelBand RoFormer, and SCNet still require separate Mac/MPS adaptation
before they can be considered working native Mac routes.
