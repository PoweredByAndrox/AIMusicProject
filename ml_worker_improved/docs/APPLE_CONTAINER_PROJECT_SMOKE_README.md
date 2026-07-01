# Apple Container Project Smoke

`tools/run_apple_container_project_smoke.py` is a metadata-only Mac runtime
smoke for the approved Apple `container` route. It mounts the project read-only
into an Apple container, sets `AIMUSIC_CONTAINER_RUNTIME=apple_container`, and
runs selected stdlib Python tests.

The smoke never renders audio and never creates a listen pack:

- `delivery_allowed=false`
- `presentation_status=diagnostic_apple_container_project_smoke_only`
- `audio_generated=false`
- `gpu_inference_skipped=true`

## Mac Run

Run on the Mac host:

```bash
python3 tools/run_apple_container_project_smoke.py \
  --project-root /Users/andrejviluha/aimusic_remote_work_it024 \
  --output-dir /Users/andrejviluha/aimusic_local_artifacts/host_artifacts/loop/it035/acsm001 \
  --image docker.io/library/python:3.11-alpine \
  --timeout-sec 600
```

The command uses Apple `container` at `/usr/local/bin/container`. It does not
use Docker Desktop, Docker daemon, `docker compose`, or Docker VMM.
The `docker.io/library/...` value is only an OCI image reference consumed by
Apple `container`; it is not a Docker CLI or daemon route.

## Outputs

The output directory contains short, machine-readable artifacts:

```text
manifest.json
progress.jsonl
cmd.json
stdout.txt
stderr.txt
```

`manifest.json` uses schema `apple_container_project_smoke.v1`. A passed smoke
proves the project can run metadata tests inside Apple `container`; it does not
prove PyTorch MPS or audio model acceleration inside the Linux container.
