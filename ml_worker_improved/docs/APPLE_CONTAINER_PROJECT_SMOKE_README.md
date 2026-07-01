# Apple Container Project Smoke

`tools/run_apple_container_project_smoke.py` is a metadata-only Mac runtime
smoke for the approved Apple `container` route. It mounts the project read-only
into an Apple container, sets `AIMUSIC_CONTAINER_RUNTIME=apple_container`, and
runs selected stdlib Python tests.

Before writing any files, the smoke runs the shared Mac Windows-project mount
preflight. The output directory must be inside the mounted Windows workspace
under `artifacts`, `lab_artifacts`, or `artifact_rag`; otherwise the smoke
returns a blocked manifest in memory and does not create a Mac-local output
directory.

The smoke never renders audio and never creates a listen pack:

- `delivery_allowed=false`
- `presentation_status=diagnostic_apple_container_project_smoke_only`
- `audio_generated=false`
- `gpu_inference_skipped=true`

## Mac Run

Run on the Mac host:

```bash
python3 tools/run_apple_container_project_smoke.py \
  --project-root /Volumes/AIMusicProject \
  --mac-windows-project-mount /Volumes/AIMusicProject \
  --windows-project-root 'D:\AIMusicProject' \
  --output-dir /Volumes/AIMusicProject/artifacts/mac_api/acsm001 \
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
The manifest includes `mount_preflight`, `mac_windows_project_mount`, and
`windows_output_path` so the Mac path can be traced back to the Windows project
folder.
