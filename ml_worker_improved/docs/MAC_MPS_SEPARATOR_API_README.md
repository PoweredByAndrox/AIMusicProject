# Mac MPS Separator API

`tools/mac_mps_separator_api.py` is a diagnostic-only native macOS MPS separator
API. It may run model export commands on the Mac, but it cannot publish listen
or release artifacts.

Mac artifact output is fail-closed:

- `--artifacts-root` must point into the mounted Windows project workspace,
  normally `/Users/andrejviluha/AIMusicProject_win/artifacts`;
- `--mac-windows-project-mount` must point at the mounted Windows project root,
  normally `/Users/andrejviluha/AIMusicProject_win`;
- every job runs `mac_windows_project_mount_preflight.v1` before creating its
  output directory or starting a worker thread;
- if the mount is missing, unwritable, outside the mounted project, or not under
  `artifacts`, `lab_artifacts`, or `artifact_rag`, the API returns a blocked
  in-memory manifest and does not create Mac-local artifacts;
- passed manifests include `mount_preflight`, `mac_windows_project_mount`, and
  `windows_output_path`.

Example Mac launch:

```bash
python3 tools/mac_mps_separator_api.py \
  --project-root /Users/andrejviluha/AIMusicProject_win \
  --artifacts-root /Users/andrejviluha/AIMusicProject_win/artifacts \
  --mac-windows-project-mount /Users/andrejviluha/AIMusicProject_win \
  --windows-project-root 'D:\AIMusicProject' \
  --python-bin /path/to/python \
  --host 127.0.0.1 \
  --port 8765
```
