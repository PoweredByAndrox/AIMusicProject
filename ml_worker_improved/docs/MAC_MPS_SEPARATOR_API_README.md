# Mac MPS Separator API

`tools/mac_mps_separator_api.py` is a diagnostic-only native macOS MPS separator
API. It may run model export commands on the Mac, but it cannot publish listen
or release artifacts.

Native MPS jobs record `observed_runtime.runtime_environment=native_mac_mps_diagnostic`.
This is distinct from Apple-container orchestration and from Docker execution; diagnostic
candidate output remains in its run artifact directory and is never copied into a listen pack.

`status=ok` is an execution status, not a quality verdict. Dedicated minus
outputs must be evaluated by `tools/audit_minus_quality_ensemble.py` with at
least two independent vocal-model manifests. See
`docs/MINUS_QUALITY_ENSEMBLE_README.md`. Even `proxy_pass` remains
diagnostic-only until an exact-reference or accepted production policy gate
also passes.

Supported route ids:

- `htdemucs_6s`
- `mdx23c_vocals`
- `melband_roformer_vocals`
- `becruily_instrumental_vocals`
- `scnet_xl_ihf_musdb18`
- `bs_roformer_large_inst`

The Large-Inst route can use the persistent native MPS worker by starting the
API with `--large-inst-worker-socket /tmp/aimusic_li.sock`. The socket is
validated at startup and the exporter fails closed if the resident contract is
not available. See `ml_worker_improved/docs/MAC_MPS_LARGE_INST_WORKER_README.md`.

Mac artifact output is fail-closed:

- `--artifacts-root` must point into the mounted Windows project workspace,
  normally `/Users/andrejviluha/AIMusicProject_win/artifacts`;
- `--mac-windows-project-mount` must point at the mounted Windows project root,
  normally `/Users/andrejviluha/AIMusicProject_win`;
- if either option is omitted, the API uses those canonical paths by default;
- server startup fails before accepting jobs when the configured
  `--artifacts-root` is forbidden, outside the mounted Windows project, or not
  under `/Users/andrejviluha/AIMusicProject_win/artifacts`;
- every job runs `mac_windows_project_mount_preflight.v1` before creating its
  output directory or starting a worker thread;
- each accepted job atomically creates a previously nonexistent output root;
  existing paths are rejected instead of being reused, including empty paths;
- synchronous and asynchronous exporters always receive a finite subprocess
  timeout; `--route-timeout-sec` defaults to `1800` seconds and acts as the
  maximum request override;
- `mdx23c_vocals` passes that effective request timeout to its inner MSST
  subprocess, so the exporter's `180` second standalone default cannot shorten
  a full-track API request;
- if the mount is missing, unwritable, outside the mounted project, or not under
  `artifacts`, `lab_artifacts`, or `artifact_rag`, the API returns a blocked
  in-memory manifest and does not create Mac-local artifacts;
- future result roots under
  `/Users/andrejviluha/aimusic_local_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/host_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/mac_api`,
  `/Users/andrejviluha/win_artifacts`, `/Users/andrejviluha/artifacts`, or
  `/tmp/artifacts` are forbidden and blocked by the shared mount preflight;
- passed manifests include `mount_preflight`, `mac_windows_project_mount`, and
  `windows_output_path`.
- HTTP access logs are written to `artifacts/mac_api/server.log` in the mounted
  Windows workspace, so revoked SSH stdout/stderr descriptors cannot break API
  responses; `/health` reports the access-log path, status, write-failure count,
  and last error.

Example Mac launch:

```bash
python3 tools/mac_mps_separator_api.py \
  --project-root /Users/andrejviluha/AIMusicProject_win \
  --artifacts-root /Users/andrejviluha/AIMusicProject_win/artifacts \
  --mac-windows-project-mount /Users/andrejviluha/AIMusicProject_win \
  --windows-project-root 'D:\AIMusicProject' \
  --python-bin /path/to/python \
  --route-timeout-sec 1800 \
  --host 127.0.0.1 \
  --port 8765
```
