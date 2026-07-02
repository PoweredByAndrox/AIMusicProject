# macOS Remote API Command

`tools/run_macos_remote_api_command.py` sends a diagnostic shell command to the
Mac over SSH and writes command artifacts under a mounted Windows project
workspace.

The route is fail-closed:

- `--mac-windows-project-mount` must point at the Mac-visible Windows project
  mount, normally `/Users/andrejviluha/AIMusicProject_win`; when omitted, the
  runner defaults to that canonical Windows project mount;
- `--mac-artifact-root` must be inside that mount under `artifacts`,
  `lab_artifacts`, or `artifact_rag`; when omitted, the runner defaults to
  `/Users/andrejviluha/AIMusicProject_win/artifacts`;
- without a mount mapping, the client returns `status=blocked`,
  `returncode=78`, and does not open SSH;
- when a mapping is present, the remote script performs a write/read/delete
  probe under `$MOUNT/.aimusic_mount_probe` before creating the output
  directory or running the command;
- the client result and remote manifest record `mount_preflight`,
  `mac_windows_project_mount`, and `windows_output_path`;
- output remains diagnostic-only with `delivery_allowed=false`.

Current local-network contract:

- Windows share: `AIMusicProject` -> `D:\AIMusicProject`;
- Mac mount path: `/Users/andrejviluha/AIMusicProject_win`;
- `scripts\macos-ssh.ps1` exports `AIMUSIC_MAC_PROJECT_ROOT`,
  `AIMUSIC_MAC_ARTIFACT_ROOT`, `AIMUSIC_MAC_LAB_ARTIFACTS_ROOT`, and
  `AIMUSIC_MAC_ARTIFACT_RAG_ROOT` for ad-hoc `-RemoteCommand` runs, so result
  artifacts can be written directly into the mounted Windows project workspace;
- SMB account: `MacAudioApi` with write access to the project artifact roots;
- a Mac route must not write production artifacts to Mac-local home paths,
  `/tmp`, or Apple-container scratch as the final artifact location; the
  configured home path above is valid only because it is an SMB-mounted Windows
  project share;
- forbidden result roots are
  `/Users/andrejviluha/aimusic_local_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/host_artifacts`,
  `/Users/andrejviluha/aimusic_local_artifacts/mac_api`,
  `/Users/andrejviluha/win_artifacts`, `/Users/andrejviluha/artifacts`, and
  `/tmp/artifacts`; the client blocks these before opening SSH;
- `scripts\macos-ssh.ps1` must also block those roots when they are supplied as
  `-MacProjectRoot` or appear in derived exported roots such as
  `AIMUSIC_MAC_ARTIFACT_ROOT`, not only when they appear literally in
  `-RemoteCommand`;
- if SMB authentication or the write/read/delete probe fails, the route is
  blocked instead of creating Mac-local listen/result artifacts.

Bootstrap from Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-workspace-mount.ps1
```

`macos-bootstrap-and-ssh.ps1` is the from-zero command: it calls the workspace
mount bootstrap, verifies Apple `/usr/local/bin/container`, and prints the
future SSH command. The lower-level mount script verifies the explicit-key SSH
route, ensures the Windows shares and `MacAudioApi` access, mounts
`\\<windows-ip>\AIMusicProject` on the Mac at
`/Users/andrejviluha/AIMusicProject_win`, and runs the mount write/read/delete
probe. For a full credential reset and remount:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-workspace-mount.ps1 -ForceCredentialReset
```

Connect later from Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 -RemoteCommand '/usr/local/bin/container --version'
```

Example:

```bash
python tools/run_macos_remote_api_command.py \
  --ssh-target andrejviluha@192.168.0.138 \
  --ssh-key "$USERPROFILE/.ssh/id_ed25519_phone" \
  --mac-windows-project-mount /Users/andrejviluha/AIMusicProject_win \
  --windows-project-root 'D:\AIMusicProject' \
  --mac-artifact-root /Users/andrejviluha/AIMusicProject_win/artifacts \
  --artifact-subdir mac_api/c001 \
  --command 'printf ok'
```
