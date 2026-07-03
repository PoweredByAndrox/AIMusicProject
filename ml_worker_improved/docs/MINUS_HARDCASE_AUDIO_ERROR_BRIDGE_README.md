# Minus Hardcase Audio Error Bridge

`tools/build_minus_hardcase_audio_error_bridge.py` connects existing minus hardcase runtime roots to the audio-error training-view pipeline.

Pipeline:

1. runtime roots with `candidate_pool.jsonl` and `minus_windowed_judge.jsonl`
2. `build_audio_error_grounding_dataset`
3. `audit_audio_error_dataset_provenance`
4. `repair_audio_error_dataset_provenance`
5. `build_audio_error_training_views`
6. `validate_audio_error_training_views`

The bridge writes only evidence/training artifacts:

- `bridge_manifest.json`
- `progress.jsonl`
- `mount_preflight.json` (for Mac artifact-producing runs)
- `ground/`
- `audit/`
- `repaired/`
- `views/`

Safety contract:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `listen_delivery_artifact=false`
- `release_ready=false`
- `training_started=false`
- no diagnostic candidate is promoted to a listen/result artifact
- runtime rows containing `testing_flstudio_music` are blocked before grounding

Mac execution should use the project Mac wrapper and write under the mounted Windows workspace, for example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 -MacHost 192.168.0.138 -MacUser andrejviluha -SshKey "$env:USERPROFILE\.ssh\id_ed25519_phone" -MacProjectRoot /Users/andrejviluha/AIMusicProject_win -RemoteCommand 'cd "$AIMUSIC_MAC_PROJECT_ROOT" && /usr/local/bin/container --version'
```

For artifact-producing runs, use the project-approved Apple-container/Mac route and output under:

`/Users/andrejviluha/AIMusicProject_win/artifacts/...`

Mac runs that can create bridge artifacts must pass mount preflight evidence into the bridge:

- run `tools/mac_windows_mount_preflight.py` logic on the Mac host before the container run;
- write the resulting JSON as `mount_preflight.json` under the same Windows-mounted output directory;
- invoke the bridge with `--mount-preflight-json ... --require-mount-preflight`;
- keep `delivery_allowed=false` and `presentation_status=training_evidence_not_for_listen_pack`.

The final `bridge_manifest.json` must record:

- `mount_preflight_status=passed`
- `mount_probe_status=write_read_delete_passed`
- `mac_output_path=/Users/andrejviluha/AIMusicProject_win/artifacts/...`
- `windows_output_path=D:\AIMusicProject\artifacts\...`

If mount preflight is missing or blocked, the bridge must stop with `output_status=blocked_mount_preflight` before grounding/training-view generation.
