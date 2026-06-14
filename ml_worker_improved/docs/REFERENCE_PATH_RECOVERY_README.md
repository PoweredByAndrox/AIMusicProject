# Reference Path Recovery Plan

`tools/plan_reference_path_recovery.py` consumes Phase 10E source-path mapping rows and plans
reference path recovery for rows still blocked by missing vocal/minus reference paths.

This phase is plan-only. It does not copy audio, read audio content, hash audio content, build a
ground-truth bundle, run reference metrics, promote artifacts, start training, change policy,
change thresholds, add models, add weights, or add cleanup/debleed/denoise stages.

## Inputs

- `source_path_mapping_plan.jsonl`
- `source_path_mapping_manifest.json`
- `docker_mount_verification.json`
- `reference_bundle_recovery_plan.jsonl`
- `collected_evidence.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`

Only rows with `mapping_status=blocked_missing_reference_path` are included. Source rows that are
already mapped, ambiguous for source paths, or blocked for other reasons remain out of scope.

## Outputs

- `reference_path_recovery_plan.jsonl`
- `reference_path_recovery_manifest.json`
- `reference_path_recovery_summary.md`

Each output row includes:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `source_manifest`
- `current_mapping_status`
- `discovered_source_audio_paths`
- `discovered_reference_candidates`
- `missing_reference_types`
- `reference_search_roots`
- `candidate_match_method`
- `recovery_status`
- `can_recover_reference_audio`
- `can_run_metrics_after_reference_recovery`
- `can_build_gt_bundle_after_reference_recovery`
- `required_next_tool`
- `blocked_reason`
- `forbidden_actions`

## Recovery Policy

- `reference_candidate_found`: exactly one existing isolated reference candidate is found under an
  allowed root.
- `blocked_ambiguous_reference`: multiple allowed reference candidates exist, so automatic recovery
  is blocked.
- `blocked_missing_reference`: no isolated vocal/minus reference candidate is found.
- `blocked_reference_outside_allowed_roots`: a reference-looking path exists, but only outside the
  approved roots.

Generated, selected, listen-pack, adaptive, proxy, and candidate audio paths are not treated as
reference paths. This prevents a failed/generated vocal stem from being promoted into a reference.

## Allowed Roots

The default allowed reference search roots are:

- `/app/lab_artifacts`
- `/host_testing_stuff_music`
- `/app/testing_flstudio_music`

By default, the tool uses these roots as an allowlist for explicit reference paths found in
manifests/recovery rows. It does not perform broad recursive discovery through `/app/lab_artifacts`
unless `--reference-search-root` is passed explicitly for a bounded investigation. The tool records
path existence and root membership only. It does not read audio bytes.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_reference_path_recovery.py \
  --source-path-mapping-jsonl /app/lab_artifacts/control_plane_phase10e_20260528/remap_after_mount/source_path_mapping_plan.jsonl \
  --source-path-mapping-manifest-json /app/lab_artifacts/control_plane_phase10e_20260528/remap_after_mount/source_path_mapping_manifest.json \
  --docker-mount-verification-json /app/lab_artifacts/control_plane_phase10e_20260528/docker_mount_verification.json \
  --reference-recovery-plan-jsonl /app/lab_artifacts/control_plane_phase10c_20260528/reference_bundle_recovery_plan.jsonl \
  --collected-evidence-jsonl /app/lab_artifacts/control_plane_phase10b_20260528/collected_evidence.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10f_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_reference_path_recovery.py \
  tests/test_plan_source_path_mapping.py \
  tests/test_artifact_status_contract.py \
  -q
```
