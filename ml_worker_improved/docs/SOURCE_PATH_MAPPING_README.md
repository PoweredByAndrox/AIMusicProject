# Source Path Mapping Plan

`tools/plan_source_path_mapping.py` consumes Phase 10C recovery rows and creates a deterministic
plan for Docker-visible source/reference path mapping.

This phase is plan-only. It does not copy audio, read audio content, hash audio, build a
ground-truth bundle, run reference metrics, promote artifacts, start training, change policy, relax
thresholds, add models, add weights, or add cleanup/debleed/denoise stages.

## Inputs

- `reference_bundle_recovery_plan.jsonl`
- `reference_bundle_recovery_manifest.json`
- `collected_evidence.jsonl`
- `missing_evidence_plan.jsonl`
- `hard_case_benchmark.jsonl`
- `golden_corpus.jsonl`

## Outputs

- `source_path_mapping_plan.jsonl`
- `source_path_mapping_manifest.json`
- `source_path_mapping_summary.md`
- `docker_mount_suggestions.md`

Each row includes:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `source_manifest`
- `original_paths_found`
- `missing_paths`
- `path_prefixes_detected`
- `docker_visible_candidate_paths`
- `host_path_candidates`
- `suggested_mapping_rule`
- `mapping_status`
- `can_repair_manifest_paths`
- `can_recover_source_audio`
- `can_recover_reference_audio`
- `can_run_phase10e_after_mapping`
- `required_docker_mount`
- `forbidden_actions`

## Mapping Policy

- `mapping_found`: all relevant paths are already Docker-visible.
- `mapping_candidate_found`: one deterministic Docker-visible candidate path exists for a missing
  manifest path.
- `blocked_missing_host_path`: no Docker-visible source path or known mount mapping can be found.
- `blocked_missing_docker_mount`: a legacy `/host_*` prefix appears in manifests but that prefix is
  absent in Docker.
- `blocked_missing_reference_path`: source audio is visible or mappable, but reference vocal/minus
  input is still missing.
- `blocked_ambiguous_mapping`: more than one Docker-visible candidate path matches a missing input.

`/host_testing_stuff_music` is treated as a legacy manifest prefix. If that prefix appears and is not
mounted in Docker, the planner suggests a read-only mount:

```text
./testing_stuff_music:/host_testing_stuff_music:ro
```

The planner does not edit `docker-compose.yml`; it only writes `docker_mount_suggestions.md`.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_source_path_mapping.py \
  --reference-recovery-plan-jsonl /app/lab_artifacts/control_plane_phase10c_20260528/reference_bundle_recovery_plan.jsonl \
  --reference-recovery-manifest-json /app/lab_artifacts/control_plane_phase10c_20260528/reference_bundle_recovery_manifest.json \
  --collected-evidence-jsonl /app/lab_artifacts/control_plane_phase10b_20260528/collected_evidence.jsonl \
  --missing-evidence-plan-jsonl /app/lab_artifacts/control_plane_phase10a_20260528/missing_evidence_plan.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10d_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_source_path_mapping.py \
  tests/test_plan_reference_bundle_recovery.py \
  tests/test_artifact_status_contract.py \
  -q
```
