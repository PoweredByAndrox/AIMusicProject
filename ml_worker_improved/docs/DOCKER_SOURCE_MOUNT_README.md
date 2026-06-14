# Docker Source Mount Repair

Phase 10E adds one read-only Docker mount so legacy source paths in manifests can be inspected by
`ml-worker`:

```text
./testing_stuff_music:/host_testing_stuff_music:ro
```

This is a visibility repair only. It does not copy audio, export audio as a result, read audio for
metrics, build a ground-truth bundle, run reference metrics, start training, change policy, change
thresholds, add models, add weights, or add cleanup/debleed/denoise stages.

## Why

Phase 10D found 9 recovery rows whose source manifests point to `/host_testing_stuff_music/...`.
The host directory exists as `testing_stuff_music`, but Docker did not expose it under the legacy
manifest prefix. The added mount makes those paths visible without rewriting manifests.

## Verification

Static config check:

```bash
docker compose run --rm --no-deps -v ./docker-compose.yml:/app/docker-compose.yml:ro \
  ml-worker pytest tests/test_docker_source_mount_config.py -q
```

Runtime visibility check:

```bash
docker compose run --rm --no-deps ml-worker sh -lc \
  "test -d /host_testing_stuff_music && find /host_testing_stuff_music -maxdepth 2 -type f | head -20"
```

Planner-only remap:

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_source_path_mapping.py \
  --reference-recovery-plan-jsonl /app/lab_artifacts/control_plane_phase10c_20260528/reference_bundle_recovery_plan.jsonl \
  --reference-recovery-manifest-json /app/lab_artifacts/control_plane_phase10c_20260528/reference_bundle_recovery_manifest.json \
  --collected-evidence-jsonl /app/lab_artifacts/control_plane_phase10b_20260528/collected_evidence.jsonl \
  --missing-evidence-plan-jsonl /app/lab_artifacts/control_plane_phase10a_20260528/missing_evidence_plan.jsonl \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --output-dir /app/lab_artifacts/control_plane_phase10e_20260528/remap_after_mount
```
