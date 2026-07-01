# Mac Reference Path Recovery

`tools/plan_mac_reference_path_recovery.py` adapts a blocked Mac candidate
readiness summary into the existing metadata-only reference path recovery
planner.

The tool does not copy audio, read audio bytes, hash audio, build ground-truth
bundles, run metrics, start training, promote candidates, or change thresholds.
It writes standard `plan_reference_path_recovery.py` inputs under `inputs/` and
then writes:

```text
reference_path_recovery_plan.jsonl
reference_path_recovery_manifest.json
reference_path_recovery_summary.md
```

The current Mac checkpoint is:

```text
it7rd002/sel: blocked_source_reference_mismatch
actual source: be-nicegreenhorn
wrong reference: square-a-saw-oxygen-tevlo-vocals-resynthesized
```

Docker-only example:

```bash
docker compose run --rm --no-deps \
  -v D:\AIMusicProject\artifacts:/host_artifacts \
  -e REQUIRE_CUDA=0 \
  -e ML_TORCH_DEVICE=cpu \
  ml-worker \
  python tools/plan_mac_reference_path_recovery.py \
    --mac-readiness-summary-json /host_artifacts/mac_api/it7rd002/sel/summary.json \
    --output-dir /host_artifacts/mac_api/it7rp001 \
    --reference-search-root /host_testing_stuff_music
```

Current output:

```text
schema=reference_path_recovery_manifest.v1
source=mac_candidate_readiness_summary
total_rows=1
recovery_status_counts.blocked_missing_reference=1
blocked_reason=reference_bundle_mismatched_to_source
delivery_allowed=false
presentation_status=reference_recovery_plan_not_for_listen_pack
```

The plan row keeps `selected_candidate_id=it7dl003`,
`track_hash=be-nicegreenhorn`, and
`reference_mismatch_status=reference_bundle_mismatched_to_source`, while
remaining blocked until a verified actual-source reference stem exists.
