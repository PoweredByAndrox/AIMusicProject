# Mac Promising Blocked Handoff

`tools/plan_mac_promising_blocked_handoff.py` converts a blocked Mac readiness
summary into the standard `short_listen_repair_queue_manifest.v1` queue consumed
by `tools/build_full_pack_loop_iteration_plan.py`.

This is a metadata-only control-plane bridge. It does not read audio bytes, copy
audio, run inference, start training, edit model registries, relax thresholds, or
create listen output.

## Input

The input is a `mac_candidate_readiness_summary.v1` file with:

```text
delivery_allowed=false
listen_result_allowed=false
promising_blocked_handoff.status=promising_but_blocked
promising_blocked_handoff.presentation_status=diagnostic_only_handoff
```

`detail_loss` routes to the documented judge-guided minus air/detail repair
family:

```text
tools/apply_judge_guided_minus_air_detail_repair.py
```

`residual_vocal_leak` remains an explicit missing implementation. The planner
therefore emits:

```text
design_detail_preserving_residual_vocal_cleanup_route
```

instead of pretending a production cleanup route already exists.

## Outputs

The tool writes short Windows-safe files:

```text
queue.json
rows.jsonl
summary.md
progress.jsonl
```

The queue remains blocked/non-deliverable:

```text
queue_source=mac_promising_blocked_handoff
queue_status=ready_for_full_pack_loop
delivery_allowed=false
presentation_status=mac_promising_handoff_queue_not_for_listen_pack
```

The full-pack loop planner turns the row into:

```text
route_intent=mac_promising_blocked_handoff
action_status=blocked_mac_promising_handoff_route_resolution_required
expected_acceleration_mode=external_only_cpu_handoff_planning
```

## Docker-only example

```bash
docker compose run --rm --no-deps \
  -v D:\AIMusicProject\artifacts:/host_artifacts \
  -e REQUIRE_CUDA=0 \
  -e ML_TORCH_DEVICE=cpu \
  ml-worker \
  python tools/plan_mac_promising_blocked_handoff.py \
    --mac-readiness-summary-json /host_artifacts/loop/it013/rd/summary.json \
    --output-dir /host_artifacts/loop/it014/queue
```

Then hand the queue to the loop planner:

```bash
docker compose run --rm --no-deps \
  -v D:\AIMusicProject\artifacts:/host_artifacts \
  -e REQUIRE_CUDA=0 \
  -e ML_TORCH_DEVICE=cpu \
  ml-worker \
  python tools/build_full_pack_loop_iteration_plan.py \
    --repair-queue-manifest-json /host_artifacts/loop/it014/queue/queue.json \
    --training-plan-json /host_artifacts/loop/it014/training/plan.json \
    --output-dir /host_artifacts/loop/it014/plan
```

Both artifacts are planning evidence only and must not be presented as listen
results.
