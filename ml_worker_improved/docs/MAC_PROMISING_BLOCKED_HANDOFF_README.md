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

`residual_vocal_leak` routes to the documented judge-guided minus vocal-leakage
repair family:

```text
tools/apply_judge_guided_minus_vocal_leakage_repair.py
```

That route creates a candidate-only minus repair from judge/self-judge leakage
windows and still requires post-render minus self-judge, result judge, selector,
and listen-preview gates before any delivery wrapper can accept it.

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
action_status=ready_for_documented_mac_promising_repair_routes
expected_acceleration_mode=external_only_cpu_dsp_repair_then_audit
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

When the full-pack planner reports
`ready_for_documented_mac_promising_repair_routes`, use
`tools/materialize_mac_promising_repair_commands.py` to resolve concrete Docker
commands from the readiness/source manifests. The materializer can prepare the
vocal-leakage repair command from objective guard regions, but it blocks the
air/detail route until a matching `instrumental_self_judge_gate` manifest with
safe windowed detail evidence is supplied.
