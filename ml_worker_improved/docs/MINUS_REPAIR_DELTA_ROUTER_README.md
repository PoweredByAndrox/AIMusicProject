# Minus Repair Delta Router

`tools/plan_minus_repair_delta_router.py` turns a
`minus_repair_delta_gate.v1` manifest into a candidate-only next-route plan.

This is a container-backed metadata planner:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_metadata_planning`
- no audio rendering
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

## Inputs

The planner requires:

- `--delta-gate-manifest`: output from `evaluate_minus_repair_delta_gate.py`;
- `--output-dir`: short-name artifact directory;
- optional `--source-route`: route that produced the failed repair. When this is
  omitted, the router attempts to infer the source route from the delta gate's
  `repair_manifest`;
- optional `--candidate-id`: provenance override.

The delta-gate manifest must point at readable `rows.jsonl` segment delta rows.

## Routing Policy

The planner reads unresolved segment transitions and classifies persistent
blockers:

- unsafe vocal/full-mix proxy blockers route to
  `switch_to_leakage_repair` / `judge_guided_minus_vocal_leakage_repair`,
  unless that same leakage route already produced zero blocker reduction;
- failed leakage repairs with zero blocker reduction route to
  `route_to_alternative_model_candidate` /
  `no_reference_minus_candidate_model_router`;
- detail-only blockers route to `retune_detail_repair` and block repeating the
  same detail repair parameters;
- unclassified persistent blockers route to `switch_to_segment_router`;
- a passed delta gate routes only to downstream acceptance gates.

The planner blocks repeating the same repair route when the delta gate reports
zero blocker reduction.

When the selected route is `judge_guided_minus_vocal_leakage_repair`, pass the
router plan to `tools/materialize_minus_delta_leakage_repair.py`. That
materializer converts persistent unsafe delta rows into windowed judge rows by
joining the delta evidence with the post-repair self-judge segments. It is the
safe bridge from plan-only routing to the existing candidate-only leakage repair
CLI.

When the selected route is `no_reference_minus_candidate_model_router`, pass the
router plan and the failed repair manifest to
`tools/materialize_minus_delta_model_candidate_route.py`. That materializer
does not run inference or publish audio; it mines registry-backed minus-capable
model candidates and writes `in_container_command` arrays for
`/app/tools/run_registered_model_candidate.py --device cuda --target-stem minus`.
Those generated commands are ready only when source/output paths are
container-visible, for example under `/host_artifacts/...`; otherwise the
materializer blocks with `container_path_mapping_missing`.

For Mac targets, route orchestration must use Apple `container`
(`/usr/local/bin/container`). Docker Desktop, Docker daemon, `docker compose`,
and Docker VMM are not approved Mac project execution routes. The registered
CUDA command-plan path is blocked under `native_mac_cpu_dsp` until a real
Mac/Apple-container registered-model adapter is implemented. Every generated
model candidate remains blocked for listening until it reruns the instrumental
self-judge gate, the minus repair delta gate, the no-reference result judge, and
the selector.

Example:

```bash
/usr/local/bin/container run --rm <project-image> python /app/tools/materialize_minus_delta_model_candidate_route.py \
  --router-plan-json /host_artifacts/loop/it029/router/plan.json \
  --source-repair-manifest /host_artifacts/loop/it028/lkg/manifest.json \
  --output-dir /host_artifacts/loop/it030/mdlcmd \
  --max-candidates 3 \
  --short-output-names
```

## Outputs

The tool writes short Windows-safe artifacts:

```text
plan.json
rows.jsonl
summary.md
progress.jsonl
```

`plan.json` includes:

- `router_status`
- `selected_next_action`
- `selected_route`
- `blocked_repeat_routes`
- `failure_taxonomy`
- `required_actions`
- `issue_summary`

## Delivery Contract

The router never publishes audio:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `presentation_status=router_plan_not_for_listen_pack`
- `requires_downstream_acceptance=true`

It only plans the next candidate-only route. Any generated candidate must rerun
self-judge, delta gate, result judge, selector, and listen-preview status gates
before listen-pack acceptance.

## Example

```bash
/usr/local/bin/container run --rm <project-image> python /app/tools/plan_minus_repair_delta_router.py \
  --delta-gate-manifest /host_artifacts/loop/it018/gate/gate.json \
  --output-dir /host_artifacts/loop/it019/router \
  --source-route judge_guided_minus_air_detail_repair \
  --candidate-id it017_air \
  --short-output-names
```
