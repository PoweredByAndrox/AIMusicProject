# Minus Delta Leakage Repair Materializer

`tools/materialize_minus_delta_leakage_repair.py` turns a
`minus_repair_delta_router_plan.v1` leakage decision into a candidate-only
command for `tools/apply_judge_guided_minus_vocal_leakage_repair.py`.

This tool is metadata-only and Docker-only:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_metadata_planning`
- no audio rendering
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

## Inputs

The materializer requires:

- `--router-plan-json`: output from `plan_minus_repair_delta_router.py`;
- `--source-repair-manifest`: manifest for the candidate being repaired;
- `--output-dir`: short-name artifact directory;
- optional `--candidate-id`: provenance override.

The router must select `judge_guided_minus_vocal_leakage_repair`. The linked
delta gate must point at readable `rows.jsonl`, and the delta gate's
`after_self_judge_manifest` must point at readable self-judge `segments_jsonl`.

## Window Policy

The materializer does not invent timestamps. It selects only delta rows where:

- `transition` is `persisting_blocker` or `new_blocker`;
- `after_blocker=true`;
- `after_issues` contains `vocal_projection_proxy`,
  `vocal_stem_corr_proxy`, `full_mix_like_risk_proxy`, or
  `vocal_leakage_in_minus`.

Selected delta rows are joined back to post-repair self-judge rows by
`segment_id` to recover `window_sec`, metrics, severities, and provenance.
Missing timing or missing joined segments blocks fail-closed.

## Outputs

The tool writes short Windows-safe artifacts:

```text
plan.json
rows.jsonl
cmds.jsonl
summary.md
progress.jsonl
```

`rows.jsonl` contains a `windowed_minus_judge_row.v1` wrapper with
`window_metrics` for the selected leakage windows. `cmds.jsonl` contains the
candidate-only Docker command for the existing leakage repair route.

## Delivery Contract

The materializer never publishes audio:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `presentation_status=materializer_not_for_listen_pack`
- `execution_started=false`
- `requires_post_judge=true`
- `requires_delta_gate=true`

The generated command may create a `candidate_only` repair artifact, but that
artifact must pass post-render self-judge, delta gate, result judge, selector,
musical-detail preservation, and listen-preview status gates before any listen
pack can accept it.

## Example

```bash
docker compose run --rm --no-deps -v D:\AIMusicProject\artifacts:/host_artifacts ml-worker \
  python tools/materialize_minus_delta_leakage_repair.py \
  --router-plan-json /host_artifacts/loop/it019/router/plan.json \
  --source-repair-manifest /host_artifacts/loop/it017/cmd/air/manifest.json \
  --output-dir /host_artifacts/loop/it020/lkgcmd \
  --candidate-id it017_air \
  --short-output-names
```
