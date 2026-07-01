# Minus Delta Model Candidate Materializer

`tools/materialize_minus_delta_model_candidate_route.py` turns a
`minus_repair_delta_router_plan.v1` that selected
`no_reference_minus_candidate_model_router` into registered-model minus
candidate commands.

This is a container-backed metadata materializer:

- `gpu_inference_skipped=true` for the materializer itself;
- generated candidate commands require `gpu_registered_model_inference`;
- no audio is rendered by the materializer;
- no model training, fine-tune, threshold relaxation, or registry edit happens;
- no generated command row is publishable as a listen result.

For Mac targets, the approved container orchestrator is Apple `container`
(`/usr/local/bin/container`), not Docker Desktop, Docker daemon, `docker compose`,
or Docker VMM. The current registered-model command rows are Linux/CUDA
in-container command plans. When the materializer is run as
`native_mac_cpu_dsp`, it blocks with
`command_plan_status=blocked_mac_registered_model_runtime_unavailable` until a
real Mac/Apple-container registered-model adapter exists.

The materializer reads the router failure taxonomy and unresolved issue summary,
expands proxy labels such as `vocal_projection_proxy` into registry labels such
as `vocal_residue_in_minus`, and asks `default_model_registry()` for
minus-capable candidates. Each ready row calls:

```bash
python /app/tools/run_registered_model_candidate.py <source_audio> <output_dir> \
  --model-id <model_id> \
  --device cuda \
  --target-stem minus
```

`<source_audio>` and `<output_dir>` in `in_container_command` must be
container-visible paths. Host provenance remains in `source_audio_path` and
`output_dir`, but command arguments are remapped to container paths such as
`/host_artifacts/...`. If either path cannot be mapped to a known container-visible
prefix, the materializer writes
`command_plan_status=blocked_missing_container_path_mapping` with
`failed_checks=["container_path_mapping_missing"]` instead of emitting a fake-ready
command.

For Mac artifact-loop runs that use `/host_artifacts/...`, the surrounding
Apple-container route must mount the host artifact root at `/host_artifacts`.
The materializer emits inside-container command arrays only; it does not add
container run flags or make host-only `/Users/...` paths visible by itself.

## Required Inputs

- `--router-plan-json`: router plan whose selected route is
  `no_reference_minus_candidate_model_router`;
- `--source-repair-manifest`: failed repair manifest used as source evidence
  for the original source audio path;
- `--output-dir`: short-name command-plan output directory.

## Outputs

The tool writes short Windows-safe artifacts:

```text
plan.json
rows.jsonl
cmds.jsonl
summary.md
progress.jsonl
```

`plan.json` and `cmds.jsonl` keep:

- selected registered model ids;
- source audio provenance;
- container-visible source/output paths;
- `in_container_command` arrays;
- availability checks from the model registry;
- required downstream gates.

## Delivery Contract

The materializer never publishes audio:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `presentation_status=materializer_not_for_listen_pack`
- `requires_post_judge=true`
- `requires_result_judge=true`
- `requires_delta_gate=true`

Generated registered-model candidates must pass the post-render instrumental
self-judge gate, minus repair delta gate, no-reference result judge, selector,
and listen-preview artifact contract before any listen pack can include them.
