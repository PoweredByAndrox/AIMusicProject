# Mac Promising Repair Command Materializer

`tools/materialize_mac_promising_repair_commands.py` turns a
`full_pack_loop_iteration_plan.v1` row with
`route_intent=mac_promising_blocked_handoff` into concrete route commands where
the required runtime paths can be recovered from source manifests.

This is a Docker-only metadata/control-plane bridge. It does not publish audio,
does not mark candidates accepted, does not start training, and does not relax
thresholds.

## Inputs

The input is the full-pack plan from
`tools/build_full_pack_loop_iteration_plan.py`:

```text
plan_status=ready_for_documented_mac_promising_repair_routes
delivery_allowed=false
```

The materializer follows each row's `readiness_summary_json` to the source Mac
manifest, then resolves:

```text
candidate.raw_ref_path
stem_paths.minus or stem_paths.instrumental
stem_paths.vocals or stem_paths.vocal
guard.score_path -> guard/score/regions.jsonl
```

For source manifests that do not have `stem_paths`, the materializer keeps the
legacy compatibility fallback:

```text
candidate.minus_path / candidate.instrumental_path
candidate.vocals_path / candidate.vocal_path
```

When `stem_paths` is present, it is the role source of truth. This prevents a
route-specific `candidate.instrumental_path` or `candidate.vocals_path` field
from overriding the packaged product roles used by downstream repair commands.

## Route Handling

`judge_guided_minus_vocal_leakage_repair` can be materialized when the source
manifest has the three audio paths and pitch-guard regions. The tool writes a
short generated judge input:

```text
leak/rows.jsonl
```

The generated rows are provenance-safe conversions of objective guard regions
into `vocal_leakage_in_minus` window rows. They are not human timestamps or
track-specific hardcodes.

`judge_guided_minus_air_detail_repair` can be materialized when
`--minus-self-judge-manifest` points at an `instrumental_self_judge_gate`
manifest whose `segments_jsonl` contains safe detail blocker windows. The
materializer writes:

```text
dtl/rows.jsonl
```

Only detail issues are kept:

```text
air_loss_proxy
presence_loss_proxy
low_rms_muffling_proxy
```

Rows with unsafe vocal/full-mix issues are counted and excluded. If multiple
self-judge manifests match the same source mix, the detail route is blocked as
ambiguous instead of silently choosing one. Global human feedback such as
`detail_loss` is not enough to select repair windows safely.

## Outputs

The tool writes short Windows-safe files:

```text
cmds.json
rows.jsonl
summary.md
progress.jsonl
leak/rows.jsonl
dtl/rows.jsonl
```

`cmds.json` keeps delivery blocked:

```text
delivery_allowed=false
presentation_status=command_plan_not_for_listen_pack
execution_started=false
```

If one route is ready and another route is missing inputs, the status is:

```text
command_plan_status=partial_ready_missing_inputs
```

Ready leakage and detail commands still create only candidate-only repairs. They
use `--short-output-names` and must pass post-render minus self-judge, result
judge, selector, and listen-preview contracts before any accepted delivery
wrapper can use them.
