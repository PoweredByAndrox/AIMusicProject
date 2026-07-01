# Minus Repair Delta Gate

`tools/evaluate_minus_repair_delta_gate.py` compares pre-repair and post-repair
`instrumental_self_judge_gate` manifests for a minus candidate.

This is a Docker-only metadata gate:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_metadata_gate`
- no audio rendering
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

## Inputs

The gate requires:

- `--before-self-judge-manifest`: self-judge manifest for the base candidate;
- `--after-self-judge-manifest`: self-judge manifest after a repair route;
- optional `--repair-manifest`: candidate-only repair manifest;
- `--candidate-id`: provenance label for the evaluated repair.

Both self-judge manifests must point at readable `segments_jsonl` rows from
`instrumental_self_judge_gate`.

## Gate Policy

The gate compares segment-level blocker transitions and issue counts.

It blocks fail-closed with:

- `blocked_missing_evidence` for missing manifests or missing `segments_jsonl`;
- `blocked_corrupt_evidence` for unreadable JSON/JSONL evidence;
- `blocked_no_blocker_reduction` when post-repair `blocker_count` is not lower
  than pre-repair `blocker_count`;
- `blocked_unsafe_risk_increase` when repair reduces blockers but increases
  unsafe vocal/full-mix proxy issues:
  - `vocal_projection_proxy`
  - `vocal_stem_corr_proxy`
  - `full_mix_like_risk_proxy`

Only a strict blocker-count reduction with no unsafe issue increase can produce
`passed_for_candidate_routing`.

## Outputs

The tool writes short Windows-safe artifacts:

```text
gate.json
rows.jsonl
summary.md
progress.jsonl
```

`rows.jsonl` contains per-segment transitions:

- `persisting_blocker`
- `resolved_blocker`
- `new_blocker`
- `clean`

## Delivery Contract

The delta gate never publishes audio:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `presentation_status=delta_gate_not_for_listen_pack`
- `requires_downstream_acceptance=true`

`passed_for_candidate_routing` only authorizes the next candidate-only routing
step. Accepted listen output still requires post-render minus self-judge,
no-reference result judge, selector acceptance, musical-detail preservation, and
the listen-preview artifact status contract.

For blocked gates, pass `gate.json` to
`tools/plan_minus_repair_delta_router.py`. The router decides whether to switch
to leakage repair, retune detail repair, use the segment router, or mine an
alternative model candidate. That router output is also plan-only and cannot be
published as audio.

## Example

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/evaluate_minus_repair_delta_gate.py \
  --before-self-judge-manifest /host_artifacts/loop/it016/self/manifest.json \
  --after-self-judge-manifest /host_artifacts/loop/it017/self/manifest.json \
  --repair-manifest /host_artifacts/loop/it017/cmd/air/manifest.json \
  --output-dir /host_artifacts/loop/it018/gate \
  --candidate-id it017_air \
  --short-output-names
```
