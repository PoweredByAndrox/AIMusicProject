# Short Minus Quality Audit

`tools/build_short_minus_quality_audit.py` audits `minus` items inside a
short-name aggregate listen pack.

The audit does not create, copy, repair, or publish audio. It reads existing
short pack provenance, reuses the no-reference result/windowed minus judges when
mix+vocal+minus evidence is available, or consumes existing source judge JSONL
evidence when a source listen pack already recorded it.

## Contract

- Input is an accepted `aggregate_short_listen_pack_manifest.v1`.
- Only `stem_type=minus` rows are audited.
- Fresh judging requires sibling vocal audio plus original mix evidence from
  source provenance.
- Rows without fresh pair evidence or prior judge JSONL are marked
  `evidence_missing`; they are not accepted by the audit.
- Residual vocal and near-margin music-detail cases are warning-only rows for
  listening focus; warnings do not change release status.
- Writes only short metadata:
  - `audit.json`
  - `rows.jsonl`
  - `rj.jsonl`
  - `wj.jsonl`
  - `cache.jsonl`
  - `summary.md`
  - `progress.jsonl`
- `audited_minus_count` counts minus rows with judge evidence; `accepted_minus_count`
  counts the subset that passed both result and windowed minus gates.
- Keeps `release_ready=false`, `can_publish_as_result=false`, and
  `delivery_allowed=false`.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_minus_quality_audit.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --output-dir /app/lab_artifacts/lp_v45_001/minqa001 \
  --window-seconds 12 \
  --max-windows-per-candidate 5
```

This is a diagnostic/listen-focus audit, not a release gate and not a claim that
the minus is clean ground truth.

## Consume The Audit

Use the existing subset packer with `--minus-audit-json` to build the next
short listen artifact from the audit:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_subset_pack.py \
  --source-manifest-json /app/lab_artifacts/lp_v45_001/pack.json \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001 \
  --alias-prefix g
```

In this mode:

- non-minus rows remain selectable;
- `minus` rows are selected only when `audit_decision=passed`;
- blocked or evidence-missing `minus` rows are written to `excluded.jsonl`;
- missing audit coverage for a selected minus row blocks the subset;
- the output remains listen-preview only and keeps `release_ready=false`.

Passed minus rows can still carry warning reasons such as
`residual_vocal_watch`. Blocked rows can carry hard blockers such as
`minus_detail_gate_blocked`, `transient_preservation_below_detail_gate`, or
`windowed_minus_not_accepted`. To keep both warnings and blockers actionable
without inventing a human rejection verdict, build a machine audit repair queue:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_repair_queue.py \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/mwq001
```

The resulting `queue.json` uses `queue_source=minus_audit_warning` for backward
compatibility and writes:

- `verdict=machine_warning` for passed rows with warnings;
- `verdict=machine_blocked` for blocked audit rows, with `priority=high`.

It is not a human feedback manifest, not a listen result, not repaired audio,
and not `release_ready`.

When building a listen coverage scorecard, pass the same audit with
`--minus-audit-json`. Any non-zero `blocked_minus_count` or evidence-missing
minus count keeps the scorecard blocked even when 6-stem coverage and semantic
proxy review pass:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_listen_coverage_scorecard.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/pack.json \
  --coverage-json /app/lab_artifacts/lp_v45_001/gsub001/coverage.json \
  --goal-audit-json /app/lab_artifacts/lp_v45_001/gsub001/audit/goal_status_audit.json \
  --idea-funnel-json /app/lab_artifacts/lp_v45_001/gsub001/idea/funnel.json \
  --minus-audit-json /app/lab_artifacts/lp_v45_001/minqa001/audit.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/score001
```

The resulting subset pack can be handed to the standard feedback template and
review handoff tools:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_feedback_template.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/pack.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/tpl001
```

```bash
docker compose run --rm --no-deps ml-worker python tools/build_short_listen_review_handoff.py \
  --pack-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/pack.json \
  --feedback-template-manifest-json /app/lab_artifacts/lp_v45_001/gsub001/tpl001/tpl.json \
  --output-dir /app/lab_artifacts/lp_v45_001/gsub001/rh001
```
