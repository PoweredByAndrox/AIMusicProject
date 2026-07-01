# Mac Residual Pareto Selection

`tools/select_mac_residual_pareto_candidate.py` is a metadata-only selector for
Mac residual refiner runs. It does not copy audio, read audio bytes, run model
inference, start training, change thresholds, create listen packs, or mark any
candidate as delivery-ready.

The selector scores residual manifests by pitch cleanup and instrumental damage
status, then writes:

```text
selection.json
```

The output remains diagnostic:

```text
schema=mac_residual_pareto_selection.v1
delivery_allowed=false
listen_pack_created=false
presentation_status=pareto_selection_diagnostic_only
```

## Residual Outcome Funnel

Each candidate row includes `next_action`:

```text
run_full_holdout_validation
run_damage_audit
run_damage_limited_repair
continue_residual_training_or_rebatch
```

This prevents a pitch-clean batch run from being sent back into another blind
training loop when the correct next gate is damage audit, damage-limited repair,
or full-holdout validation.

`run_full_holdout_validation` is resolved by
`tools/validate_mac_residual_candidate_full_holdout.py` for already-built
residual candidates such as `mac_residual_damage_limited_repair_route.v1`.
This resolver is metadata-only: it consumes the selected candidate manifest and
existing full-holdout evidence manifests, writes
`mac_residual_candidate_full_holdout_validation.v1`, and remains diagnostic:

```text
delivery_allowed=false
listen_pack_created=false
presentation_status=residual_candidate_full_holdout_diagnostic_only
```

It fails closed when holdout evidence is missing, unlinked to the candidate,
overlaps the candidate source guard, fails its own validation, or violates the
no-delivery contract.

`stop_retraining_reason` is populated when pitch has already passed for the
candidate scope:

```text
pitch_pass_waiting_for_damage_audit
pitch_pass_damage_failed
pitch_and_damage_pass_waiting_for_full_holdout
```

## Manifest Glob Input

Residual runs can be scanned directly from short artifact roots:

```bash
docker compose run --rm --no-deps \
  -v D:\AIMusicProject\artifacts:/host_artifacts \
  -e REQUIRE_CUDA=0 \
  -e ML_TORCH_DEVICE=cpu \
  ml-worker \
  python tools/select_mac_residual_pareto_candidate.py \
    --manifest-glob '/host_artifacts/mac_api/it7rr*/r*/manifest.json' \
    --out /host_artifacts/mac_api/it7rsel001
```

The current `it7rr001..it7rr008` checkpoint is blocked:

```text
status=blocked_no_pareto_candidate
candidate_count=8
next_action_counts.continue_residual_training_or_rebatch=3
next_action_counts.run_damage_audit=1
next_action_counts.run_damage_limited_repair=4
delivery_allowed=false
```

No current residual run is a listen result. The top actionable route is to run
the missing damage audit for the pitch-pass/no-damage candidate and damage-limited
repair for pitch-pass/damage-failed candidates before any more same-batch
training.

Current full-holdout resolver checkpoint:

```text
it7rsel002: selected it7dl003_r001 for full-holdout consideration
it7fh001/d003: validation_blocked_no_holdout_evidence
it7fh001/d005: validation_blocked_no_holdout_evidence
it7rd004/sel: blocked_failed_full_holdout_validation
delivery_allowed=false
listen_result_allowed=false
```

This is expected: `it7dl003` and `it7dl005` passed local pitch+damage gates, but
no linked independent full-holdout evidence exists yet.
