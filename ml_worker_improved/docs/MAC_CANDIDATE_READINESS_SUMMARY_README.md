# Mac Candidate Readiness Summary

`tools/summarize_mac_candidate_readiness.py` turns a Mac candidate or selector
manifest into a short machine-readable readiness report.

The tool does not repair audio, run inference, relax thresholds, or promote a
candidate. It only explains whether the artifact can be treated as a listen
result and which gate is blocking it.

Accepted listen output requires all of:

```text
delivery_status=accepted
delivery_allowed=true
presentation_status=accepted_for_listen_pack
```

If an audibility delta audit is supplied, it must also pass. The audit is a
diagnostic metadata gate: it compares source and candidate audio, writes
`mac_candidate_audibility_delta.v1`, and never creates listen output. A failed
audibility audit blocks an otherwise accepted-looking manifest with
`blocked_inaudible_candidate`. For Mac human-listen handoff checks, run the
summary with `--require-audibility-audit`; without an audit manifest it then
fails closed as `blocked_missing_audibility_audit`.

If a human listen feedback manifest is supplied with `--listen-feedback`, it is
treated as diagnostic evidence only. Rejected feedback blocks delivery with
`blocked_human_listen_feedback` even when the source manifest otherwise looks
accepted. Known issue tags are mapped into machine-actionable blockers:

```text
residual_vocal_* / vocal_leak / pitch_shifted_vocal -> human_feedback_residual_vocal_leak
detail_loss / missing_detail / muffled / presence_loss -> human_feedback_detail_loss
```

The corresponding next actions are:

```text
run_detail_preserving_residual_vocal_cleanup
run_judge_guided_minus_air_detail_repair
```

This is the handoff for promising-but-blocked minus candidates: listener notes
about blurred vocal leak or missing detail are preserved in the readiness
summary, but the artifact remains `delivery_allowed=false` until objective,
audibility, holdout, reference, and delivery gates pass.

The same information is also emitted as `promising_blocked_handoff` so a planner
does not need to infer route intent from free-form notes. For `detail_loss`, the
recommended route points to the documented
`tools/apply_judge_guided_minus_air_detail_repair.py` family. For
`residual_vocal_leak`, the handoff currently marks the route as
`missing_implementation` instead of pretending that a production cleanup route
already exists.

If a residual candidate full-holdout validation manifest is supplied with
`--full-holdout-validation`, it must also pass. This evidence is produced by
`tools/validate_mac_residual_candidate_full_holdout.py` and remains
diagnostic-only. A failed validation blocks a selected candidate with
`blocked_failed_full_holdout_validation`; a passed validation only removes the
full-holdout blocker and does not bypass exact-reference or delivery gates.

Diagnostic Mac artifacts remain blocked even when they are the current best
candidate. Typical blockers are:

- `pitch_guard_failed`
- `selection_diagnostic_only`
- `full_holdout_not_evaluated`
- `exact_reference_audit_missing`
- `reference_bundle_mismatched_to_source`
- `diagnostic_presentation_status`
- `audibility_delta_below_gate`

For alternate minus route manifests such as `mac_alt_minus_route_eval.v1`, the
objective summary is read from the nested `guard` block:

```text
guard.score_region_count
guard.after_repair_region_count
guard.after_repair_verdict
guard.blocker_reasons
```

This keeps full-region pitch-vocal leaks visible in readiness reports even when
the route did not use the older flat `before_score` / `after_score` fields.

Example Docker-only run for root `artifacts` manifests:

```bash
docker compose run --rm --no-deps \
  -v D:\AIMusicProject\artifacts:/host_artifacts \
  -e REQUIRE_CUDA=0 \
  -e ML_TORCH_DEVICE=cpu \
  ml-worker \
  python tools/summarize_mac_candidate_readiness.py \
    --manifest /host_artifacts/mac_api/it7sel003/selection.json \
    --audibility-audit /host_artifacts/mac_api/it7aud001/d003/manifest.json \
    --listen-feedback /host_artifacts/loop/it013/fbk.json \
    --full-holdout-validation /host_artifacts/mac_api/it7fh001/d003/manifest.json \
    --require-audibility-audit \
    --out /host_artifacts/mac_api/it7rd001/sel
```

Current checkpoint evidence:

```text
it7cmp001: blocked_failed_objective_gate, listen_result_allowed=false
it7sel003: blocked_needs_full_holdout_and_reference, listen_result_allowed=false
it7rd002/sel: blocked_source_reference_mismatch, listen_result_allowed=false
it7br001/plan: blocked_waiting_for_mac_reference_recovery, delivery_allowed=false
it7rp001: blocked_missing_reference for be-nicegreenhorn, delivery_allowed=false
it7rd003/sel: audibility_audit_status=pass, candidate_ready_for_full_holdout=true, listen_result_allowed=false
it7fh001/d003: validation_blocked_no_holdout_evidence, delivery_allowed=false
it7rd004/sel: blocked_failed_full_holdout_validation, listen_result_allowed=false
```

This summary layer is intended to be used before any human listen handoff so a
diagnostic artifact cannot be mistaken for an accepted production result.

When a wrong reference bundle is supplied, the summary must keep the artifact
blocked and emit `reference_bundle_mismatched_to_source`. The resulting full-pack
loop row uses route intent `mac_selected_candidate_reference_gap` and requires:

```text
reference_path_recovery
verified_reference_corpus_expand
holdout_eval_gate
train_ready_decision_gate
full_pack_gate_rebuild
```

The next metadata-only handoff is `tools/plan_mac_reference_path_recovery.py`,
which converts a blocked Mac readiness summary into the standard
`reference_path_recovery_plan.jsonl` format.

For promising-but-blocked human feedback, use
`tools/plan_mac_promising_blocked_handoff.py`. It converts
`promising_blocked_handoff` into a standard full-pack loop queue with
`route_intent=mac_promising_blocked_handoff`, preserving both documented repair
routes and missing-implementation route gaps while keeping `delivery_allowed`
false.
