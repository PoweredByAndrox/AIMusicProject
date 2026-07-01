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
