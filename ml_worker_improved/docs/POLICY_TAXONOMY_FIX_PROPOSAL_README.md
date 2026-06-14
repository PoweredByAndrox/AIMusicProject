# Policy Taxonomy Fix Proposal

`tools/propose_policy_taxonomy_fixes.py` reads `root_cause_plan.jsonl` and creates an auditable
proposal for rows that are already classified as policy/taxonomy problems.

This phase does not edit `tools/adaptive_policy.py`, does not edit
`tools/autonomous_audio_judge.py`, and does not apply policy changes automatically.

## Inputs

- `root_cause_plan.jsonl`
- `root_cause_plan_manifest.json`

Only rows where `can_change_policy=true` or `safe_next_action=improve_taxonomy_mapping` are
included.

## Outputs

- `policy_taxonomy_fix_proposal.jsonl`
- `policy_taxonomy_fix_summary.md`
- `policy_taxonomy_fix_manifest.json`
- `proposed_tests_patch.md`

Each proposal row contains:

- `case_id`
- `track_hash`
- `stem_type`
- `split`
- `dominant_failure_reason`
- `failure_labels`
- `current_root_cause_bucket`
- `current_safe_next_action`
- `suspected_policy_gap`
- `proposed_taxonomy_mapping`
- `proposed_policy_change_type`
- `risk_level`
- `requires_human_approval=true`
- `can_apply_automatically=false`
- `proposed_test_name`
- `expected_behavior_after_fix`

## Safety Contract

- No audio is copied or exported.
- No training or fine-tune run starts.
- No policy code is changed.
- No model, weight, cleanup/debleed/denoise route, or threshold is added.
- Holdout rows may receive a proposal, but are marked protected and cannot be used for tuning.
- Minus rows remain minus-scoped and are not proposed as vocal taxonomy fixes.
- Unknown reasons are clarified as evidence/taxonomy problems and are not routed to automatic
  training.

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/propose_policy_taxonomy_fixes.py \
  --root-cause-plan-jsonl /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan.jsonl \
  --root-cause-manifest-json /app/lab_artifacts/control_plane_phase8_20260528/root_cause_plan_manifest.json \
  --output-dir /app/lab_artifacts/control_plane_phase9a_20260528
```
