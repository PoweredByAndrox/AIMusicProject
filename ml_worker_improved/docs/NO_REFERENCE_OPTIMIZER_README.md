# No-Reference Optimizer

This tool optimizes existing generated vocal/minus candidates without reference
stems. It is not a training path and it does not create a new separation model
or cleanup chain.

## Contract

- No reference vocal/minus files are required.
- No human CSV is required.
- No fine-tune, training, dataset build, or model registry edit is performed.
- No cleanup, debleed, denoise, or threshold route is added.
- No diagnostic audio is exported as a final result.
- All candidate changes are bounded to existing generated candidates or
  plan-only segment blend recommendations.
- If a variant does not improve by `min_delta`, the optimizer rolls back to the
  input candidate.
- If a variant introduces a severe blocker or damages the opposite stem, it is
  rejected.
- In `mode_specific` mode, vocal and minus are selected independently from
  separate judge files. The optimizer does not average vocal and minus into one
  selector score; pair compatibility is checked after selection.
- Mode-specific selection is fail-closed. A row is eligible only when the
  selected mode has explicit `{mode}_score`, `{mode}_decision`, and
  `{mode}_blockers` fields. Legacy generic `overall_no_reference_score`,
  `decision`, or empty/missing blocker fallback is not selection evidence.
- If one mode has an accepted candidate and the other mode has only
  non-blocked `select_alternative_candidate` rows, the optimizer may return
  `needs_segment_blend` as a preview-only diagnostic route. This is not a
  listen-pack or release-ready result; `selected_candidate_id` remains empty
  and downstream rendering must stay preview-only with post-render judging.

## Input

The optimizer can consume existing no-reference judge rows:

```bash
python tools/run_no_reference_optimizer.py \
  --judge-jsonl /app/lab_artifacts/no_reference_result_loop/no_reference_result_judge.jsonl \
  --output-dir /app/lab_artifacts/no_reference_optimizer \
  --min-delta 0.03
```

It can also consume the same candidate JSONL/manifests as
`run_no_reference_result_judge.py`; in that mode it first runs the no-reference
judge and then optimizes the judged candidates:

```bash
python tools/run_no_reference_optimizer.py \
  --candidate-jsonl /app/path/to/candidates.jsonl \
  --output-dir /app/lab_artifacts/no_reference_optimizer
```

To make the input candidate explicit, set either:

- `is_input_candidate=true`
- `candidate_role=input`
- `candidate_id=original` or `candidate_id=baseline`

If no marker exists, selection is deterministic by candidate id.

Mode-specific run:

```bash
python tools/run_no_reference_optimizer.py \
  --judge-mode mode_specific \
  --vocal-judge-jsonl /app/path/no_reference_windowed_judge_vocal.jsonl \
  --minus-judge-jsonl /app/path/no_reference_windowed_judge_minus.jsonl \
  --output-jsonl /app/path/no_reference_optimizer_mode_specific_result.jsonl \
  --summary-md /app/path/no_reference_optimizer_mode_specific_summary.md
```

## Fitness

The final fitness uses:

```text
final_fitness =
  0.30 * min(vocal_score, minus_score)
+ 0.25 * artifact_score
+ 0.25 * consistency_score
+ 0.20 * defect_reduction_score
```

Dominant defects include vocal bleed in minus, instrumental bleed in vocal,
vocal body loss, vocal tail loss, robotic or musical noise, silence
hallucination, transient smearing, phase comb filtering, high-band damage, and
mix reconstruction drift.

## Output

The tool writes:

- `no_reference_optimizer_result.jsonl`
- `no_reference_optimizer_summary.md`

Each row records the input candidate, selected candidate, tried/rejected
actions, before/after fitness, final decision, selected existing vocal/minus
paths, rollback reason, and forbidden actions. It does not write audio.

For every non-blocked optimizer row, the output also carries candidate-row
compatible metadata required by downstream no-reference delivery checks:
`candidate_id`, `mix_path`, `vocal_path`, `instrumental_path`, and `minus_path`.
Blocked rows intentionally leave those fields empty so they cannot be judged or
promoted as listenable results. Missing metadata in a non-blocked row is treated
as a blocker by the delivery pipeline rather than as a reason to export a
best-of-bad artifact.

Mode-specific rows additionally include `selected_vocal_candidate_id`,
`selected_minus_candidate_id`, `same_candidate_pair`,
`pair_compatibility_decision`, and candidate-row compatible `mix_path`,
`vocal_path`, and `instrumental_path` fields for follow-up pair compatibility
judging.

When a mode-specific row is rejected because required selected-mode fields are
missing, `actions_rejected[].mode_blockers` records blockers such as
`missing_mode_specific_score:vocal_score`,
`missing_mode_specific_decision:vocal_decision`, or
`missing_mode_specific_blockers:vocal_blockers`.

For preview-only mode-specific rows, `segment_blend_needed=true`,
`segment_blend_plan.status=preview_only_not_executed`, and
`can_publish_as_result=false` / `can_mark_release_ready=false` are preserved by
the downstream preview tools.
