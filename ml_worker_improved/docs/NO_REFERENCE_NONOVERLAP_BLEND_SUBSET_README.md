# No-Reference Non-Overlap Blend Subset

`tools/plan_no_reference_nonoverlap_blend_subset.py` selects a deterministic,
non-overlapping subset from the existing no-reference segment blend plan.

This phase is metadata-only. It does not render audio, export audio, require
reference stems, train, add models, add cleanup routes, or mark anything as
release-ready.

## Inputs

- `no_reference_segment_blend_plan.jsonl`
- `no_reference_segment_blend_dry_run.jsonl`
- one or more `no_reference_segment_blend_preview_window_judge*.jsonl`
- one or more `no_reference_windowed_judge*.jsonl` files for baseline score
  matrix rows and pair feedback rows
- `no_reference_candidate_pool_300.jsonl`

Expansion is allowed only when the preview window judge contains at least one:

```text
preview_decision=preview_improved_window
allowed_next_step=expand_preview_policy
```

When preview feedback exists for a specific track/window/candidate/path tuple,
that feedback is enforced per row:

- `preview_improved_window` with `expand_preview_policy` keeps the row eligible;
- `preview_regressed_window`, `preview_noop_self_blend`, and inconclusive or
  unsafe preview rows reject only the matching row;
- matching includes base/alternate vocal and minus paths, so duplicate
  candidate IDs with different stem files are not collapsed.

When exact-path feedback is missing, regressed/noop feedback for the same
track/window/source-stem/base-candidate/alternate-candidate identity blocks
unjudged path variants. This prevents repeatedly rendering the same candidate
identity after a post-render judge has already shown that candidate/window is
unsafe or a no-op.

The planner also consumes windowed judge rows with `score_mode=both` as pair
feedback when those rows preserve
`track_id/base_candidate_id/alternate_candidate_id/source_stem/window_*`.
If a segment preview later fails pair compatibility, the same candidate/window
identity is rejected with `selection_reason=pair_feedback=<pair_decision>`.
This prevents repeatedly expanding a locally improved vocal/minus window that
is not compatible as a pair.

The same `--windowed-judge-jsonl` input is also used as a mode-specific score
matrix by `candidate_id`. If the alternate candidate already has a severe minus
blocker from `no_reference_score_matrix.detect_blockers(..., "minus")`, the row
is rejected with `selection_reason=matrix_minus_guard=<blocker>`. This prevents
choosing a vocal-window improvement whose paired minus is already known to have
severe vocal bleed.

## Selection Policy

Rows are rejected when preview policy is not met, when the dry-run reports a
safety blocker, when matching preview feedback rejected that exact row, when
the original planner blocker check failed, or when the `source_stem` does not
match the metric that improved. Rows are also rejected before rendering when the
target stem path is identical between base and alternate:

- `source_stem=vocal` requires different base/alternate vocal paths;
- `source_stem=minus` requires different base/alternate minus paths;
- `source_stem=both` requires both target stem paths to differ.

For remaining rows, the planner greedily selects the highest
`expected_improvement_delta` rows while preventing overlap on the same
`track_id` and target stem after `pre_roll_ms` / `post_roll_ms` margins.
`source_stem=both` conflicts with both vocal and minus windows.

Every output row is one of:

- `selected_nonoverlap`
- `rejected_lower_score_overlap`
- `rejected_safety_blocker`
- `rejected_preview_policy_not_met`

`rejected_preview_policy_not_met` includes both global preview-policy failures
  and row-local preview feedback failures. Row-local feedback failures are
reported with `selection_reason=preview_feedback=<decision>` or
`selection_reason=pair_feedback=<pair_decision>`. Matrix guard failures are
reported with `selection_reason=matrix_minus_guard=<blocker>`.

## Output

- `no_reference_nonoverlap_blend_subset.jsonl`
- `no_reference_nonoverlap_blend_subset_summary.md`

Every row keeps:

- track and candidate IDs;
- source stem and window bounds;
- base/alternate vocal and minus paths required by the preview renderer;
- `render_status` / `can_render_later`;
- `crossfade_ms`, `pre_roll_ms`, and `post_roll_ms`;
- expected improvement delta;
- overlap group and group rank;
- conflict resolution status;
- `can_render_preview_later`;
- `can_publish_as_result=false`;
- `can_mark_release_ready=false`;
- forbidden actions.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/plan_no_reference_nonoverlap_blend_subset.py \
  --segment-blend-plan-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl \
  --dry-run-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl \
  --preview-window-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl \
  --preview-window-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge_next_iteration.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_compatibility_v5_pair_feedback.jsonl \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_blend_subset.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_blend_subset_summary.md
```

The selected subset is approval metadata and a preview render contract for the
existing renderer. Only rows with `conflict_resolution_status=selected_nonoverlap`
are emitted as `render_status=render_ready`; rejected rows are emitted with
non-renderable `render_status` values. Rendered audio remains preview-only and
must not be published as a final or release-ready result.
