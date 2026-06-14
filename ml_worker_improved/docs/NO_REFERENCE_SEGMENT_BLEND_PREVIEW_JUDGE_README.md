# No-Reference Segment Blend Preview Window Judge

`tools/judge_no_reference_segment_blend_preview.py` verifies rendered segment
blend previews on the exact target window from the dry-run/render contract.

The tool does not render, copy, export, publish, train, or require reference
stems. It reads only the preview/base/alternate/mix audio windows needed for
the no-reference comparison.

## Inputs

- `no_reference_segment_blend_preview_manifest.json`
- `no_reference_segment_blend_render_contract.jsonl`
- `no_reference_segment_blend_dry_run.jsonl`
- `no_reference_windowed_judge_300.jsonl`

The preview manifest provides rendered preview paths. The render contract
provides base and alternate paths plus the exact window. The windowed judge
artifact provides the original mix path for the same track.

## Output

- `no_reference_segment_blend_preview_window_judge.jsonl`
- `no_reference_segment_blend_preview_window_judge_summary.md`

Each row reports:

- base and preview target-window no-reference scores using the row's
  `source_stem` mode (`vocal`, `minus`, or pair compatibility for `both`);
- score and blocker deltas;
- candidate id and path identity flags;
- target-window waveform identity check;
- preview decision;
- `can_publish_as_result=false`;
- `can_mark_release_ready=false`;
- `allowed_next_step`.

## Decisions

- `preview_noop_self_blend`: base/alternate target paths are identical or the
  preview target window is near-identical to the base target window.
- `preview_improved_window`: preview score improves beyond the margin and no
  new blocker appears.
- `preview_regressed_window`: preview score drops beyond the guardrail or a new
  blocker appears.
- `preview_inconclusive`: the score movement is too small or mixed.

Same candidate id is recorded as alias risk but is not enough by itself to
declare a noop. Path identity and target-window waveform identity are checked.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/judge_no_reference_segment_blend_preview.py \
  --preview-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_manifest.json \
  --render-contract-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_render_contract.jsonl \
  --dry-run-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge_summary.md
```

The output is diagnostic preview evidence only. It must not be promoted to
release/listen-pack delivery without a separate delivery gate.
