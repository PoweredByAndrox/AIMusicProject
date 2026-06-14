# No-Reference Segment Blend Dry-Run

This tool validates planned no-reference segment blend windows without rendering,
copying, exporting, or publishing audio.

## Contract

`tools/prepare_no_reference_segment_blend_dry_run.py` consumes the NR-5 segment
blend plan plus existing no-reference candidate metadata and writes:

- `no_reference_segment_blend_dry_run.jsonl`
- `no_reference_segment_blend_render_contract.jsonl`
- `no_reference_segment_blend_dry_run_summary.md`

The render contract is approval-only metadata for a later phase. It is not a
listen result and cannot be marked release-ready.

## Validation Per Window

For each planned window the tool records:

- base and alternate vocal/minus paths from the segment blend plan;
- selected target stem: `vocal`, `minus`, or `both`;
- selected alternate source path for the planned window. For `source_stem=both`,
  `selected_source_path_for_window` is a structured object with `vocal` and
  `minus` paths;
- `crossfade_ms`, `pre_roll_ms`, and `post_roll_ms`;
- header-only sample-rate, channel, duration, boundary, and overlap checks;
- `render_status`;
- `can_publish_as_result=false`;
- `can_mark_release_ready=false`.

`render_status` values:

- `render_ready`
- `blocked_incompatible_audio_shape`
- `blocked_overlap_conflict`
- `blocked_missing_source_path`
- `blocked_noop_self_blend`
- `blocked_unsafe_boundary`

`blocked_noop_self_blend` is used when the active replacement stem would render
from the same base and alternate path. For `source_stem=both`, the row remains
renderable when at least one active stem path is distinct; an unchanged paired
stem is kept only as a base preview companion.

## Safety Rules

- No reference stems are required.
- No human CSV is required.
- No model inference is run.
- No fine-tune or training is started.
- No cleanup, debleed, or denoise route is added.
- No thresholds or gates are relaxed.
- No audio is rendered, copied, exported, or published.
- Only JSONL metadata and lightweight audio headers are inspected.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/prepare_no_reference_segment_blend_dry_run.py \
  --segment-blend-plan-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --optimizer-result-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl \
  --render-contract-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_render_contract.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run_summary.md \
  --crossfade-ms 80 \
  --pre-roll-ms 120 \
  --post-roll-ms 120
```

Use the project Docker route for validation and artifact generation.
