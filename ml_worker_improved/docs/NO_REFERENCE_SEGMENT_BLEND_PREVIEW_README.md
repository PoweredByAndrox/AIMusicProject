# No-Reference Segment Blend Preview

`tools/render_no_reference_segment_blend_preview.py` renders preview-only audio
for render-ready NR-6 segment blend rows. It does not publish a final result and
does not mark anything release-ready.

## Contract

Inputs:

- `no_reference_segment_blend_dry_run.jsonl`
- `no_reference_segment_blend_render_contract.jsonl`

Outputs:

- `no_reference_segment_blend_preview_manifest.json`
- `no_reference_segment_blend_preview_summary.md`
- preview audio files under `segment_blend_previews/`

Only rows with `render_status=render_ready` are rendered. The default
`--max-render-count` is `1`, so blocked rows and extra render-ready rows are
skipped.

The dry-run stage can explicitly enable two preview-only safety helpers before
rendering:

- `--allow-sample-rate-resample` lets preview rendering resample alternate
  stems to the base stem sample rate. The rendered preview still preserves the
  base sample rate, channel count, and duration, and the manifest records
  `sample_rate_resample_applied=true`.
- `--resolve-overlap-conflicts` selects the highest-improvement non-overlapping
  renderable window while leaving other overlapping windows blocked. Selection
  only considers rows that already pass missing-path, shape, and boundary
  checks.

## Preview Semantics

- The preview is a diagnostic/listening preview, not a final delivery artifact.
- Output paths are created only inside the configured preview folder.
- Original candidate files are never overwritten.
- Sample rate, channel count, and duration are preserved from the base stems.
- Crossfade uses the `crossfade_ms` value from the dry-run contract.
- A render contract row may set `alternate_weight` in `[0, 1]`. The default is
  `1.0`, which replaces the target window with the alternate under the crossfade
  envelope. Values below `1.0` produce a fractional micro-blend, for example
  `0.99` keeps 1% of the base window while using 99% of the alternate. This is
  still preview-only and must pass post-render judge gates before any delivery
  wrapper can accept it.
- `can_publish_as_result=false`.
- `can_mark_release_ready=false`.
- `requires_post_render_judge=true`.

For `source_stem=both`, both vocal and minus previews are blended for the same
window. For single-stem rows, the target stem is blended and the paired stem is
written as a preview pair from the base stem.

## Safety

- No reference vocal/minus stems are required.
- No human CSV is required.
- No model inference is run.
- No training or fine-tune is started.
- No cleanup, debleed, or denoise route is added.
- No thresholds or gates are relaxed.
- The preview is not a release/listen-pack result.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/render_no_reference_segment_blend_preview.py \
  --dry-run-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl \
  --render-contract-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_render_contract.jsonl \
  --output-dir /app/lab_artifacts/no_reference_optimizer_20260531/segment_blend_previews \
  --manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_manifest.json \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_summary.md \
  --max-render-count 1
```

After rendering, run the no-reference judge on the preview candidate and the
base candidate only. Do not promote automatically from the judge result.
