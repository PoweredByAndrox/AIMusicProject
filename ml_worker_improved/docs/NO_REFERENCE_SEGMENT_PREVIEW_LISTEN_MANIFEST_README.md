# No-Reference Segment Preview Listen Manifest

`tools/build_no_reference_segment_preview_listen_manifest.py` converts
post-render segment blend preview evidence into a listen-preview manifest.

This gate does not render audio, copy audio, train, add models, add cleanup
routes, relax thresholds, or mark anything release-ready.

## Acceptance Contract

A preview row can enter the listen pack only when all checks pass:

- preview audio exists for vocal and minus paths;
- preview row has `render_status=render_ready`;
- preview row has `requires_post_render_judge=true`;
- sample rate, channel count, and duration were preserved;
- render contract row exists and is `render_ready`;
- targeted preview judge row matches the same track/window/candidates/source
  stem and the exact preview vocal/minus paths;
- targeted judge reports `preview_decision=preview_improved_window`;
- `allowed_next_step=expand_preview_policy`;
- `score_delta > 0`;
- `blocker_delta <= 0`;
- all source rows keep publish/release flags false.

Rejected/noop/regressed/inconclusive previews stay blocked and are listed in
`skipped_rows`.

When the targeted preview judge row carries `mix_path` or `source_audio_path`,
the accepted listen manifest preserves that value at top level and inside each
accepted row. This is required provenance for downstream pair compatibility
judging; missing mix provenance is not invented by this tool.

## Output

Accepted output:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `release_ready=false`

Blocked output:

- `delivery_status=blocked`
- `delivery_allowed=false`
- `presentation_status=blocked_from_listen_pack`
- `dominant_failure_reason=no_post_render_improved_preview_rows`

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_segment_preview_listen_manifest.py \
  --preview-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_segment_blend_preview_v4_feedback_strict_manifest.json \
  --preview-window-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_segment_blend_preview_window_judge_v4_feedback_strict.jsonl \
  --render-contract-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_blend_subset_v4_feedback_strict.jsonl \
  --output-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_listen_manifest_v4_feedback_strict.json \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_listen_manifest_v4_feedback_strict_summary.md
```

Next step for pair compatibility is metadata-only:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_segment_preview_pair_candidate.py \
  --segment-preview-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_listen_manifest_v4_feedback_strict.json \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict.jsonl \
  --summary-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict_manifest.json \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict_summary.md
```
