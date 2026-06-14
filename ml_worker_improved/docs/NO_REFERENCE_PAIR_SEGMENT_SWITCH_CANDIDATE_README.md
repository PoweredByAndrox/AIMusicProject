# No-Reference Pair Segment Switch Candidate

`tools/render_no_reference_pair_segment_switch_candidate.py` renders a preview-only vocal/minus pair by switching both stems from the same already-generated candidate on each safe window.

It does not run models, train, copy reference stems, add cleanup/debleed/denoise, or relax gates. The output is always blocked from listen/release until a post-render judge accepts it through the normal artifact contract.

The selector is mode-aware:

- default selection uses `minus_score`;
- vocal exploration can use `--selection-score-field vocal_score --selection-decision-field vocal_decision`;
- `--allow-low-activity-proxy-gap` can fill a preview window only when the window score passes, the only blocker is `vocal_missing_on_active_region`, and the window reasons are limited to `no_reference_proxy_passed`;
- `--ignore-severe-reasons-for-selection` is allowed only for preview exploration when the selected artifact is still gated by a post-render judge.

Outputs:

- `pair_segment_switch_candidate_row.jsonl`
- `pair_segment_switch_manifest.json`
- `segment_switch_vocal_preview.flac`
- `segment_switch_minus_preview.flac`

Safety contract:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `can_mark_release_ready=false`
- `requires_post_render_judge=true`
