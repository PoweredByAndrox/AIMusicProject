# No-Reference Segment Preview Pair Listen Manifest

`tools/build_no_reference_segment_preview_pair_listen_manifest.py` builds a
listen-preview pair manifest from one accepted segment-preview listen manifest
and one exact `score_mode=both` pair compatibility row.

This is not the independent vocal-minus master contract. It exists for segment
preview evidence where the vocal/minus preview files come from one accepted
segment preview, not from separate mode-specific vocal and minus manifests.

The gate accepts only when:

1. the segment-preview source manifest is promotable by
   `artifact_status_contract.can_promote_to_listen_result()`;
2. the source manifest remains `release_status=listen_preview_only`;
3. source manifest type is `no_reference_segment_preview_listen`;
4. source mix, selected vocal, and selected minus paths are present;
5. `--pair-compatibility-jsonl` contains an exact matching row for the same
   mix/vocal/minus paths;
6. that row has `score_mode=both`, `pair_decision=pair_compatible`,
   empty `pair_blockers`, no blocking semantic labels in generic
   `decision/reasons`, `pair_compatibility_score`, and at least five judged
   windows.

Generic both-mode `decision/reasons` are not vocal/minus selectors, but they
are semantic-clearance evidence. Labels such as `vocal_leakage_in_minus`,
`foreign_source_recovered:*`, residual, leak/bleed, or mix-conservation
blockers fail closed before this pair manifest can become listenable.

Accepted output remains listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=both`
- `accepted_stem_count=2`
- `release_ready=false`
- `can_mark_release_ready=false`

Run:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_segment_preview_pair_listen_manifest.py \
  --segment-preview-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_listen_manifest_v4_feedback_strict.json \
  --pair-compatibility-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_compatibility_v4_feedback_strict.jsonl \
  --output-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_listen_manifest_v4_feedback_strict.json \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_listen_manifest_v4_feedback_strict_summary.md
```
