# No-Reference Full-Track Segment Switch Candidate

`tools/render_no_reference_full_track_segment_switch_candidate.py` renders a
full-duration candidate from existing no-reference vocal/minus pairs.

It differs from the older preview segment-switch tools: it starts from a
full-duration base candidate and overlays only selected safe windows from
alternate existing candidates. Output duration, sample rate, and channels follow
the base candidate.

## Contract

- No reference vocal/minus stems are required.
- No model inference, training, fine-tune, model registry edit, threshold
  relaxation, cleanup, debleed, or denoise route is added.
- The rendered audio is a candidate artifact only:
  `delivery_allowed=false`, `can_mark_release_ready=false`, and
  `requires_dense_post_reconciliation_judge=true`.
- The candidate can enter a listen-preview manifest only after the existing
  dense delivery pipeline accepts it in `vocal`, `minus`, and `both` modes.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/render_no_reference_full_track_segment_switch_candidate.py \
  --candidate-pool-jsonl /app/lab_artifacts/.../no_reference_candidate_pool_nothing_can_wait.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/.../windowed_minus_90_6s.jsonl \
  --base-candidate-id vocal_mdx23c__minus_mdx23c \
  --output-dir /app/lab_artifacts/.../full_track_segment_switch_v1 \
  --manifest-json /app/lab_artifacts/.../full_track_segment_switch_v1/full_track_segment_switch_manifest.json \
  --summary-md /app/lab_artifacts/.../full_track_segment_switch_v1/full_track_segment_switch_summary.md
```

Then run `tools/run_no_reference_delivery_pipeline.py` on
`full_track_segment_switch_candidate_row.jsonl` output. The delivery pipeline
must remain the only path that can create an accepted listen-preview manifest.
