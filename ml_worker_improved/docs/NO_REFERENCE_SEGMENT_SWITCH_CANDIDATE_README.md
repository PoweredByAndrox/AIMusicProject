# No-Reference Segment Switch Candidate

`tools/render_no_reference_segment_switch_candidate.py` builds a preview-only minus candidate by switching between existing generated minus candidates on windows that passed no-reference minus scoring.

It does not run models, does not train, does not require stems or reference files, and does not add cleanup/debleed/denoise processing. The output is never release-ready by itself:

- `delivery_allowed=false`
- `can_publish_as_result=false`
- `can_mark_release_ready=false`
- `requires_post_render_judge=true`

The rendered preview must be judged again with the no-reference windowed/dense gate before it can become a listen-pack candidate.

Typical use:

```bash
docker compose run --rm --no-deps ml-worker python tools/render_no_reference_segment_switch_candidate.py \
  --candidate-pool-jsonl /app/lab_artifacts/.../expanded_cross_pair_pool.jsonl \
  --minus-windowed-judge-jsonl /app/lab_artifacts/.../windowed_minus.jsonl \
  --output-dir /app/lab_artifacts/.../segment_switch_candidate \
  --manifest-json /app/lab_artifacts/.../segment_switch_candidate/segment_switch_manifest.json \
  --summary-md /app/lab_artifacts/.../segment_switch_candidate/segment_switch_summary.md
```
