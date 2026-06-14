# No-Reference Minus Segment Switch Candidate

`tools/render_no_reference_minus_segment_switch_candidate.py` builds a minus-only candidate from existing generated artifacts. It does not run models, does not require reference stems, and does not add cleanup/debleed/denoise stages.

The tool reads `instrumental_self_judge_gate.py` manifests, finds blocker segments in the chosen base minus candidate, and replaces only those blocker windows with a non-blocked alternate minus segment from another existing candidate. The base vocal path is preserved as provenance/proxy context; no vocal audio is rendered by this tool.

Output is preview/candidate only:

- `delivery_allowed=false`
- `presentation_status=preview_only_not_for_listen_pack`
- `release_status=not_release_ready`
- `requires_post_render_minus_self_judge=true`
- `audio_exported_as_result=false`

The rendered minus cannot be published as a listen/result artifact until a later accepted delivery gate explicitly sets `delivery_status=accepted`, `delivery_allowed=true`, and `presentation_status=accepted_for_listen_pack`.

Example:

```bash
python tools/render_no_reference_minus_segment_switch_candidate.py \
  --self-judge-manifest /app/path/base/instrumental_self_judge_manifest.json \
  --self-judge-manifest /app/path/alt/instrumental_self_judge_manifest.json \
  --output-dir /app/lab_artifacts/example/minus_segment_switch \
  --manifest-json /app/lab_artifacts/example/minus_segment_switch/manifest.json \
  --summary-md /app/lab_artifacts/example/minus_segment_switch/summary.md \
  --base-candidate-id vocal_hybrid__minus_scnet
```
