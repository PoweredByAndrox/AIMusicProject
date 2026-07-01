# No-Reference Vocal Listen Preview

`tools/build_no_reference_vocal_listen_preview_manifest.py` creates a contract-safe listen-preview manifest for a vocal stem only.

It does not copy audio, export audio, run models, train, add cleanup/debleed/denoise, relax gates, or mark anything `release_ready`.

The manifest can be accepted only when:

1. the candidate manifest is preview-only and has `render_status=existing_vocal_candidate_proxy_pass` or `render_status=rendered_vocal_segment_switch_candidate`;
2. the candidate manifest requires post-candidate vocal judging;
3. the vocal judge row has `score_mode=vocal`, explicit stem-specific vocal
   evidence (`vocal_score`, `vocal_decision=accept_candidate`, `vocal_blockers=[]`,
   and `window_count >= 5`), and no vocal blockers;
4. the selector accepts the same candidate;
5. every evidence path points to the same vocal path;
6. the candidate manifest does not report any source safety violation such as training, model changes, cleanup route additions, threshold relaxation, or audio export/copy.

Preview-only source status is strict: the source candidate must have `delivery_status=blocked`, `delivery_allowed=false`, `presentation_status=preview_only_not_for_listen_pack`, and `release_status=not_release_ready` before post-candidate judging promotes it to listen-preview.

Accepted output is listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=vocal`
- `can_mark_release_ready=false`

This does not relax the shared pair dense delivery gate. Full vocal+minus delivery still requires separate accepted vocal and minus evidence.
Generic `decision=accept_candidate` or `overall_no_reference_score` is diagnostic context only and cannot substitute for explicit vocal evidence.

Example:

```bash
python tools/build_no_reference_vocal_listen_preview_manifest.py \
  --candidate-manifest-json /app/lab_artifacts/example/vocal_candidate_manifest.json \
  --vocal-judge-jsonl /app/lab_artifacts/example/no_reference_dense_vocal.jsonl \
  --selector-jsonl /app/lab_artifacts/example/no_reference_candidate_selector/no_reference_candidate_selection.jsonl \
  --output-manifest-json /app/lab_artifacts/example/no_reference_vocal_listen_preview_manifest.json \
  --summary-md /app/lab_artifacts/example/no_reference_vocal_listen_preview_summary.md
```
