# No-Reference Segment Preview Pair Candidate

`tools/build_no_reference_segment_preview_pair_candidate.py` converts an
accepted segment-preview listen manifest into candidate rows for
`run_no_reference_windowed_judge.py --score-mode both`.

It is a provenance bridge only. It does not render audio, copy audio, run
models, train, add cleanup/debleed/denoise, relax thresholds, or mark anything
`release_ready`.

The bridge writes candidate rows only when:

1. the source segment-preview manifest is promotable by
   `artifact_status_contract.can_promote_to_listen_result()`;
2. the source manifest remains `release_status=listen_preview_only`;
3. source manifest type is `no_reference_segment_preview_listen`;
4. each candidate row has source `mix_path` or `source_audio_path`;
5. selected vocal and minus preview paths are present and exist;
6. source preview judge evidence remains `preview_improved_window`,
   `allowed_next_step=expand_preview_policy`, and `score_delta > 0`;
7. no source safety flag indicates training, model changes, cleanup route
   additions, threshold relaxation, audio copy/export, or `release_ready`.

Ready output is not a listen result:

- `delivery_status=ready_for_pair_judge`
- `delivery_allowed=false`
- `presentation_status=candidate_rows_ready_for_pair_judge`
- `release_status=listen_preview_only`
- `stem_type=both`
- `release_ready=false`

Run:

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_segment_preview_pair_candidate.py \
  --segment-preview-manifest-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_listen_manifest_v4_feedback_strict.json \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict.jsonl \
  --summary-json /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict_manifest.json \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict_summary.md
```

Then run dense pair compatibility:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_candidate_v4_feedback_strict.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_compatibility_v4_feedback_strict.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_compatibility_v4_feedback_strict_summary.md \
  --cache-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_preview_pair_compatibility_cache_v4_feedback_strict.jsonl \
  --score-mode both \
  --window-seconds 12 \
  --max-windows-per-candidate 5
```
