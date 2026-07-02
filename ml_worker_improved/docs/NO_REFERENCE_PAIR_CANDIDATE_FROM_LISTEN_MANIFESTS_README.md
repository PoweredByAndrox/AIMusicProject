# No-Reference Pair Candidate From Listen Manifests

`tools/build_no_reference_pair_candidate_from_listen_manifests.py` builds a single pair-compatibility candidate row from one accepted vocal listen-preview manifest and one accepted minus listen-preview manifest.

It is a metadata/provenance bridge only. It does not copy audio, export audio, run models, train, add cleanup/debleed/denoise, relax gates, or mark anything `release_ready`.

The bridge writes candidate rows only when:

1. the vocal source manifest is promotable by `artifact_status_contract.can_promote_to_listen_result()`;
2. the minus source manifest is promotable by `artifact_status_contract.can_promote_to_listen_result()`;
3. source stem types are exactly `vocal` and `minus`;
4. both source manifests remain `release_status=listen_preview_only`;
5. both manifests use the same `track_id`, or they have different legacy `track_id` conventions but the same non-empty source `mix_path`;
6. selected vocal and minus paths are present;
7. neither source manifest reports a safety violation such as training, model changes, cleanup route additions, threshold relaxation, audio copy/export, or `release_ready`.

Source promotability includes recursive semantic blocker detection. Nested
generic fields such as `reasons`, `decision`, and `pair_score_mode_reasons`
or failure-class fields cannot carry labels like `vocal_leakage_in_minus`,
`foreign_source_recovered:*`, or `mix_conservation_failed`; such sources are
blocked before pair-candidate rows are written.

Blocked sources produce an empty JSONL plus a summary manifest with:

- `delivery_status=blocked`
- `delivery_allowed=false`
- `presentation_status=blocked_from_pair_judge`
- `dominant_failure_reason=<reason>`
- `candidate_count=0`

Ready output is not a listen result. It is only input for dense pair judging:

- `delivery_status=ready_for_pair_judge`
- `delivery_allowed=false`
- `presentation_status=candidate_rows_ready_for_pair_judge`
- `release_status=listen_preview_only`
- `stem_type=both`
- `candidate_count=1`
- `release_ready=false`
- `can_mark_release_ready=false`

Next step after a ready row:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/example/pair_candidate_rows.jsonl \
  --output-jsonl /app/lab_artifacts/example/no_reference_pair_compatibility.jsonl \
  --summary-md /app/lab_artifacts/example/no_reference_pair_compatibility_summary.md \
  --cache-jsonl /app/lab_artifacts/example/no_reference_pair_compatibility_cache.jsonl \
  --score-mode both \
  --window-seconds 12 \
  --max-windows-per-candidate 5
```

Only an accepted `score_mode=both` row from that judge can be passed to `tools/build_no_reference_vocal_minus_master_manifest.py`.
