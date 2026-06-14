# No-Reference Vocal Delivery Pipeline

`tools/run_no_reference_vocal_delivery_pipeline.py` runs a vocal-only delivery path from existing no-reference vocal judge rows.

It is mode-specific for `stem_type=vocal`:

1. read existing `run_no_reference_windowed_judge.py --score-mode vocal` rows;
2. reject rows that are not `score_mode=vocal`;
3. accept a row by stem-specific vocal evidence: `vocal_decision=accept_candidate`
   when present, otherwise legacy `decision=accept_candidate`, and always require
   empty `vocal_blockers`;
4. fail closed before selection when the mix or vocal audio path is missing;
5. optionally choose the base candidate from
   `tools/build_no_reference_vocal_candidate_score_matrix.py` via
   `--candidate-score-matrix-json`;
6. otherwise rank accepted vocal-mode rows by `vocal_score` only, with
   `overall_no_reference_score` used only as a missing-score fallback;
7. build `no_reference_vocal_listen_preview_manifest.json` through the artifact status contract.

The listen-preview manifest also requires selector evidence with `score_mode=vocal`; legacy `both` selector JSONL is blocked before listen-pack promotion.

Accepted rendered vocal candidate statuses include:

- `existing_vocal_candidate_proxy_pass`
- `rendered_vocal_segment_switch_candidate`
- `rendered_vocal_source_quiet_silence_repair_candidate`
- `rendered_vocal_judge_guided_silence_repair_candidate`

The rendered repair statuses are provenance only; post-render vocal judge and selector acceptance are
still mandatory before listen-pack promotion.

It does not run separation models, add checkpoints, require reference stems, require human CSV review, train, add cleanup/debleed/denoise stages, or use `minus`/`both`/shared overall scoring as the vocal selector.

When `--candidate-score-matrix-json` is supplied, the pipeline manifest records
`base_selection_source=candidate_score_matrix`. Without it, the fallback source
is `vocal_score_sort`; explicit `--base-candidate-id` records
`base_selection_source=explicit_base_candidate_id`.

Accepted output is listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=vocal`
- `score_modes_used=["vocal"]`
- selector summary uses `selection_metric=vocal_score`
- `can_mark_release_ready=false`

If any step fails, the final pipeline manifest is blocked and includes `dominant_failure_reason`.

Example:

```bash
python tools/run_no_reference_vocal_delivery_pipeline.py \
  --vocal-judge-jsonl /app/lab_artifacts/example/no_reference_windowed_judge_vocal.jsonl \
  --candidate-score-matrix-json /app/lab_artifacts/example/no_reference_vocal_candidate_score_matrix.json \
  --output-dir /app/lab_artifacts/example/vocal_delivery_pipeline
```

Outputs:

- `no_reference_vocal_delivery_candidates.jsonl`
- `vocal_candidate_manifest.json`
- `no_reference_candidate_selector/no_reference_candidate_selection.jsonl`
- `no_reference_vocal_listen_preview_manifest.json`
- `no_reference_vocal_delivery_pipeline_manifest.json`
- matching summary `.md` files
