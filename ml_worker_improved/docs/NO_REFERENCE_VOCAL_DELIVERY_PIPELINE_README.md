# No-Reference Vocal Delivery Pipeline

`tools/run_no_reference_vocal_delivery_pipeline.py` runs a vocal-only delivery path from existing no-reference vocal judge rows.

It is mode-specific for `stem_type=vocal`:

1. read existing `run_no_reference_windowed_judge.py --score-mode vocal` rows;
2. reject rows that are not `score_mode=vocal`;
3. accept a row only when it has explicit stem-specific vocal evidence:
   `vocal_score`, `vocal_decision=accept_candidate`, `vocal_blockers=[]`,
   and `window_count >= 5`;
4. fail closed before selection when the mix or vocal audio path is missing;
5. optionally choose the base candidate from
   `tools/build_no_reference_vocal_candidate_score_matrix.py` via
   `--candidate-score-matrix-json`;
6. otherwise rank accepted vocal-mode rows by `vocal_score` only;
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
Generic `decision=accept_candidate` or `overall_no_reference_score` is diagnostic context only and cannot substitute for explicit vocal evidence.

When `--candidate-score-matrix-json` is supplied, the pipeline manifest records
`base_selection_source=candidate_score_matrix`. Without it, the fallback source
is `vocal_score_sort`; explicit `--base-candidate-id` records
`base_selection_source=explicit_base_candidate_id`.

Accepted `no_reference_vocal_listen_preview_manifest.json` remains source
evidence only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=vocal`
- `score_modes_used=["vocal"]`
- selector summary uses `selection_metric=vocal_score`
- `can_mark_release_ready=false`

The final `no_reference_vocal_delivery_pipeline_manifest.json` is not a
user-facing listen-pack artifact. Even when the listen-preview source is
promotable, the wrapper fails closed until staged aggregate validation and an
accepted machine stem gate are present:

- `delivery_status=blocked`
- `delivery_allowed=false`
- `presentation_status=aggregate_machine_gate_required_not_for_listen_pack`
- `release_status=not_release_ready`
- `listen_source_promotable=true`
- `requires_staged_aggregate_machine_gate=true`
- `final_delivery_blockers=["requires_staged_aggregate_validation_and_machine_gate"]`

If any earlier step fails, the final pipeline manifest is blocked and includes
`dominant_failure_reason` for that earlier failure.

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
