# No-Reference Minus Candidate Score Matrix

`tools/build_no_reference_minus_candidate_score_matrix.py` builds a ranked
control-plane matrix from existing minus evidence.

It reads:

- `no_reference_result_judge.jsonl`
- `instrumental_self_judge_manifest.json`
- `no_reference_candidate_selection.jsonl`
- optional `no_reference_minus_listen_preview_manifest.json`
- readable candidate `stem_manifest.json` files for GPU provenance

It does not read audio, copy audio, export audio, run models, train, fine-tune,
edit the registry, relax thresholds, or mark anything `release_ready`.

## Outputs

- `no_reference_minus_candidate_score_matrix.json`
- `no_reference_minus_candidate_score_matrix.jsonl`
- `no_reference_minus_candidate_score_matrix_summary.md`

Each row includes:

- `rank`
- `canonical_candidate_id`
- `aliases`
- `result_judge.accepted`
- `result_judge.minus_score`
- `self_judge.status`
- `self_judge.blocker_count`
- `selector.accepted`
- `gpu_provenance`
- `selection_priority`
- `next_action`
- `delivery_allowed=false` unless an accepted listen-preview manifest is supplied
- `release_ready=false`

Rows are ranked fail-closed. A result-judge `accept_candidate` does not override
self-judge blockers. Among candidates with the same action priority, rows with
fewer `self_judge.blocker_count` values rank ahead of rows with higher proxy
minus score. This prevents a louder or higher-scoring proxy candidate with many
audible blocker windows from outranking a cleaner candidate that is closer to
passing the artifact contract.

## Next Actions

The matrix chooses route actions, not release status:

- `run_segment_router_for_self_blockers`
- `candidate_ready_for_listen_preview_contract`
- `accepted_for_listen_pack_not_release`
- `do_not_release_consider_leakage_repair_or_new_model_route`
- `do_not_release_music_detail_regressed_choose_different_candidate`
- `blocked_generate_or_select_better_minus_candidate`

## Docker Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_minus_candidate_score_matrix.py \
  --result-judge-jsonl /app/lab_artifacts/example/no_reference_result_judge.jsonl \
  --self-judge-root /app/lab_artifacts/example/self_judges \
  --selector-jsonl /app/lab_artifacts/example/no_reference_candidate_selection.jsonl \
  --output-dir /app/lab_artifacts/example/minus_candidate_score_matrix
```

The matrix can be passed to the minus delivery runner:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_minus_delivery_pipeline.py \
  --self-judge-root /app/lab_artifacts/example/self_judges \
  --candidate-score-matrix-json /app/lab_artifacts/example/minus_candidate_score_matrix/no_reference_minus_candidate_score_matrix.json \
  --output-dir /app/lab_artifacts/example/minus_delivery_pipeline
```
