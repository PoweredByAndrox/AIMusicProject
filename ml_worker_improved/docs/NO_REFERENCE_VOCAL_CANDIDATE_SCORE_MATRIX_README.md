# No-Reference Vocal Candidate Score Matrix

`tools/build_no_reference_vocal_candidate_score_matrix.py` builds a ranked
control-plane matrix from existing vocal-mode evidence.

It reads:

- vocal-mode judge JSONL rows (`score_mode=vocal`)
- optional `no_reference_candidate_selection.jsonl`
- optional `no_reference_vocal_listen_preview_manifest.json`
- readable candidate `stem_manifest.json` files when rows expose
  `candidate_manifest`

It does not read audio, copy audio, export audio, run models, train, fine-tune,
edit the registry, relax thresholds, or mark anything `release_ready`.

## Outputs

- `no_reference_vocal_candidate_score_matrix.json`
- `no_reference_vocal_candidate_score_matrix.jsonl`
- `no_reference_vocal_candidate_score_matrix_summary.md`

Each row includes:

- `rank`
- `canonical_candidate_id`
- `aliases`
- `vocal_judge.accepted`
- `vocal_judge.vocal_score`
- `selector.accepted`
- `gpu_provenance`
- `selection_priority`
- `next_action`
- `delivery_allowed=false` unless an accepted listen-preview manifest is supplied
- `release_ready=false`

## Next Actions

The matrix chooses route actions, not release status:

- `accepted_for_listen_pack_not_release`
- `candidate_ready_for_listen_preview_contract`
- `run_selector_and_listen_preview_contract`
- `do_not_release_select_cleaner_vocal_or_new_route`
- `run_no_reference_result_judge_vocal`

## Provenance Note

Rows without `candidate_manifest` can still be ranked by vocal judge and selector
evidence, but `has_gpu_proof=false` because the matrix cannot inspect the
candidate execution contract. Production candidate generation should preserve
`candidate_manifest` in vocal judge rows so GPU provenance remains visible.

## Docker Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_vocal_candidate_score_matrix.py \
  --vocal-judge-jsonl /app/lab_artifacts/example/no_reference_vocal_judge.jsonl \
  --selector-jsonl /app/lab_artifacts/example/no_reference_candidate_selection.jsonl \
  --listen-manifest /app/lab_artifacts/example/no_reference_vocal_listen_preview_manifest.json \
  --output-dir /app/lab_artifacts/example/vocal_candidate_score_matrix
```

The matrix can be passed to the vocal delivery runner:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_vocal_delivery_pipeline.py \
  --vocal-judge-jsonl /app/lab_artifacts/example/no_reference_vocal_judge.jsonl \
  --candidate-score-matrix-json /app/lab_artifacts/example/vocal_candidate_score_matrix/no_reference_vocal_candidate_score_matrix.json \
  --output-dir /app/lab_artifacts/example/vocal_delivery_pipeline
```
