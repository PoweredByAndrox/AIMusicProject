# No-Reference Stem Candidate Score Matrix

`tools/build_no_reference_stem_candidate_score_matrix.py` builds a proxy score
matrix for accepted non-vocal stem listen-preview manifests.

Supported stems:

- `melody`
- `drums`
- `bass`
- `other`

The matrix consumes `no_reference_stem_listen_preview_manifest.json` artifacts
and optionally `no_reference_stem_semantic_judge.jsonl`, then ranks them by
contract-safe proxy evidence:

- listen-preview status from `artifact_status_contract`;
- audio integrity gate status and metrics;
- GPU provenance gate status;
- source contract gate status;
- source safety flags;
- Docker-visible selected audio path existence.
- stem semantic judge decision, score, and blockers when
  `--semantic-judge-jsonl` is provided.

## Contract

- This route does not run model inference.
- This route does not copy or export audio.
- This route blocks `vocal`, `minus`, and `both`; those stems have separate
  mode-specific contracts.
- Accepted rows keep `release_ready=false`.
- Without a semantic judge JSONL, rows keep `semantic_quality_verified=false`.
- With a semantic judge JSONL, rows with semantic blockers are marked
  `do_not_release_semantic_judge_blocked` and cannot remain selected by proxy
  audio/GPU gates alone.
- Output is a proxy ranking only, not a semantic quality proof and not a release
  decision.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_stem_candidate_score_matrix.py \
  --listen-manifest-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612 \
  --semantic-judge-jsonl /app/lab_artifacts/testing_stuff_music_track2_nonvocal_stem_semantic_judge_20260612/no_reference_stem_semantic_judge.jsonl \
  --output-root /app/lab_artifacts/testing_stuff_music_track2_nonvocal_stem_score_matrix_20260612
```

Outputs:

- `no_reference_stem_candidate_score_matrix.json`
- `no_reference_stem_candidate_score_matrix.jsonl`
- `no_reference_stem_candidate_score_matrix_summary.md`

## Validation

```bash
docker compose run --rm --no-deps ml-worker python -m pytest \
  /app/tests/test_build_no_reference_stem_candidate_score_matrix.py \
  -q
```
