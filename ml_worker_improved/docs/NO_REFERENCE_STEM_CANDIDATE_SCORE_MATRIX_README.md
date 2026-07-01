# No-Reference Stem Candidate Score Matrix

`tools/build_no_reference_stem_candidate_score_matrix.py` builds a proxy score
matrix for non-vocal stem manifests after audio/provenance/source gates and,
for presemantic inputs, semantic judge clearance.

Supported stems:

- `melody`
- `drums`
- `bass`
- `other`

The matrix consumes `no_reference_stem_listen_preview_manifest.json` artifacts
and optionally `no_reference_stem_semantic_judge.jsonl`, then ranks them by
contract-safe proxy evidence:

- listen-preview status from `artifact_status_contract`;
- explicit presemantic judge-input status
  (`presentation_status=semantic_judge_input_not_for_listen_pack`);
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
- Presemantic input-only manifests are not accepted by proxy gates alone.
- Without a semantic judge JSONL, presemantic rows keep
  `semantic_quality_verified=false` and remain blocked from listen packs.
- With a semantic judge JSONL, rows with `decision=accept_candidate`,
  `semantic_quality_verified=true`, no semantic blockers, and passed
  audio/provenance/source gates can become matrix rows with
  `delivery_status=accepted`, `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack`; the source presemantic
  manifest remains non-presentable.
- With a semantic judge JSONL, rows with semantic blockers are marked
  `do_not_release_semantic_judge_blocked` and cannot remain selected by proxy
  audio/GPU gates alone.
- Output is a proxy ranking/listen-pack selection surface only, not a
  release-ready decision.

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
