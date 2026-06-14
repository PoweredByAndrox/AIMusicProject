# No-Reference Minus Candidate Pool

This tool collects existing generated instrumental/minus artifacts for
no-reference pair reconciliation. It does not generate audio.

## Contract

- No reference vocal/minus stems are required.
- No human CSV is required.
- No models, weights, training, fine-tune, cleanup, debleed, denoise, or
  threshold changes are added.
- No audio is read, copied, hashed, rendered, or exported.
- Only existing `instrumental` / `minus` audio paths already present in manifests
  or supplied seed candidate pools are emitted.
- Stem-only manifests such as `bass.flac + drums.flac + vocals.flac` are skipped
  with `stem_only_requires_render_route`; this collector does not sum stems.
- Rows are compatible with `tools/reconcile_no_reference_vocal_minus_pairs.py`:
  `minus_path` and `instrumental_path` point to the same existing artifact,
  while `vocal_path` is intentionally empty.

## Run

```bash
python tools/collect_no_reference_minus_candidate_pool.py \
  --lab-artifacts-root /app/lab_artifacts \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_minus_candidate_pool.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_minus_candidate_pool_summary.md \
  --skipped-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/skipped_no_reference_minus_candidates.jsonl
```

Existing complete candidate pools can be supplied with `--candidate-pool-jsonl`.
Those rows are used only as existing minus candidates and are deduplicated by
`track_id + mix_path + minus_path`.

## Outputs

- `no_reference_minus_candidate_pool.jsonl`
- `no_reference_minus_candidate_pool_summary.md`
- `skipped_no_reference_minus_candidates.jsonl`

Every complete row includes:

- `track_id`
- `candidate_id`
- `mix_path`
- `vocal_path`
- `minus_path`
- `instrumental_path`
- `selected_minus_path`
- `source_manifest`
- `candidate_manifest`
- `candidate_family`
- `candidate_origin`
- `score_hints`
- `is_complete_pair`
- `is_complete_minus_candidate`
- `skip_reason`

`is_complete_pair=true` is retained only for compatibility with the existing
pair reconciliation tool. It does not mean a vocal artifact is present in this
minus-only pool.
