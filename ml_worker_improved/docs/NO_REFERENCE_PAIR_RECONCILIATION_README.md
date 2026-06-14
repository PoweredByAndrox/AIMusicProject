# No-Reference Vocal/Minus Pair Reconciliation

`tools/reconcile_no_reference_vocal_minus_pairs.py` builds candidate rows that
pair a selected no-reference vocal improvement with existing minus candidates
from the artifact pool.

## Contract

- No reference vocal/minus stems are required.
- No human CSV is required.
- No model inference, fine-tune, training, dataset build, or model registry edit
  is performed.
- No cleanup, debleed, denoise, or threshold route is added.
- No audio is copied or exported.
- Rows produced by this tool are not listen results:
  `can_publish_as_result=false`, `can_mark_release_ready=false`, and
  `requires_dense_post_reconciliation_judge=true`.
  They use `final_decision=pair_reconciliation_candidate`; this status is only
  eligible for the dense delivery gate when the explicit dense-post flag is
  present. Without a passing dense vocal/minus/both gate, the listen-preview
  manifest remains blocked.

## Intended Use

Use this route when a local vocal improvement is promising but the paired
compensated minus fails the no-reference gate. The tool creates alternate pair
candidates:

```text
selected vocal preview + existing minus candidate
```

Every generated pair must be rejudged with dense `vocal`, `minus`, and `both`
no-reference windowed judges, then passed through the dense delivery gate before
it can enter a listen-preview manifest.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/reconcile_no_reference_vocal_minus_pairs.py \
  --vocal-candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/nonoverlap_preview_expansion_20260606_v11_window_local_delta_current/phase_alignment_optimizer_seeded_2tracks_max2/no_reference_phase_alignment_optimizer_result.jsonl \
  --minus-candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/nonoverlap_preview_expansion_20260606_v7_seeded_pool/no_reference_candidate_pool_seeded_with_previews_2tracks.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/pair_reconciliation/no_reference_reconciled_pairs.jsonl \
  --skipped-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/pair_reconciliation/skipped_no_reference_reconciled_pairs.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/pair_reconciliation/no_reference_reconciled_pairs_summary.md
```
