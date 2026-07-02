# No-Reference Delivery Batch

`tools/run_no_reference_delivery_batch.py` runs the dense-gated
no-reference delivery pipeline over multiple existing optimizer outputs.

It does not run separation models, train, fine-tune, copy/export audio, add
cleanup/debleed/denoise routes, relax thresholds, or mark outputs
`release_ready`.

Each optimizer output is processed by:

1. `tools/run_no_reference_delivery_pipeline.py`
2. dense `score_mode=vocal` judging
3. dense `score_mode=minus` judging
4. dense `score_mode=both` judging
5. dense delivery gate
6. listen-preview source manifest wrapper

When `--enable-phase-alignment-rescue` is supplied, each per-track pipeline may
also run the bounded phase/stereo correction route and then re-run dense
post-correction judging before creating a promotable listen-preview source
manifest.

Blocked rows remain blocked. The batch command exits successfully when it
produces a report; source-preview readiness is represented per row in the
manifest. Final accepted delivery still requires staged aggregate validation
plus an accepted `machine_stem_acceptance_gate` manifest.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_delivery_batch.py \
  --optimizer-result-jsonl /app/lab_artifacts/track_a/no_reference_optimizer_mode_specific_result.jsonl \
  --optimizer-result-jsonl /app/lab_artifacts/track_b/no_reference_optimizer_result.jsonl \
  --output-root /app/lab_artifacts/no_reference_delivery_batch \
  --window-seconds 6 \
  --window-step-seconds 6 \
  --max-windows-per-candidate 0 \
  --min-window-count 5 \
  --enable-phase-alignment-rescue
```

## Outputs

- `no_reference_delivery_batch_rows.jsonl`
- `no_reference_delivery_batch_manifest.json`
- `no_reference_delivery_batch_summary.md`
- one subdirectory per optimizer input with the single-track delivery outputs
