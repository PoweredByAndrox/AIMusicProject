# No-Reference Listen Preview Manifest

`tools/build_no_reference_listen_preview_manifest.py` converts an accepted
no-reference dense delivery gate manifest into a contract-safe listen-preview
manifest.

It does not copy audio, export audio, train, add models, add cleanup/debleed,
or mark anything `release_ready`.

## Contract

The generated manifest is listenable only when:

- `--dense-delivery-gate-manifest-json` is provided;
- the dense gate manifest is promotable by
  `artifact_status_contract.can_promote_to_listen_result()`;
- the dense gate source manifest hash matches `--optimizer-result-jsonl`;
- accepted rows include `dense_delivery_gate` evidence for vocal, minus, and
  pair checks.
- `phase_corrected_candidate_selected` rows are accepted only after the dense
  post-correction judge evidence is present and promotable.

Optimizer-only input is blocked with
`dominant_failure_reason=missing_dense_delivery_gate`. This prevents a raw
optimizer decision from bypassing dense `vocal/minus/both` evidence.

The manifest writes:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `can_mark_release_ready=false`

The tool validates the generated manifest through
`artifact_status_contract.can_promote_to_listen_result()`.

## Run

```bash
python tools/build_no_reference_listen_preview_manifest.py \
  --optimizer-result-jsonl /app/lab_artifacts/.../no_reference_optimizer_mode_specific_result.jsonl \
  --dense-delivery-gate-manifest-json /app/lab_artifacts/.../no_reference_dense_delivery_gate_manifest.json \
  --output-manifest-json /app/lab_artifacts/.../no_reference_listen_preview_manifest.json \
  --summary-md /app/lab_artifacts/.../no_reference_listen_preview_summary.md
```
