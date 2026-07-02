# No-Reference Listen Preview Manifest

`tools/build_no_reference_listen_preview_manifest.py` converts an accepted
no-reference dense delivery gate manifest into a contract-safe listen-preview
source manifest.

It does not copy audio, export audio, train, add models, add cleanup/debleed,
or mark anything `release_ready`.

## Contract

The generated manifest is an intermediate listen-preview source. It is
promotable by the artifact status contract only when:

- `--dense-delivery-gate-manifest-json` is provided;
- the dense gate manifest is promotable by
  `artifact_status_contract.can_promote_to_listen_result()`;
- the dense gate and generated manifest contain no blocking failure labels such
  as `foreign_source_recovered:*`, `mix_conservation_failed`, vocal leak/bleed,
  residual, separator-server, delivery-pair, or GPU-memory blockers;
- the dense gate source manifest hash matches `--optimizer-result-jsonl`;
- accepted rows include `dense_delivery_gate` evidence for vocal, minus, and
  pair checks;
- embedded dense gate evidence is schema-closed: vocal rows must include
  `vocal_score`, `vocal_decision=accept_candidate`, and empty
  `vocal_blockers`; minus rows must include `minus_score`,
  `minus_decision=accept_candidate`, and empty `minus_blockers`; pair rows
  must include `pair_compatibility_score`, `pair_decision=pair_compatible`,
  and empty `pair_blockers`; every embedded mode must also keep
  `window_count >= max(min_window_count, 5)`.
- `phase_corrected_candidate_selected` rows are accepted only after the dense
  post-correction judge evidence is present and promotable.

Optimizer-only input is blocked with
`dominant_failure_reason=missing_dense_delivery_gate`. This prevents a raw
optimizer decision from bypassing dense `vocal/minus/both` evidence.
Legacy or forged dense gate manifests that only contain generic
`decision/window_count` evidence are blocked with
`dominant_failure_reason=dense_delivery_gate_missing_mode_specific_evidence`.
Optimizer metadata such as `pair_compatibility_decision` and generic
`overall_no_reference_score` is not accepted as a substitute for embedded
dense pair evidence.

The manifest writes:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `can_mark_release_ready=false`

The tool validates the generated manifest through
`artifact_status_contract.can_promote_to_listen_result()`.

That validation is a source-manifest contract, not final delivery permission.
Final user-facing listen delivery still requires staged aggregate validation
plus an accepted `machine_stem_acceptance_gate` manifest. Pipeline runners that
wrap this manifest must not copy its `delivery_allowed=true` status into a final
delivery status without those later gates.

## Run

```bash
python tools/build_no_reference_listen_preview_manifest.py \
  --optimizer-result-jsonl /app/lab_artifacts/.../no_reference_optimizer_mode_specific_result.jsonl \
  --dense-delivery-gate-manifest-json /app/lab_artifacts/.../no_reference_dense_delivery_gate_manifest.json \
  --output-manifest-json /app/lab_artifacts/.../no_reference_listen_preview_manifest.json \
  --summary-md /app/lab_artifacts/.../no_reference_listen_preview_summary.md
```
