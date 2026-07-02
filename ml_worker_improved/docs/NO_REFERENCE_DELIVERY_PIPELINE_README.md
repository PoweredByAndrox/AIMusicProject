# No-Reference Delivery Pipeline

`tools/run_no_reference_delivery_pipeline.py` is the dense-gated listen-source
staging runner for existing no-reference optimizer outputs. It does not create
final accepted delivery by itself.

It runs only control/scoring stages:

1. convert accepted optimizer rows into delivery candidate rows;
2. reject candidates whose vocal/minus duration does not match the original mix;
3. run dense no-reference windowed judging for `vocal`, `minus`, and `both`;
4. build `no_reference_dense_delivery_gate_manifest.json`;
5. build a dense-gated `no_reference_listen_preview_manifest.json` source
   manifest for later aggregate validation.

With `--enable-phase-alignment-rescue`, if the first dense gate blocks an
otherwise accepted candidate, the runner may execute the existing bounded
phase/stereo correction route from
`tools/run_no_reference_phase_alignment_optimizer.py`. The corrected candidate
is still preview-only until the runner re-runs dense `vocal`, `minus`, and
`both` judges and builds a promotable listen-preview source manifest from that
post-correction evidence. If the optimizer selects no corrected candidate, the
rescue stops with `no_phase_corrected_delivery_candidate` and does not rejudge
the original row again.

It does not run separation models, add models, copy/export audio, train,
fine-tune, add cleanup/debleed/denoise routes, relax thresholds, or mark
anything `release_ready` or final `accepted_for_listen_pack` delivery.

## Contract

The dense-gated listen-preview source manifest can be promotable only when:

- an optimizer row is already accepted;
- vocal/minus audio duration matches the original mix duration within the
  preflight tolerance;
- dense windowed judges pass for `score_mode=vocal`, `score_mode=minus`, and
  `score_mode=both`;
- dense gate rows carry explicit selected-mode evidence: `vocal_score`,
  `vocal_decision`, and `vocal_blockers` for vocal; `minus_score`,
  `minus_decision`, and `minus_blockers` for minus; and
  `pair_compatibility_score`, `pair_decision`, and `pair_blockers` for both.
  Generic `decision=accept_candidate` or `overall_no_reference_score` alone is
  not sufficient delivery evidence;
- if phase alignment rescue is used, a corrected candidate is selected, the
  selected correction row has
  `requires_post_correction_judge=true` and the post-correction dense judges
  pass;
- the dense delivery gate manifest is promotable by
  `artifact_status_contract.can_promote_to_listen_result()`;
- the listen-preview wrapper points to that dense gate manifest.

The pipeline manifest itself must remain fail-closed for final delivery:

- `final_status=blocked_from_listen_pack`;
- `delivery_allowed=false`;
- `presentation_status=aggregate_machine_gate_required_not_for_listen_pack`;
- `release_status=not_release_ready`;
- `requires_staged_aggregate_machine_gate=true`.

Final user-facing listen delivery requires the staged aggregate validation path
and an accepted `machine_stem_acceptance_gate` manifest. A promotable dense
listen-preview source is only evidence for that later aggregate path.

The runner exits with code `1` when final delivery is still blocked, including
the normal case where a dense listen-preview source was produced but aggregate
validation and the machine gate have not accepted it.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_delivery_pipeline.py \
  --optimizer-result-jsonl /app/lab_artifacts/.../no_reference_optimizer_mode_specific_result.jsonl \
  --output-dir /app/lab_artifacts/.../delivery_pipeline \
  --window-seconds 6 \
  --window-step-seconds 6 \
  --max-windows-per-candidate 0 \
  --min-window-count 5 \
  --enable-phase-alignment-rescue
```

## Outputs

- `no_reference_delivery_candidates.jsonl`
- `no_reference_delivery_rejected_candidates.jsonl`
- `no_reference_dense_vocal.jsonl`
- `no_reference_dense_minus.jsonl`
- `no_reference_dense_both.jsonl`
- `no_reference_dense_delivery_gate_manifest.json`
- `no_reference_listen_preview_manifest.json`
- `no_reference_delivery_pipeline_manifest.json`
- matching summary `.md` files
- optional `phase_alignment_rescue/` artifacts when bounded phase rescue is
  enabled and attempted
