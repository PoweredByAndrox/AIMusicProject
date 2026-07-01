# No-Reference Minus Delivery Pipeline

`tools/run_no_reference_minus_delivery_pipeline.py` runs a minus-only delivery path from existing generated artifacts.

It does not run separation models, does not add checkpoints, does not require reference stems, does not require human CSV review, does not train, and does not add cleanup/debleed/denoise stages.

The pipeline is mode-specific for `stem_type=minus`:

1. read existing `instrumental_self_judge_gate.py` manifests;
2. if the selected minus candidate already has zero self-judge blockers, keep the existing minus audio path as a preview candidate without rendering audio;
3. otherwise, render a minus segment-switch candidate only for base blocker windows that have a non-blocked alternate segment;
4. run post-candidate `instrumental_self_judge_gate.py`;
5. if post-candidate self-judge still reports a vocal-leakage proxy blocker
   (`vocal_projection_proxy` or `vocal_stem_corr_proxy`), run the existing
   `judge_guided_minus_vocal_leakage_repair` route, then run self-judge again
   on the repaired candidate;
6. run `evaluate_minus_repair_delta_gate.py` for any repair branch that has
   pre/post self-judge evidence; the repair remains blocked unless
   `blocker_count` strictly decreases and unsafe vocal/full-mix proxy issues do
   not increase;
7. if the delta gate blocks the repair, run
   `plan_minus_repair_delta_router.py` to choose the next candidate-only route
   and prevent repeating the same ineffective repair parameters;
8. run `run_no_reference_result_judge.py` with `score_mode=minus` so each row carries `minus_score`,
   `minus_decision`, `minus_blockers`, and `minus_musical_detail_preservation_gate`;
9. run `run_no_reference_candidate_selector.py` with `score_mode=minus`;
10. build `no_reference_minus_listen_preview_manifest.json` through the artifact status contract.

The listen-preview manifest fails closed if either result-judge evidence or selector evidence is missing
`score_mode=minus`; legacy `both` selector JSONL cannot be promoted as a minus listen-preview result.
It also fails closed if the result judge does not provide a passing musical-detail preservation gate.
That gate is a no-reference proxy guard for music holes, transient loss, spectral damage, phase/stereo
instability, and weak mixture consistency; it is not SDR/SI-SDR and does not mark release readiness.
The listen-preview wrapper also accepts a `judge_guided_minus_vocal_leakage_repair`
candidate-only manifest only after the same post-render self-judge, minus
result-judge, selector, musical-detail gate, candidate-id, and audio-path
evidence all match. The repair manifest itself remains blocked/candidate-only
and cannot be published directly.
Repair branches must also pass `evaluate_minus_repair_delta_gate.py`; a repair
that changes issue mix but does not reduce self-judge blocker count remains
blocked as `blocked_no_blocker_reduction`.
Blocked delta gates should be routed through
`plan_minus_repair_delta_router.py`, which produces a plan-only next route and
keeps `delivery_allowed=false`.
The final pipeline manifest records `repair_manifest_path` only when that
post-render leakage repair branch was actually used, and
`post_candidate_self_judge_manifest` points at the final self-judge evidence
used by the listen-preview wrapper.

Accepted output is listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=minus`
- `score_mode=minus`
- selector summary uses `selection_metric=minus_score`
- `can_mark_release_ready=false`

If any step fails, the final pipeline manifest is blocked and includes `dominant_failure_reason`.

This pipeline does not relax the shared pair/dense delivery gate. Vocal delivery and full vocal+minus delivery still need their own accepted evidence.

The runner can either receive explicit `--self-judge-manifest` inputs or discover them from `--self-judge-root`. When a base candidate is not provided, the preferred production path is to pass `--candidate-score-matrix-json` from `tools/build_no_reference_minus_candidate_score_matrix.py`; the runner then selects the top actionable matrix row that matches an available self-judge candidate. Matrix-selectable actions include `run_no_reference_result_judge_minus`, `run_segment_router_for_self_blockers`, `candidate_ready_for_listen_preview_contract`, `run_self_judge_and_selector_before_delivery_contract`, and `accepted_for_listen_pack_not_release`. Selecting a `run_no_reference_result_judge_minus` row only chooses the base candidate; it does not accept or publish that candidate. The matrix combines result-judge, self-judge, selector, listen-preview, and GPU provenance evidence, so a candidate with an accepted minus result can outrank a lower-blocker self-judge candidate that still fails leakage/detail gates. If the matrix is missing or has no actionable match, the runner falls back to the self-judge heuristic: lowest nonzero blocker count, then blocker ratio, then max segment severity, then candidate id. If all discovered candidates have zero blockers, the runner uses the best existing candidate directly and still requires the post-candidate self-judge/result-judge/selector/listen-preview gates before acceptance.

`--allow-repairable-blocked-alternate` is explicit. It allows the segment-switch
renderer to use an alternate segment that still has only repairable vocal-leakage
proxy blockers, but the candidate remains blocked until the post-render leakage
repair branch and all downstream gates pass. Without this flag, those alternates
stay blocked at render planning time.

The pipeline manifest records `base_selection_source` as one of:

- `candidate_score_matrix`
- `explicit_base_candidate_id`
- `self_judge_heuristic`

`instrumental_self_judge_gate.py` treats high source/candidate similarity as a full-mix risk only when it has no vocal evidence metrics or when the segment also has vocal evidence (`vocal_stem_corr` or `vocal_projection`). This prevents dense instrumental segments from being blocked solely because they correlate with a music-dominated mix, while preserving the fail-closed passthrough guard when no vocal evidence is available.

`run_no_reference_result_judge.py` keeps the same minus blocker threshold for
`vocal_leakage_in_minus_score`, but the weak-vocal-proxy passthrough component is
continuous instead of binary. A true full-mix passthrough (`minus == mix`) still
gets the full passthrough leakage score. A candidate that has measurable
source-minus removal near the passthrough boundary is scored by the amount of
under-removal plus residual leakage instead of being saturated to `1.0` only
because `mix_passthrough_similarity` is high. This keeps low-energy vocal tracks
from being blocked as fake full-mix candidates while preserving the full-mix
guard.

Example:

```bash
python tools/run_no_reference_minus_delivery_pipeline.py \
  --self-judge-manifest /app/path/base/instrumental_self_judge_manifest.json \
  --self-judge-manifest /app/path/alternate/instrumental_self_judge_manifest.json \
  --output-dir /app/lab_artifacts/example/minus_delivery_pipeline \
  --base-candidate-id vocal_hybrid__minus_scnet \
  --segment-sec 6 \
  --crossfade-ms 80
```

Auto-discovery example:

```bash
python tools/run_no_reference_minus_delivery_pipeline.py \
  --self-judge-root /app/lab_artifacts/example/minus_only_self_judge_top5 \
  --candidate-score-matrix-json /app/lab_artifacts/example/no_reference_minus_candidate_score_matrix.json \
  --output-dir /app/lab_artifacts/example/minus_delivery_pipeline \
  --allow-repairable-blocked-alternate \
  --segment-sec 6 \
  --crossfade-ms 80
```
