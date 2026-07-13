# Judge-Guided Minus Air/Detail Repair

`tools/apply_judge_guided_minus_air_detail_repair.py` builds a new minus candidate from an existing
base minus and source mix, using `instrumental_self_judge_gate` rows to select windows where the
minus lost musical air, presence, or RMS detail.

This is a Docker-only external CPU/DSP route:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_dsp_repair`
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

## Window Policy

The route selects only self-judge blocker rows for the requested candidate/path when they contain at
least one detail issue:

- `air_loss_proxy`
- `presence_loss_proxy`
- `low_rms_muffling_proxy`

It rejects windows that also contain unsafe evidence:

- `vocal_projection_proxy`
- `vocal_stem_corr_proxy`
- `full_mix_like_risk_proxy`

Window timings come from `window_sec`/`start_sec` in the judge rows. The route must not encode
track names or fixed timestamps in production logic.

## Repair

The repair adds bounded band-limited `source_mix - base_minus` detail inside a smooth time gate:

- body band: low RMS support;
- presence band: instrumental presence recovery;
- air band: high-frequency detail recovery.

If `--vocal-guide-path` is supplied, the added detail layer is projected against the vocal guide and
suppressed before writing the candidate. This is a guardrail, not an acceptance gate.

Vocal suppression supports two modes:

- `--vocal-suppression-mode global` (default) uses one bounded projection over all selected windows;
- `--vocal-suppression-mode segment_local` computes a bounded projection independently for each
  selected judge window, preventing opposite local gains from cancelling each other.

Both modes apply suppression through the smooth repair gate, so unselected audio is not modified.
The manifest records the selected mode and per-window projection metrics. Segment-local mode is the
preferred retune when a global repair reduces aggregate projection but introduces a local vocal
proxy blocker.

## Output Names

Use `--short-output-names` for generated loop/listen/debug artifacts. In that
mode the route writes:

```text
m001.flac
manifest.json
rows.jsonl
```

The manifest records `short_output_names=true` and an
`artifact_name_policy` block with the short filenames and provenance status.

## Delivery Contract

The manifest is intentionally candidate-only:

- `render_status=rendered_minus_air_detail_repair_candidate`
- `delivery_status=candidate_only`
- `delivery_allowed=false`
- `presentation_status=blocked_until_judged`
- `release_status=not_release_ready`
- `requires_post_judge=true`

The generated FLAC must pass post-render minus self-judge, result judge, selector, and listen-preview
contract before it can be included in any listen pack. The listen-preview builder accepts this
render status only as proof that audio was rendered; it does not make the repair manifest itself a
delivery artifact.
After post-render self-judge, compare pre/post evidence with
`tools/evaluate_minus_repair_delta_gate.py`. If blocker count does not strictly
decrease, unsafe vocal/full-mix proxy issues increase, or any post-repair
self-judge blocker remains, the repair remains blocked and cannot advance to
listen-preview evaluation.

## Independent Non-Vocal Support

When an independent 6-stem lane is available, its non-vocal sum may be passed to
`instrumental_self_judge_gate.py --diagnostic-non-vocal-support <path>`. This support audio is
diagnostic evidence only and must never be exported or presented as the production minus.

The self-judge may explain only a `presence_loss_proxy`, and only when the candidate closely matches
the support in correlation, RMS, and presence while the separate vocal correlation and projection
checks remain safe. The manifest records the support path, SHA-256, and
`diagnostic_non_vocal_support_is_production_minus=false`. The self-judge remains non-deliverable;
normal result-judge, selector, and listen-preview gates still apply.

## Example

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/apply_judge_guided_minus_air_detail_repair.py \
  /host_testing_stuff_music/input.mp3 \
  /app/lab_artifacts/base_minus.flac \
  /app/lab_artifacts/post_repair_self_judge/instrumental_self_judge_segments.jsonl \
  /app/lab_artifacts/minus_air_detail_repair \
  --candidate-id base_minus__judge_guided_leakage_repair \
  --vocal-guide-path /app/lab_artifacts/accepted_vocal.flac
```
