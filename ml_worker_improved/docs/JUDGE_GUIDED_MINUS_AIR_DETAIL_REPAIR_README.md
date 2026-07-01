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
