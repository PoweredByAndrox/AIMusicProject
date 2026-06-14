# Judge-Guided Vocal Silence Hallucination Repair

`tools/apply_judge_guided_vocal_silence_hallucination_repair.py` builds a new vocal candidate from
an existing vocal candidate, source mix, accepted minus path, and `run_no_reference_windowed_judge.py`
vocal-mode rows.

This is a Docker-only external CPU/DSP route:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_dsp_repair`
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

## Window Policy

The route selects only windows for the requested candidate/path where the vocal judge reports
`silence_hallucination` or `silence_hallucination_score >= --hallucination-threshold`.

Timings come from judge rows (`start_sec` and `duration_sec`). The production route must not encode
track names or fixed timestamps.

Inside each selected judge window, repair is applied at local 50 ms subwindow granularity. The route
attenuates only subwindows where the source mix is locally quiet, matching the no-reference judge's
quiet-region evidence instead of muting the entire judge window. For exact judge-bin repair use:

```bash
--window-margin-sec 0.0 --gate-smoothing-sec 0.0
```

## Delivery Contract

The manifest is a candidate source, not a delivery result:

- `render_status=rendered_vocal_judge_guided_silence_repair_candidate`
- `delivery_status=blocked`
- `delivery_allowed=false`
- `presentation_status=preview_only_not_for_listen_pack`
- `release_status=not_release_ready`
- `requires_post_render_vocal_judge=true`

The generated vocal candidate must pass post-render vocal judge, selector, and listen-preview
contract before it can enter a listen pack.

## Example

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/apply_judge_guided_vocal_silence_hallucination_repair.py \
  /host_testing_stuff_music/input.mp3 \
  /app/lab_artifacts/base_vocal.wav \
  /app/lab_artifacts/accepted_minus.flac \
  /app/lab_artifacts/vocal_judge/no_reference_vocal_judge.jsonl \
  /app/lab_artifacts/vocal_silence_repair \
  --candidate-id base_vocal_candidate \
  --track-id track_id \
  --window-margin-sec 0.0 \
  --gate-smoothing-sec 0.0
```
