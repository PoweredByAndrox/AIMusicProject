# Judge-Guided Minus Vocal Leakage Repair

`tools/apply_judge_guided_minus_vocal_leakage_repair.py` builds a new minus candidate from an existing base minus and a vocal guide, using no-reference windowed judge evidence to choose repair windows.

This route is Docker-only and external-only CPU/DSP:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_scoring`
- no model training
- no model registry edits
- no threshold relaxation
- no per-track hardcoded timestamps

The route can target either source of no-reference leakage evidence:

- dense/windowed judge rows where the selected candidate/path has `vocal_leakage_in_minus_score` above the configured threshold and the window carries `severe_vocal_bleed_in_minus` or `vocal_leakage_in_minus`;
- post-render `instrumental_self_judge_gate` segment JSONL rows for the same `candidate_label` when a segment is a blocker with `vocal_projection_proxy` or `vocal_stem_corr_proxy`;
- a full post-render `instrumental_self_judge_manifest.json`; the route converts its `candidate` and `segments` fields into the same internal judge-row schema before selecting repair windows.

It then subtracts a bounded vocal-basis projection inside a smooth time gate. Self-judge windows are normalized from `window_sec` into `start_sec`/`duration_sec`; non-vocal blockers such as presence/air/RMS issues are not used as vocal-leakage repair targets.

Projection can run in two modes:

- `--projection-mode global` (default) computes one signed projection gain over
  every selected repair window.
- `--projection-mode segment_local` computes a bounded gain per selected window.
  Use this when different windows have different vocal leakage sign/scale and a
  single global coefficient leaves a local self-judge blocker. The output remains
  `candidate_only` and still requires a post-render minus self-judge before any
  delivery wrapper can accept it.

`segment_local` is implemented as bounded window-sliced DSP: audio stays in
`float32`, the source mix is inspected through metadata instead of being decoded
into an unused full-track array, and each gate/filter/projection working set is
limited to the selected window plus deterministic filter context. The manifest
metrics record `memory_strategy=window_sliced_float32`,
`full_track_gate_per_window=false`, and the maximum local working span. This keeps
peak memory proportional to full-track audio plus one local window rather than
full-track length multiplied by selected-window count; it does not change how
judge windows are selected and does not introduce track names or fixed timestamps.

`--max-projection-gain` is a processing bound, not an acceptance gate. The
default remains conservative, but the route allows higher bounded gains for
low-energy vocal guides where the mathematically required projection coefficient
can be greater than `1.0`. High-gain repairs are still candidate-only and must
pass post-render self-judge, result judge, selector, and listen-preview contract;
they must not be presented as listen results directly.

If a high-gain repair creates peaks above full scale, limiting is applied only
inside active repair windows. The route must not globally scale unrelated
sections of the track, because unselected audio is part of the preservation
contract.

The manifest is intentionally candidate-only:

- `delivery_status=candidate_only`
- `delivery_allowed=false`
- `presentation_status=blocked_until_judged`
- `release_status=not_release_ready`
- `requires_post_judge=true`

The generated FLAC is not a listen-pack result and must be re-scored through
the post-render instrumental self-judge, the mode-specific no-reference minus
result judge, and the minus selector before any delivery wrapper can accept it.
`tools/build_no_reference_minus_listen_preview_manifest.py` accepts this
candidate family only when the manifest provenance is
`judge_guided_minus_vocal_leakage_repair`, the candidate row declares
`requires_post_render_minus_self_judge=true`, all post gates pass, the minus
musical-detail preservation gate is present and passing, and candidate/path
evidence matches across every source. If post-judge still blocks the candidate,
the artifact remains diagnostic evidence only.

Example:

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/apply_judge_guided_minus_vocal_leakage_repair.py \
  /host_testing_stuff_music/input.mp3 \
  /app/lab_artifacts/base_minus.flac \
  /app/lab_artifacts/accepted_vocal.wav \
  /app/lab_artifacts/no_reference_windowed_judge_minus.jsonl \
  /app/lab_artifacts/judge_guided_minus_repair \
  --candidate-id vocal_hybrid_htdemucs_ft_bs_roformer_6s__minus_hybrid_non_vocal_sum \
  --projection-mode segment_local
```

For a second-pass repair from post-render self-judge, pass that candidate's
`instrumental_self_judge_manifest.json` or `instrumental_self_judge_segments.jsonl`
as the judge evidence path and use the candidate label/id from the self-judge
manifest.
