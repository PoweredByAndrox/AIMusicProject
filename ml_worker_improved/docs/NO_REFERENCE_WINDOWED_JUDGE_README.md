# No-Reference Windowed Judge

This tool evaluates existing generated vocal/minus candidates with deterministic
audio windows instead of full-track decoding for every candidate. It is a
no-reference route: it does not require stems, reference vocal/minus files, or a
human CSV.

## Contract

- No models or weights are added.
- No fine-tune, training, dataset build, cleanup, debleed, denoise, or threshold
  relaxation is performed.
- No diagnostic audio windows are exported.
- Audio is read only for metric computation; windows are never copied or saved.
- Output rows preserve the top-level no-reference judge schema expected by
  `run_no_reference_optimizer.py`.
- Output rows preserve optional candidate provenance fields such as `stem_type`,
  `route_id`, `route_family`, `model_id`, `vocal_route_id`, and
  `minus_route_id` so corpus route-prior feedback can block repeated bad source
  routes without inferring identity from paths alone.
- Output rows also include `window_metrics` for downstream metadata-only segment
  planning. Each window metric row records `kind`, `start_sec`, `duration_sec`,
  component scores, blocker reasons, and `window_fitness`.
- `window_fitness` is a routing fitness, not a raw component score. When the
  selected `--score-mode` decision is hard-blocked by vocal/minus/pair blockers,
  the window fitness is forced to `0.0` so downstream segment planners cannot
  rank a blocked window as a usable repair target.
- `--score-mode vocal|minus|both` controls the decision matrix. Vocal and minus
  modes evaluate one stem independently; `both` is for pair compatibility, not
  default optimizer selection.
- In `--score-mode minus`, input rows may provide `vocal_companion_path` instead
  of `vocal_path` when the vocal-side audio is only a residual/source-conservation
  companion from an instrumental route. The judge uses it as analysis input but
  preserves it separately in output; it is not promoted to `selected_vocal_path`.
- In `both` mode, the top-level `decision` follows the pair compatibility gate.
  A blocked pair remains `blocked_hard_case` even when the legacy shared
  `overall_no_reference_score` is high, and pair blockers are copied into
  top-level `reasons`.

## Window Strategy

For each candidate, the judge selects up to five deterministic windows:

- `start`
- `middle`
- `tail`
- `high_energy`
- `low_energy`

If the track is shorter than the requested window duration, the full audio is
used as a single window.

For denser inspection, `--window-step-seconds > 0` switches to an opt-in grid.
The grid starts at `0`, advances by the configured step, and includes the tail
window when it is not already aligned. In grid mode,
`--max-windows-per-candidate 0` means no artificial window cap. The default
`--window-step-seconds 0` keeps the original five-window strategy unchanged.

## Cache

`no_reference_metric_cache.jsonl` stores one row per candidate/window. The cache
key includes:

- `audio_path`
- file size
- file mtime
- metric version
- window start
- window duration
- metric name

The streaming output checkpoint key also includes `score_mode`, window
configuration, `window_step_seconds`, `METRIC_VERSION`, and source file
size/mtime. Re-running the same candidate paths with `--score-mode vocal` and
then `--score-mode minus`, or with semantic windows and then dense grid windows,
writes separate output rows instead of silently reusing the wrong decision.

The CLI writes the judge JSONL and cache incrementally after each candidate and
prints one structured progress row per candidate. If a long run is interrupted,
rerunning the same command reuses the persisted metric cache; it does not export
or copy audio windows.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_summary.md \
  --cache-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_metric_cache.jsonl \
  --score-mode vocal \
  --window-seconds 12 \
  --window-step-seconds 0 \
  --max-windows-per-candidate 5
```

Then optimize without re-reading full audio:

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_optimizer.py \
  --judge-mode windowed \
  --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_summary.md
```
