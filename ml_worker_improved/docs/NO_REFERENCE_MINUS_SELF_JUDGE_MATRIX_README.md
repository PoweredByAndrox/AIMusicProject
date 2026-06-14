# No-Reference Minus Self-Judge Matrix

`tools/run_no_reference_minus_self_judge_matrix.py` runs the existing
`instrumental_self_judge_gate` over a JSONL pool of minus candidates and writes a
deterministic matrix of delivery-gate eligibility.

This is a diagnostic/control-plane route. It does not render audio, train models,
change thresholds, copy artifacts, or mark release/listen delivery.

## Inputs

Each candidate row should include:

- `track_id`
- `candidate_id`
- `mix_path` or `source_audio_path`
- `instrumental_path`, `minus_path`, or `selected_minus_path`
- optional `vocal_path` or `selected_vocal_path`

`mix_path` and minus path are required. Vocal path is optional; when absent, the
self-judge still checks music-preservation proxies but cannot compute vocal-stem
correlation/projection proxies.

## Output Files

- `no_reference_minus_self_judge_matrix.jsonl`
- `no_reference_minus_self_judge_matrix_manifest.json`
- `no_reference_minus_self_judge_matrix_summary.md`
- `no_reference_minus_self_judge_matrix_progress.jsonl`

Rows with `self_judge_status=proxy_pass` and `blocker_count=0` get
`listen_candidate_eligible=true`. All other rows stay blocked.

The underlying self-judge also blocks full-mix passthrough risk. In addition to
high RMS-ratio risk, it records `source_candidate_corr_abs` and
`source_candidate_delta_rms_ratio`; a candidate that is almost identical to the
source mix with too little source-minus delta is blocked as
`full_mix_like_risk_proxy`, even when it is slightly quieter than the source.
This prevents a contaminated full-mix-like minus from being accepted as a clean
instrumental.

## Safety Contract

- `reference_required=false`
- `human_csv_required=false`
- `training_started=false`
- `fine_tune_started=false`
- `models_changed=false`
- `model_registry_edited=false`
- `cleanup_routes_added=false`
- `thresholds_relaxed=false`
- `delivery_allowed=false`
- `release_ready=false`

## Docker Example

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/run_no_reference_minus_self_judge_matrix.py \
  --candidate-jsonl /app/lab_artifacts/example/minus_candidates.jsonl \
  --output-root /app/lab_artifacts/example/minus_self_judge_matrix \
  --track-id example_track_id \
  --minus-path-contains route_or_artifact_substring \
  --segment-sec 6
```

Omit `--track-id` to evaluate every row in the input JSONL. Use it for targeted
corpus debugging when a full corpus self-judge pass would be too expensive.
Use `--minus-path-contains` to re-check one route/source path without rebuilding
or re-running the full matrix.
