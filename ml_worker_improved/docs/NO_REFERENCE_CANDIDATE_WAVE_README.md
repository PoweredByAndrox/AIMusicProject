# No-Reference Candidate Wave

`tools/run_no_reference_candidate_wave.py` creates a deterministic
`candidate_pool.jsonl` for the autonomous no-reference runtime. It collects
existing local separator outputs from a source JSONL, copies produced stems into
a run-local candidate folder, and emits only complete vocal+minus pairs to the
runtime pool.

## Command

```bash
docker compose run --rm ml-worker python tools/run_no_reference_candidate_wave.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001/candidates \
  --candidate-pool-jsonl /app/lab_artifacts/autonomous_runs/song_001/candidate_pool.jsonl \
  --profile safe
```

Optional source rows are read from `--source-jsonl` or, by default, from
`candidate_wave_sources.jsonl`, `mock_separator_outputs.jsonl`, or
`separator_outputs.jsonl` inside the wave output directory.

## Contract

- Codex is not required at runtime.
- Reference stems and human CSV labels are not required.
- Audio separators are not trained or fine-tuned.
- No separator models, registry entries, cleanup/debleed/denoise chains, or
  threshold relaxations are added.
- Source audio and source separator outputs are never overwritten.
- Candidate stems are copied into the run-local candidate folder.
- Incomplete or failed source rows are written to
  `candidate_wave_failures.jsonl` with `skip_reason`.
- `release_ready` is always false.

## Source Row Fields

Each source row should include:

```json
{
  "target_mode": "vocal|minus|both",
  "candidate_family": "vocal_focused",
  "candidate_config_id": "safe-vocal",
  "vocal_path": "/app/path/vocals.wav",
  "minus_path": "/app/path/minus.wav",
  "source_tool": "existing_local_separator",
  "source_model_id": "local-model-id"
}
```

`vocal` rows require `vocal_path`, `minus` rows require `minus_path`, and
`both` rows require both. The wave cross-products usable vocal and minus rows
into complete pair candidates.

## Outputs

```text
candidate_pool.jsonl
candidates/candidate_wave_failures.jsonl
candidates/candidate_wave_summary.json
candidates/candidate_wave_manifest.json
```

Every pool row includes `run_id`, `track_id`, `input_audio_path`,
`candidate_id`, `candidate_family`, `candidate_config_id`, `target_mode`,
`vocal_path`, `minus_path`, `can_use_for_vocal`, `can_use_for_minus`,
`can_use_for_pair`, `generation_profile`, `generation_status`, `skip_reason`,
`source_tool`, `source_model_id`, `source_config_hash`, `created_at`, and
`release_ready=false`.
