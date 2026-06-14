# No-Reference Candidate Shape Normalization

`tools/normalize_no_reference_candidate_audio.py` is a shape-only normalization
utility for locally generated no-reference separator candidates.

It exists because some separator entrypoints can emit valid audio at a sample
rate or channel count that differs from the input mix. Those candidates should
not be rejected before the no-reference judges if the mismatch is recoverable
by shape-only normalization.

## Allowed Operations

- resample the generated candidate to the input sample rate;
- match generated candidate channels to the input channel count;
- trim or zero-pad tiny duration drift within tolerance;
- write the normalized file under the current run directory;
- preserve the raw generated separator output path in manifests.

## Forbidden Operations

- denoise;
- debleed;
- cleanup chains;
- EQ enhancement;
- loudness mastering;
- threshold relaxation;
- destructive overwrite of the raw separator output;
- marking any artifact `release_ready`.

## Job Manifest Fields

`tools/run_local_separator_candidate_job.py` records the shape contract in
`job_manifest.json`:

```text
input_sample_rate
input_channels
input_duration_sec
raw_output_sample_rate
raw_output_channels
raw_output_duration_sec
normalized_output_sample_rate
normalized_output_channels
normalized_output_duration_sec
shape_normalization_applied
shape_normalization_reason
raw_vocal_path
raw_minus_path
normalized_vocal_path
normalized_minus_path
shape_normalization_outputs
original_files_overwritten=false
release_ready=false
```

When normalization succeeds, `output_paths` points to the normalized candidate
paths for downstream judge/scoring. The raw paths remain in `raw_vocal_path`
and `raw_minus_path`.

## Statuses

Allowed job statuses are:

- `generated`
- `generated_with_shape_normalization`
- `generated_with_late_timeout`
- `partial_output_timeout`
- `failed_timeout_before_any_output`
- `failed_missing_output`
- `failed_invalid_audio_shape_unrecoverable`
- `failed_runtime_error`
- `skipped_unavailable_entrypoint`

`generated_with_shape_normalization` is still only a candidate input for later
no-reference judges. It is not a listen-pack result and cannot become
`release_ready`.

## Runtime Flags

Shape normalization is enabled by default:

```bash
docker compose run --rm ml-worker python tools/run_no_reference_candidate_wave.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001/candidates \
  --candidate-pool-jsonl /app/lab_artifacts/autonomous_runs/song_001/candidate_pool.jsonl \
  --execution-mode auto \
  --allow-shape-normalization true
```

The autonomous controller passes the same flag:

```bash
docker compose run --rm ml-worker python tools/run_autonomous_no_reference_pipeline.py \
  --input-audio /app/input/song.wav \
  --output-dir /app/lab_artifacts/autonomous_runs/song_001 \
  --model-dir /app/lab_artifacts/no_reference_ml_critic/models \
  --profile safe \
  --generate-candidates true \
  --candidate-execution-mode auto \
  --allow-shape-normalization true
```
