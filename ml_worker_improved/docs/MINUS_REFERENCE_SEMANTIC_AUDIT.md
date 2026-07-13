# Minus Reference Semantic Audit

`tools/audit_minus_reference_pair_semantics.py` is a fail-closed Docker-only
CPU/DSP audit for verifying that a candidate `minus_reference` bundle is
semantically aligned with its `source_mix`.

## Contract

- Docker execution only.
- Reads existing audio only.
- No model inference.
- No ASR dependence.
- No audio copies or audio exports.
- Writes only:
  - `sem.json`
  - `progress.jsonl`
- Uses `expected_acceleration_mode=external_only_cpu_dsp_scoring`.
- Always records `delivery_allowed=false`.
- `training_allowed=true` only when `decision=accepted`.

## Inputs

Positional arguments:

1. `source_mix_path`
2. `minus_reference_path`
3. `output_dir`

Repeatable detector argument:

```bash
--detector DETECTOR_ID SOURCE_VOCAL_PATH REFERENCE_VOCAL_PATH
```

Each detector contributes:

- one vocal estimate derived from the source mix;
- one vocal estimate derived from the minus reference;
- per-detector `vocal_to_mix_rms_db` evidence after mono conversion and
  sample-rate/length alignment against the corresponding parent audio.

The audit is fail-closed unless at least two unique detector IDs are present.

## Decision Logic

Per detector:

- compute `source_vocal_to_mix_rms_db`
- compute `reference_vocal_to_mix_rms_db`
- compute `source_reference_gap_db = source - reference`

Aggregate:

- `source_median_vocal_to_mix_rms_db`
- `reference_median_vocal_to_mix_rms_db`
- `median_source_reference_gap_db`

Thresholds:

- `min_gap_db` default: `6.0`
- `max_reference_vocal_share_db` default: `-20.0`

Decisions:

- `accepted`
  - median source-reference gap is at least `min_gap_db`
  - reference median is at or below `max_reference_vocal_share_db`
  - at least two unique detector IDs individually support the accepted result
- `roles_reversed`
  - median reference-source gap is at least `min_gap_db`
  - at least two unique detector IDs individually support the reversed result
- `unresolved`
  - detector quorum exists, but the medians do not prove accepted or reversed
- `blocked`
  - structural failure, such as missing inputs, unreadable audio, or fewer than
    two unique detector IDs

## `sem.json` Schema

Schema id:

```json
"minus_reference_semantic_audit.v1"
```

Required integration fields:

- `decision`
  - one of: `accepted`, `roles_reversed`, `unresolved`, `blocked`
- `training_allowed`
- `source_mix`
  - `path`
  - `sha256`
- `minus_reference`
  - `path`
  - `sha256`
- `detector_count`
- `detectors`
- `evidence`
- `thresholds`

Additional required safety fields from this producer:

- `delivery_allowed=false`
- `asr_dependency=false`
- `audio_copied=false`
- `audio_exported=false`
- `release_ready=false`
- `can_publish_as_result=false`

`detectors` contains per-detector evidence:

- `detector_id`
- `support_decision`
- `source_reference_gap_db`
- `source_vocal`
  - `path`
  - `sha256`
  - `sample_rate`
  - `frames_after_resample`
  - `vocal_to_mix_rms_db`
- `reference_vocal`
  - `path`
  - `sha256`
  - `sample_rate`
  - `frames_after_resample`
  - `vocal_to_mix_rms_db`

`evidence` contains aggregate evidence:

- `source_median_vocal_to_mix_rms_db`
- `reference_median_vocal_to_mix_rms_db`
- `median_source_reference_gap_db`
- `accepted_support_detector_ids`
- `roles_reversed_support_detector_ids`
- `unresolved_support_detector_ids`
- `unique_detector_ids`
- `duplicate_detector_ids`

`thresholds` contains:

- `min_gap_db`
- `max_reference_vocal_share_db`

## Run

```bash
docker compose run --rm --no-deps -T ml-worker python tools/audit_minus_reference_pair_semantics.py \
  /app/source_mix.flac \
  /app/minus_reference.flac \
  /app/reference_bundle \
  --detector detector_a /app/detector_a_source_vocal.flac /app/detector_a_reference_vocal.flac \
  --detector detector_b /app/detector_b_source_vocal.flac /app/detector_b_reference_vocal.flac
```

When the output directory is the candidate bundle root, downstream intake can
consume `/app/reference_bundle/sem.json` directly.

## Outputs

- `sem.json`
- `progress.jsonl`

The tool returns exit code `0` only for `accepted`. All other decisions return
non-zero so the route stays fail-closed.
