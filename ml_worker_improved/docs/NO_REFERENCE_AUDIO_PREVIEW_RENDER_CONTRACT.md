# No-Reference Audio Preview Render Contract

PREVIEW-1 separates selected pair acceptance from actual listenable preview
audio. A runtime may accept a vocal/minus pair without producing audio. It may
only report `audio_preview_ready` after bounded preview files exist and validate.

## Status Taxonomy

- `accepted_pair_ready`: selected vocal/minus pair is pair-compatible.
- `manifest_preview_ready`: preview manifest exists, but no validated audio
  preview exists.
- `audio_preview_ready`: vocal and minus preview audio were rendered and
  validated inside the run directory.
- `preview_render_skipped`: render was explicitly disabled; `skip_reason` is
  required.
- `preview_render_failed`: render was attempted and failed; `failure_reason` is
  required.
- `preview_blocked_safety_contract`: selected pair or safety contract prevents
  rendering.

`preview_ready` is legacy wording and must not be used to imply audio exists
unless `rendered_preview_audio` has validated files.

## Runtime Outputs

```text
preview_manifest.json
rendered_preview_audio.jsonl
preview_render_contract.json
preview_render_summary.md
```

`preview_manifest.json` includes:

- `run_id`
- `track_id`
- `preview_status`
- `accepted_pair_ready`
- `manifest_preview_ready`
- `audio_preview_ready`
- `preview_render_attempted`
- `preview_render_skipped`
- `preview_render_failed`
- selected vocal/minus candidate ids and source paths
- rendered vocal/minus preview paths
- duration, sample rate, channels
- `validation_status`
- `failure_reason`
- `skip_reason`
- `can_publish_as_result=false`
- `can_mark_release_ready=false`
- `release_ready=false`

`rendered_preview_audio.jsonl` includes one row per preview stem:

- `run_id`
- `track_id`
- `preview_audio_id`
- `stem_type` (`vocal_preview`, `minus_preview`, optional `pair_preview`)
- `source_candidate_id`
- `source_path`
- `rendered_path`
- duration, sample rate, channels
- `render_status` (`rendered`, `skipped`, `failed`)
- `validation_status`
- `failure_reason`
- `release_ready=false`

## Rendering Rules

- Rendered preview files are written only under the current run directory.
- Original selected stems are never overwritten.
- Default preview duration is 30 seconds.
- `bounded` mode copies a segment from the selected stems without enhancement,
  mastering, cleanup, denoise, debleed, EQ, threshold changes, model changes, or
  training.
- `full` mode copies the full selected stems and is still preview-only.
- Rendered previews are listening/evaluation artifacts only. They are never
  release-ready and cannot be published as final results.
- Melody transcription and recommendation indexing remain disabled.

## Validation

Use:

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_no_reference_preview_contract.py \
  --run-dir /app/lab_artifacts/autonomous_runs/example
```

The validator fails if:

- `audio_preview_ready=true` without rendered audio rows;
- rendered audio is missing, empty, outside the run directory, or has mismatched
  metadata;
- any preview artifact sets `release_ready=true`;
- preview artifacts allow publishing or release marking.
