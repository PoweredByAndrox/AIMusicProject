# No-Reference Stem Listen Preview

`tools/build_no_reference_stem_listen_preview_manifest.py` promotes a generated
non-vocal stem candidate into a listen-preview-only manifest.

Supported target stems:

- `melody`
- `drums`
- `bass`
- `other`

This route deliberately blocks `vocal`, `minus`, and `both` so it cannot bypass
their dedicated mode-specific delivery contracts.

## Contract

- input is an existing `stem_manifest.json` from the Docker job-matrix
  candidate-generation route;
- for multistem model exports, `--target-stem` may select any supported
  non-vocal entry already present in `stems` even when the source manifest's
  own `target_stem` was produced for another stem;
- the selected target audio path must exist and be readable inside Docker;
- audio integrity gate recomputes duration, RMS, and peak from the file;
- GPU provenance is required by default:
  `gpu_inference_required=true`, `gpu_inference_skipped=false`, and
  `cuda_preflight.cuda_available=true`;
- source manifests that already claim delivery/release readiness are blocked;
- source manifests with fallback, hidden inference, per-track hardcode, safety
  violations, training, fine-tune, model registry edits, cleanup routes, copied
  audio, exported audio, threshold relaxation, or release readiness are blocked;
- accepted output is only listen-preview:
  `delivery_status=accepted`, `delivery_allowed=true`,
  `presentation_status=accepted_for_listen_pack`,
  `release_status=listen_preview_only`;
- accepted output still sets `semantic_quality_verified=false` and
  `release_ready=false`.

This contract is not a semantic stem-quality proof and must not be presented as
`release_ready`.

## Run

```bash
docker compose run --rm --no-deps ml-worker python tools/build_no_reference_stem_listen_preview_manifest.py \
  --stem-manifest-json /app/lab_artifacts/testing_stuff_music_track2_nonvocal_job_matrix_20260612/nine_thou_grant_mohrman_superstars_remix_320/melody/htdemucs_6s_aggressive_melody_first/stem_manifest.json \
  --target-stem melody \
  --track-id nine_thou_grant_mohrman_superstars_remix_320 \
  --output-manifest-json /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612/melody/no_reference_stem_listen_preview_manifest.json \
  --summary-md /app/lab_artifacts/testing_stuff_music_track2_nonvocal_listen_preview_20260612/melody/no_reference_stem_listen_preview_summary.md
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker python -m pytest \
  /app/tests/test_build_no_reference_stem_listen_preview_manifest.py \
  /app/tests/test_artifact_status_contract.py \
  -q
```
