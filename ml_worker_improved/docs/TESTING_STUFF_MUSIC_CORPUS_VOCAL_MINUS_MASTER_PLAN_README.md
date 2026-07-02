# Testing Stuff Music Corpus Vocal/Minus/Master Plan

`tools/build_testing_stuff_music_corpus_vocal_minus_master_plan.py` turns a
fail-closed corpus gate report into per-track next gates for the
vocal/minus/master/full-stem progression.

The planner is diagnostic-only. It does not render audio, does not run model
inference, does not change thresholds, and never marks a corpus result
`release_ready`.

## Inputs

- `--corpus-gate-report-json`: output from
  `build_testing_stuff_music_corpus_gate_report.py`.
- `--vocal-manifest-root`: optional roots containing accepted
  `no_reference_vocal_listen_preview_manifest.json` files.
- `--minus-manifest-root`: optional roots containing accepted
  `no_reference_minus_listen_preview_manifest.json` files.
- `--master-manifest-root`: optional roots containing accepted
  `no_reference_vocal_minus_master_manifest.json` files.
- `--full-stem-pack-root`: optional roots containing source-ready,
  fail-closed `no_reference_full_stem_pack_manifest.json` files.

## Next Gates

- `run_corpus_job_matrix_for_track`: no candidate evidence exists for the
  track in the current corpus gate report.
- `run_vocal_delivery_pipeline`: vocal candidate generation exists or is
  needed, but no accepted vocal listen-preview manifest is present.
- `run_minus_delivery_pipeline`: accepted vocal exists, but no accepted minus
  listen-preview manifest is present.
- `run_pair_compatibility_and_master_manifest`: accepted vocal and minus exist,
  but no accepted vocal/minus master manifest is present.
- `complete_nonvocal_stem_score_matrices`: master exists, but one or more
  required non-vocal stems are not accepted through a stem score matrix.
- `build_full_stem_pack_manifest`: master and required non-vocal stems are
  ready for the per-track full-stem pack builder.
- `full_stem_pack_source_ready_requires_aggregate_machine_gate`: a full-stem
  source pack exists, but final listen delivery is still blocked until staged
  aggregate validation plus machine stem gate acceptance.

## Example

```bash
python /app/tools/build_testing_stuff_music_corpus_vocal_minus_master_plan.py \
  --corpus-gate-report-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/corpus_gate_report_v2/testing_stuff_music_corpus_gate_report_manifest.json \
  --master-manifest-root /app/lab_artifacts/testing_stuff_music_hotstepper_vocal_minus_master_20260612_v2 \
  --full-stem-pack-root /app/lab_artifacts/testing_stuff_music_hotstepper_full_stem_pack_20260613_v2 \
  --output-manifest-json /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_plan_v1/testing_stuff_music_corpus_vocal_minus_master_plan.json \
  --summary-md /app/lab_artifacts/testing_stuff_music_corpus_wave_20260613_v1/vocal_minus_master_plan_v1/testing_stuff_music_corpus_vocal_minus_master_plan_summary.md
```

Use this plan to drive the next bounded Docker wave. Do not treat diagnostic
candidate generation as a listen result unless the referenced mode-specific
delivery gates produce accepted listen-preview manifests.
