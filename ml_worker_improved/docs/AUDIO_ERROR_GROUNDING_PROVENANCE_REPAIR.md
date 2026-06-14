# Audio Error Grounding Provenance Repair

MODEL-1B audits and repairs provenance for the MODEL-1 audio error grounding
dataset. It does not train a separator, train a critic, download models, relax
thresholds, run melody transcription, update recommendation indexes, publish
results, or mark `release_ready`.

## Purpose

MODEL-1 intentionally keeps rows unsafe when required provenance is missing.
Older runtime artifacts often lack `route_id` or
`pair_compatibility_status`, even when exact source artifacts still prove the
candidate route and pair decision. MODEL-1B adds a strict audit/repair layer for
those recoverable cases.

## Audit

```bash
docker compose run --rm ml-worker python tools/audit_audio_error_dataset_provenance.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001 \
  --run-roots \
    /app/lab_artifacts/autonomous_runs/stem_3_track002_conditional_route_001 \
    /app/lab_artifacts/autonomous_runs/stem_3_household_goods_conditional_route_001 \
    /app/lab_artifacts/autonomous_runs/preview_1_audio_render_contract_001 \
    /app/lab_artifacts/no_reference_meta_harness/runtime_validation_audio_preview_001 \
  --output-dir /app/lab_artifacts/audio_error_grounding/dataset_001_provenance_audit
```

Outputs:

- `audio_error_provenance_audit.json`
- `audio_error_provenance_audit.jsonl`
- `audio_error_provenance_audit_summary.md`

The audit reads only source-of-truth runtime artifacts:

- `candidate_pool.jsonl`
- `selected_candidates.jsonl`
- `pair_compatibility.jsonl`
- `minus_windowed_judge.jsonl`
- `vocal_windowed_judge.jsonl`
- `candidate_wave_jobs.jsonl`
- `candidate_job_plan.jsonl`
- recursive `job_manifest.json`
- `run_manifest.json`
- `preview_manifest.json`
- `rendered_preview_audio.jsonl`

Allowed repair evidence:

- exact candidate id match
- exact vocal/minus stem path match
- exact normalized stem path match
- exact selected candidate id match
- exact pair compatibility artifact row match
- exact route/config id from a matched candidate/job/plan/manifest row

Forbidden repair evidence:

- filename-only matching
- route family guesses
- track-level default routes
- synthetic ids
- `unknown` placeholders
- row-level blockers copied as window-level blockers
- pair compatibility guessed from final runtime status

## Repair

```bash
docker compose run --rm ml-worker python tools/repair_audio_error_dataset_provenance.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001 \
  --audit-dir /app/lab_artifacts/audio_error_grounding/dataset_001_provenance_audit \
  --output-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired
```

Outputs:

- `repaired_audio_error_grounding_manifest.json`
- `repaired_audio_error_window_dataset.jsonl`
- `repaired_audio_error_maps_index.jsonl`
- `provenance_repair_report.json`
- `provenance_repair_summary.md`

The original dataset directory is never overwritten. Repaired rows include:

- `provenance_repaired`
- `repaired_fields`
- `repair_source_files`
- `repair_confidence`

`safe_for_training=true` is set only when candidate id, route id, pair
compatibility status, window-level judge evidence, audio paths, window
boundaries, and error maps are present. If a previously unsafe row has proven
provenance but no error map, the repair command computes maps into the repaired
output directory.

## Validate

```bash
docker compose run --rm --no-deps ml-worker python tools/validate_audio_error_dataset_provenance_repair.py \
  --dataset-dir /app/lab_artifacts/audio_error_grounding/dataset_001_repaired
```

The validator fails if a safe row has missing or `unknown` route id, a missing
pair compatibility status, a synthetic candidate id, missing source evidence for
a repaired row, missing audio/map paths, `release_ready=true`, or any training
flag set to true.

## Safety

- `reference_required=false`
- `human_csv_required=false`
- `training_started=false`
- `separator_training_started=false`
- `model_downloaded=false`
- `model_registry_edited=false`
- `thresholds_relaxed=false`
- `cleanup_routes_added=false`
- `release_ready=false`
- `melody_transcription_ran=false`
- `recommendation_index_updated=false`
