# No-Reference Full Stem Pack

`tools/build_no_reference_full_stem_pack_manifest.py` builds a contract-only
manifest for a full listen-preview stem pack:

- `vocal`
- `minus`
- `bass`
- `drums`
- `melody`
- `other`

It does not copy audio, export audio, run inference, train, fine-tune, edit model
weights, add cleanup routes, or mark anything `release_ready`.

The builder accepts only source-of-truth evidence:

1. an accepted `no_reference_vocal_minus_master_manifest.json` with
   `stem_type=both`;
2. a `no_reference_stem_candidate_score_matrix.json` with accepted and semantic
   verified non-vocal rows for `bass`, `drums`, `melody`, and `other`.

The output is accepted only when all required stems are present and fail-closed
otherwise:

- missing stem: `missing_required_stem:<stem>`;
- track identity mismatch: `track_id_mismatch:<stem>`;
- no accepted row: `no_accepted_required_stem:<stem>`;
- semantic proxy not verified: `stem_semantic_quality_not_verified:<stem>`;
- semantic blockers present: `stem_semantic_blocked:<stem>`;
- source safety violation: `source_stem_safety_violation:<stem>`.

Track identity is exact by default. If a non-vocal row was produced with a
different track-id slug, the builder may still accept it only when the row's
`source_listen_manifest` can be read and its `source_audio_path` exactly matches
the accepted master manifest `source_mix_path`. The accepted manifest records
`track_identity_match_method` and per-stem `track_identity_match_method` so this
reconciliation is auditable.

Accepted output remains listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=full_stem_pack`
- `accepted_stem_count=6`
- `release_ready=false`
- `can_publish_as_release=false`

Example:

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/build_no_reference_full_stem_pack_manifest.py \
  --master-manifest-json /app/lab_artifacts/testing_stuff_music_track2_full_pair_master_gate_20260612_v2/no_reference_vocal_minus_master_manifest.json \
  --stem-score-matrix-json /app/lab_artifacts/testing_stuff_music_stem_hard_case_repair_execute_v14_20260612/semantic_batch_after_repair/nine_thou_grant_mohrman_superstars_remix_320/stem_score_matrix/no_reference_stem_candidate_score_matrix.json \
  --output-manifest-json /app/lab_artifacts/testing_stuff_music_track2_full_stem_pack_20260612/no_reference_full_stem_pack_manifest.json \
  --summary-md /app/lab_artifacts/testing_stuff_music_track2_full_stem_pack_20260612/no_reference_full_stem_pack_summary.md
```

Validation:

```bash
docker compose run --rm --no-deps ml-worker python -m pytest \
  /app/tests/test_build_no_reference_full_stem_pack_manifest.py \
  -q
```
