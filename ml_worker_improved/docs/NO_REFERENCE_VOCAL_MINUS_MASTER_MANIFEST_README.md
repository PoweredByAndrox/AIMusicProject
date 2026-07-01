# No-Reference Vocal Minus Master Manifest

`tools/build_no_reference_vocal_minus_master_manifest.py` combines one accepted vocal listen-preview manifest and one accepted minus listen-preview manifest into a master listen-preview manifest.

It is a contract wrapper only. It does not copy audio, export audio, run models, train, add cleanup/debleed/denoise, relax gates, or mark anything `release_ready`.

The master manifest can be accepted only when:

1. the vocal source manifest is promotable by `artifact_status_contract.can_promote_to_listen_result()`;
2. the minus source manifest is promotable by `artifact_status_contract.can_promote_to_listen_result()`;
3. source stem types are exactly `vocal` and `minus`;
4. both source manifests use the same `track_id`, or they have different legacy `track_id` conventions but the same non-empty source `mix_path`;
5. selected vocal and minus paths are present;
6. neither source manifest reports a safety violation such as training, model changes, cleanup route additions, threshold relaxation, or audio export/copy;
7. `--pair-compatibility-jsonl` contains an exact matching `score_mode=both` row for the selected vocal/minus paths;
8. that pair row has `pair_decision=pair_compatible`, empty `pair_blockers`, no blocking semantic labels in generic `decision/reasons`, and at least 5 judged windows.

Missing pair compatibility evidence is fail-closed with `dominant_failure_reason=pair_compatibility_evidence_missing`. A row for a different selected stem path is blocked with `pair_compatibility_no_exact_match`.

The `both` row is compatibility evidence only. Its generic `decision` field is not used as a vocal/minus selector, because vocal and minus acceptance must already come from their separate mode-specific listen-preview manifests. However, generic `decision/reasons` are still semantic-clearance evidence: labels such as `vocal_leakage_in_minus`, `foreign_source_recovered:*`, residual, leak/bleed, or mix-conservation blockers fail closed before the master can become listenable.

When the accepted vocal and accepted minus manifests exist but no exact pair row exists yet, use `tools/build_no_reference_pair_candidate_from_listen_manifests.py` to create a provenance-safe one-row candidate JSONL, then run `tools/run_no_reference_windowed_judge.py --score-mode both` on that row. Do not hand-write pair rows or reuse pair rows for different selected stem paths.

Source stem status is strict: both source manifests must remain `release_status=listen_preview_only`. A `release_ready` vocal or minus source is blocked from this no-reference master wrapper even when the generic artifact contract would consider it promotable.

When the raw `track_id` fields differ but the source mix path matches, the master manifest records `source_track_ids` and `track_identity_match_method=shared_mix_path` instead of silently rewriting either source manifest.

Accepted output remains listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=both`
- `accepted_stem_count=2`
- `pair_decision=pair_compatible`
- `pair_score_mode_decision=<raw both-mode generic decision, after semantic-clearance check>`
- `pair_score_mode_reasons=<raw both-mode generic reasons, after semantic-clearance check>`
- `pair_compatibility_score=<both-mode score>`
- `release_ready=false`
- `can_mark_release_ready=false`

Example:

```bash
python tools/build_no_reference_vocal_minus_master_manifest.py \
  --vocal-manifest-json /app/lab_artifacts/example/vocal_delivery_pipeline/no_reference_vocal_listen_preview_manifest.json \
  --minus-manifest-json /app/lab_artifacts/example/minus_delivery_pipeline/no_reference_minus_listen_preview_manifest.json \
  --pair-compatibility-jsonl /app/lab_artifacts/example/no_reference_master_pair_compatibility.jsonl \
  --output-manifest-json /app/lab_artifacts/example/no_reference_vocal_minus_master_manifest.json \
  --summary-md /app/lab_artifacts/example/no_reference_vocal_minus_master_summary.md
```
