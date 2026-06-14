# Autonomous Audio Judge

`tools/autonomous_audio_judge.py` and `tools/run_autonomous_decision_queue.py` provide a
control-plane-only autonomous decision layer for hard-case and candidate manifest evidence.

This layer does not add separation models, checkpoints, cleanup, debleed, denoise, threshold
changes, policy changes, audio exports, or training runs.

## Purpose

The judge replaces mandatory human review as a production requirement. Human review can still be
used as an optional spot-check, but every row receives a machine decision. If evidence is
insufficient, the judge blocks or marks a hard case instead of requesting human review.

## Inputs

- `hard_case_benchmark.jsonl`
- `hard_case_review_pack.jsonl`
- `golden_corpus.jsonl`
- `control_plane_dashboard.json`
- optional `--candidate-manifest` paths

The runner keeps benchmark fields as the primary source of truth because review rows do not carry
`source_manifest`, `split`, `failed_checks`, or window bounds.

## Outputs

- `autonomous_decision_queue.jsonl`
- `autonomous_decision_summary.md`
- `autonomous_judge_manifest.json`

Each decision row includes:

- `case_id`
- `source_manifest`
- `track_hash`
- `stem_type`
- `split`
- `dominant_failure_reason`
- `failure_labels`
- `failed_checks`
- `available_evidence`
- `judge_mode`
- `decision`
- `confidence`
- `confidence_band`
- `reasons`
- `forbidden_actions`
- `allowed_next_step`
- `human_review_required`
- `optional_human_spotcheck`

## Judge Modes

- `reference_aware`: source manifest contains objective/reference metrics such as SDR, SI-SDR,
  SIR, SAR, reference likeness, residual RMS/peak, leakage, alignment, or reconstruction evidence.
- `no_reference_proxy`: source manifest or control-plane rows provide status/proxy/issue evidence,
  but no reference metric stack is available.
- `manifest_only`: evidence is too thin for acceptance. This mode never returns
  `accepted_for_listen_pack` or `release_ready`.

## Decision Safety

- Low confidence cannot be accepted or release-ready.
- `delivery_allowed=false` blocks promotion.
- `diagnostic_only` cannot become a listen result.
- `accepted_for_listen_pack` requires the artifact status contract to be listenable.
- `release_ready` requires the artifact status contract to be deliverable.
- `candidate_space_exhausted` becomes `hard_case`.
- `no_accepted_candidate` never exports audio.
- Holdout rows cannot become `finetune_candidate`.
- Golden track hashes cannot become training or tuning candidates.
- Human review is never required by the autonomous path.

## Forbidden Actions

Every row records forbidden actions:

- `no_audio_export`
- `no_diagnostic_audio_as_result`
- `no_cleanup_debleed_denoise_escalation`
- `no_gate_relaxation`
- `no_new_model_or_weight`
- `no_training_start`
- `no_policy_auto_change`

## Docker Command

```bash
docker compose run --rm --no-deps ml-worker python tools/run_autonomous_decision_queue.py \
  --hard-case-benchmark-jsonl /app/lab_artifacts/control_plane_phase3_20260528/hard_case_benchmark.jsonl \
  --review-pack-jsonl /app/lab_artifacts/control_plane_phase4_20260528/hard_case_review_pack.jsonl \
  --golden-jsonl /app/lab_artifacts/control_plane_phase2_20260528/golden_corpus.jsonl \
  --dashboard-json /app/lab_artifacts/control_plane_phase2_20260528/control_plane_dashboard.json \
  --output-dir /app/lab_artifacts/control_plane_phase7_20260528
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_autonomous_audio_judge.py \
  tests/test_run_autonomous_decision_queue.py \
  tests/test_artifact_status_contract.py \
  tests/test_run_acceptance_first_vocal_ladder.py \
  tests/test_build_hard_case_benchmark.py \
  tests/test_build_hard_case_review_pack.py \
  -q
```
