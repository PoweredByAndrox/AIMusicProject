# Stabilization Control-Plane Implementation

This layer is control-plane only. It does not add model inference, checkpoints,
cleanup chains, debleed stages, denoise stages, or relaxed gates.

## Hard Prohibitions

- Do not add model registry entries.
- Do not add checkpoints, weights, `.pt`, `.ckpt`, `.npz`, `.pth`, or vendor model assets.
- Do not add cleanup/debleed/denoise stages.
- Do not relax gates or thresholds.
- Do not export diagnostic or blocked audio as a user result.
- Do not run project validation on the host; use Docker `ml-worker`.

## New Tools

- `tools/artifact_status_contract.py`: shared status/provenance/promotion contract.
- `tools/build_golden_corpus.py`: protected holdout corpus from accepted manifests only.
- `tools/build_hard_case_corpus.py`: blocked vocal/minus hard-case corpus from manifests.
- `tools/run_acceptance_first_vocal_ladder.py`: acceptance-first selection over registered candidate manifests.

## Required Semantics

- `diagnostic != result`
- `blocked != result`
- `accepted_for_listen_pack = listenable` only when `delivery_allowed=true`
- `release_ready = deliverable` only when `delivery_allowed=true`
- `candidate_space_exhausted = stop`
- golden corpus is holdout-only and cannot be used for training or threshold tuning
- train/dev/holdout split is by `track_hash`, never by window

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_artifact_status_contract.py \
  tests/test_build_golden_corpus.py \
  tests/test_build_hard_case_corpus.py \
  tests/test_run_acceptance_first_vocal_ladder.py \
  tests/test_adaptive_policy.py \
  tests/test_run_model_zoo_benchmark.py \
  tests/test_adaptive_minus_repair.py \
  -q
```
