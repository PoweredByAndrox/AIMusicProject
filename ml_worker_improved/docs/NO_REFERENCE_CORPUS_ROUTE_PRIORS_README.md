# No-Reference Corpus Route Priors

`tools/build_no_reference_corpus_route_priors.py` aggregates existing
mode-specific no-reference judge JSONL rows across tracks and turns them into
deterministic route priors.

It is a metadata-only feedback layer:

- no audio is rendered or copied;
- no reference stems are required;
- no human CSV is required;
- no training or fine-tune is started;
- no thresholds are relaxed;
- no result can be promoted to listen pack or release from this artifact.

## Why

The mode-specific score matrix can show the same pattern on many tracks:

- vocal route passes;
- pair compatibility may pass;
- minus route still fails with `severe_vocal_bleed_in_minus`.

That evidence must not be hidden by `both` mode. The route-prior artifact makes
the failure reusable for future routing: repeated failed minus baselines are
deprioritized, while accepted vocal routes remain usable as vocal candidates.

## Inputs

One or more JSONL files from `run_no_reference_windowed_judge.py`, usually from
`--score-mode vocal`, `--score-mode minus`, or `--score-mode both`.
The builder also accepts matrix/result rows that embed `result_judge` and
`self_judge` objects. This lets a no-reference minus candidate score matrix feed
route priors without flattening the self-judge evidence first.

Relevant row fields:

- `track_id`
- `candidate_id`
- `score_mode`
- `vocal_decision`, `vocal_score`, `vocal_blockers`
- `minus_decision`, `minus_score`, `minus_blockers`
- `pair_decision`, `pair_compatibility_score`, `pair_blockers`
- `vocal_leakage_in_minus_score`
- `reasons`
- optional `vocal_route_id`, `minus_route_id`, `pair_route_id`
- optional nested `self_judge.status`, `self_judge.passed`,
  `self_judge.blocker_count`, and `self_judge.top_blockers`

If explicit route ids are missing, the tool infers a stable route id from the
candidate audio path parent directory.

For minus priors, a row with `self_judge.passed=false`, a positive
`self_judge.blocker_count`, or a blocked self-judge status is treated as a
blocked route even when the no-reference result judge says `accept_candidate`.
`vocal_projection_proxy` and `vocal_stem_corr_proxy` blockers are normalized to
`vocal_leakage_in_minus` so they affect the same route-prior failure bucket.

Rows marked `holdout=true`, `evaluation_split=holdout`, `feedback_split=holdout`,
or `split=holdout` are reported but do not drive train priors.

## Outputs

- route-prior JSONL;
- manifest JSON;
- Markdown summary.

Each route-prior row includes:

- `stem_type`: `vocal`, `minus`, or `pair`;
- `route_id`;
- acceptance/block counts;
- observed track count;
- score and leakage aggregates;
- top blockers and failure reasons;
- route recommendation;
- `source_pool_eligible` evidence for minus routes that are blocked as a
  baseline but still contain a low-leakage, high-score minus source suitable for
  a later repair/source-pool selector;
- `best_source_track_id`, `best_source_candidate_id`, `best_source_audio_path`
  and source metrics when such repair-source evidence exists;
- hard delivery lock fields.

`source_pool_eligible=true` is not an acceptance signal. It only preserves
component-level evidence so downstream matrix/repair stages can reuse an
existing minus source as a donor without re-running a known-bad baseline and
without presenting that source as a listen or release artifact.

The manifest always has:

- `delivery_status=diagnostic_only`
- `delivery_allowed=false`
- `presentation_status=route_prior_not_listen_pack`
- `release_status=not_release_ready`
- `training_allowed=false`
- `fine_tune_allowed=false`

## Docker Run

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/build_no_reference_corpus_route_priors.py \
  --judge-jsonl /app/lab_artifacts/testing_stuff_music_vocal_minus_pair_matrix_judge_20260611/no_reference_windowed_judge_both.jsonl \
  --judge-jsonl /app/lab_artifacts/testing_stuff_music_track2_vocal_minus_judge_20260611/no_reference_windowed_judge_both.jsonl \
  --output-jsonl /app/lab_artifacts/testing_stuff_music_corpus_route_priors_20260611/no_reference_corpus_route_priors.jsonl \
  --manifest-json /app/lab_artifacts/testing_stuff_music_corpus_route_priors_20260611/no_reference_corpus_route_priors_manifest.json \
  --summary-md /app/lab_artifacts/testing_stuff_music_corpus_route_priors_20260611/no_reference_corpus_route_priors_summary.md
```

## Validation

```bash
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_build_no_reference_corpus_route_priors.py \
  tests/test_docker_runtime_policy.py \
  -q
```
