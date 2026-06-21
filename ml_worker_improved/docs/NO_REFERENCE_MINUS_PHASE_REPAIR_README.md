# No-Reference Minus Phase Repair

`tools/run_no_reference_minus_phase_repair.py` builds bounded stereo phase
variants for an existing low-leakage minus source.

Contract:

- Docker-only CLI execution.
- CPU-DSP audio transform only; no neural inference.
- No references, stems, human CSV, training, fine-tune, model changes, cleanup
  chain, or threshold relaxation.
- Outputs are candidate variants plus `candidate_pair.jsonl`; they are not
  listen-pack or release artifacts.
- If the source mix and stems use different sample rates, the route resamples
  the mix/vocal context to the source minus sample rate and records
  `sample_rate_alignment` in the manifest. This is alignment provenance, not a
  quality acceptance gate.
- Downstream `run_no_reference_windowed_judge.py --score-mode minus` must accept
  a variant before any later preview/listen contract can consider it.
- `tools/build_short_minus_phase_repair_selection_pack.py` is fail-closed:
  variants with `pair_decision` other than `pair_compatible`, or any judge
  reason other than `no_reference_proxy_passed`, are excluded even if the
  top-level decision says `accept_candidate`.

Generated variants:

- `minus_mid_mono`
- `minus_side_attenuate_25`
- `minus_side_attenuate_50`

The manifest always keeps:

- `delivery_status=diagnostic_only`
- `delivery_allowed=false`
- `presentation_status=phase_repair_candidate_not_listen_pack`
- `release_status=not_release_ready`
- `release_ready=false`

Example:

```bash
docker compose run --rm --no-deps ml-worker python /app/tools/run_no_reference_minus_phase_repair.py \
  --mix-path /app/path/to/mix.mp3 \
  --vocal-path /app/path/to/context_vocal.flac \
  --minus-path /app/path/to/source_minus.flac \
  --output-dir /app/lab_artifacts/minus_phase_repair_example \
  --track-id track_id \
  --candidate-id source_candidate \
  --minus-route-id source_route
```

Validation:

```bash
docker compose run --rm --no-deps ml-worker python -m pytest \
  /app/tests/test_run_no_reference_minus_phase_repair.py \
  /app/tests/test_build_testing_stuff_music_corpus_job_matrix.py \
  /app/tests/test_run_testing_stuff_music_corpus_job_matrix.py \
  -q
```
