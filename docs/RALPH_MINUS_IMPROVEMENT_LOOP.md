# RALPH Minus Improvement Loop

`run_ralph_minus_improvement_loop.py` is a Docker-only production gate for minus repair
candidates.

The loop exists to prevent a weak "accepted" minus from being delivered when it is
almost identical to the base minus, and to prevent aggressive vocal cleanup from being
delivered when it creates audible dips.

## Contract

- Generate multiple source-mix minus repair candidates.
- Compare each candidate against the base minus with windowed vocal-residue and dip
  metrics.
- Reject candidates that are too close to the base to be an audible fix.
- Reject candidates that reduce vocal residue by damaging RMS, presence, or air bands.
- Write `accepted_product_minus.flac` only when a candidate is accepted by the
  RALPH policy.
- Candidate directories use role-explicit names such as `raw_mix.flac`,
  `base_product_minus.flac`, `guide_product_minus.flac`, and
  `<recipe>_product_minus.flac`. Historical `raw_ref.flac`, `minus_base.flac`,
  `minus_guide.flac`, `minus_hybrid.flac`, `minus_windowed_hybrid.flac`,
  `minus_actual_masked_residue.flac`, and `02_adaptive_selected_minus.flac`
  artifacts are forensic evidence only and are not the current output contract.
- Candidate and selected manifests must expose `stem_type=minus` plus
  `artifact_role=candidate_product_minus` or `artifact_role=accepted_product_minus`
  so the audio role is machine-readable even when legacy compatibility keys are
  present.
- Mark blocked runs as `blocked_from_listen_pack`; do not present them as production
  listen results.

## Acceptance checks

The default policy requires:

- `rms_delta_db_vs_base >= -48.0`
- average vocal-projection improvement of at least `0.12 dB`, or peak vocal-correlation
  improvement of at least `0.0015`
- `worst_rms_dip_db >= -0.12`
- `worst_presence_dip_db >= -0.15`
- `worst_air_dip_db >= -0.12`
- `max_rms_boost_db <= 6.0`
- `max_presence_boost_db <= 8.0`
- `max_air_boost_db <= 8.0`

These values are not track timestamps and are not per-track hardcodes. They are
production safety thresholds for the minus route.

For iterative passes, pass `--reference-base-path` with the original minus baseline.
The loop then checks both:

- local damage versus the immediate input minus;
- cumulative damage versus the original baseline.

This prevents a sequence of individually small repairs from accumulating into an
audible dip.

For hardcase debugging, pass `--focus-windows-jsonl` with scored
`target_audio_error` windows and optional `--focus-track-id` /
`--focus-candidate-id` filters. Focus evaluation is written to the manifest only;
it does not relax the RALPH delivery gate and is not used to mark a candidate
accepted.

Eligible focus rows are ranked by hardcase risk before `--max-focus-windows` is
applied. This prevents early low-risk rows in the JSONL from hiding later
blocked or higher-risk low-confidence windows.

To continue from a previous RALPH result, use
`tools/build_ralph_residual_focus_windows.py` to convert the selected
candidate's `top_vocal_residue_windows` into a fresh `focus.jsonl`. That focus
view is generated from metric evidence in the manifest and remains
diagnostic-only (`delivery_allowed=false`).

When focus windows are present, the loop also adds a conservative
`focus_window_guarded` candidate. It uses the critic-selected blocked windows as a
generalized router input, not as per-track hardcoded timestamps. This candidate is
still subject to the same RALPH policy; if it reduces vocal residue by creating a
detail/RMS dip, it remains rejected and must not be promoted to a listen pack.

RALPH manifests can be fed back into `tools/build_audio_error_training_views.py`
with `--ralph-manifests`. This converts candidate focus-window evidence into
`minus_critic` score-only rows with
`source_judge_file=no_reference_minus_windowed_judge.jsonl`. The rows preserve
the RALPH window metrics and delivery status as provenance, but they are not
strict separator-training labels and rejected/blocked audio remains diagnostic.

## Runtime

Run only inside Docker:

```bash
python /app/tools/run_ralph_minus_improvement_loop.py \
  /app/path/source_mix.flac \
  /app/path/base_minus.flac \
  /app/path/guide_minus.flac \
  /app/path/primary_vocals.flac \
  /app/path/secondary_vocals.flac \
  /app/path/output_dir \
  --reference-base-path /app/path/original_base_minus.flac \
  --focus-windows-jsonl /app/path/scored_minus_critic_windows.jsonl \
  --listen-dir /app/path/listen_dir
```

The output manifest is `ralph_minus_improvement_manifest.json`.
