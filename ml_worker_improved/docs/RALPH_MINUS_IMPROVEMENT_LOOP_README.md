# RALPH Minus Improvement Loop

`tools/run_ralph_minus_improvement_loop.py` is a release-capable minus repair attempt. It is not the no-reference listen-preview wrapper.

The route uses the project-approved container runtime and external-only CPU/DSP scoring:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_scoring`
- no model training
- no model registry edits
- no threshold relaxation
- no over-boost acceptance: excessive RMS/presence/air boosts are rejected like
  detail dips

The runner writes `ralph_minus_improvement_progress.jsonl` with flushed phase records:

- `start`
- `input_normalization_start`
- `input_normalization_end`
- `candidate_start`
- `candidate_end`
- `end`

Before running candidate recipes, the runner normalizes raw audio, base minus, guide minus, and vocal guide inputs into `normalized_inputs/` at the base minus sample rate. Candidate recipes consume only those normalized files so child tools do not fail on sample-rate drift.

Normalized input files use role-explicit names so product aliases are not confused with model-output names:

- `raw_mix.flac`
- `base_product_minus.flac`
- `guide_product_minus.flac`
- `primary_vocal_reference.flac` only when word-content audit detects words
- `secondary_vocal_reference.flac` only when word-content audit detects words
- `primary_residual_reference.flac` / `secondary_residual_reference.flac` when
  the input is a no-word residual rather than a usable vocal reference
- `primary_unverified_reference.flac` / `secondary_unverified_reference.flac`
  when no content audit proves the role

The same directory must include `inputs_manifest.json`, which records each normalized file's role, source path, source SHA-256, normalized filename, normalized SHA-256, and intended/readback audio fingerprints. If two distinct roles collapse into one normalized file hash while their intended audio fingerprints differ, the runner fails fast with `normalized_input_role_collision` instead of continuing with ambiguous artifacts.

Candidate artifact directories also use role-explicit product names. New RALPH
runs must not write active candidate files named `raw_ref.flac`,
`minus_base.flac`, `minus_guide.flac`, `minus_hybrid.flac`,
`minus_windowed_hybrid.flac`, `minus_actual_masked_residue.flac`, or
`minus_focus_window.flac`. Candidate tools write:

- `raw_mix.flac`
- `base_product_minus.flac`
- `guide_product_minus.flac` when a guide minus is used
- one candidate product, for example `hybrid_product_minus.flac`,
  `windowed_product_minus.flac`, `residue_product_minus.flac`, or
  `focus_product_minus.flac`

JSON manifests may keep legacy artifact keys as compatibility aliases, but
those keys must point at the role-explicit product filenames above. The selected
release copy is `accepted_product_minus.flac`; historical runs that still
contain `02_adaptive_selected_minus.flac` or `minus_*` candidate filenames are
forensic evidence only and are not renamed in place.

When `--focus-windows-jsonl` is provided, the runner first filters rows by
track/candidate and blocked-or-low-confidence criteria, then ranks eligible
focus windows by hardcase risk before applying `--max-focus-windows`. The
manifest records local focus-window metrics for the selected minus windows. This
is diagnostic evidence only: `focus_evaluation_used_for_delivery=false`, and
blocked/rejected candidates remain blocked.

`focus_window_guarded` is enabled only when the focus JSONL resolves to at least
one blocked window. It is a guarded candidate, not a fallback: failed focus-window
attenuation remains diagnostic evidence and does not create a release artifact.

`actual_masked_residue_guarded` may include low-priority residue windows only
through an explicit suspicion-score gate. The residue mask is additionally
gated by vocal/base overlap so quiet background vocal cleanup does not broadly
attenuate instrumental detail. This candidate is still release-gated: if RMS,
presence, air, or vocal-residue metrics fail, it remains diagnostic-only.

For iterative passes, `tools/build_ralph_residual_focus_windows.py` can convert
the selected candidate's `top_vocal_residue_windows` from a prior
`ralph_minus_improvement_manifest.json` into a ranked `focus.jsonl`. This is a
general metric-derived focus view, not per-track timestamp hardcoding, and it
keeps `delivery_allowed=false`.

The route keeps `produce_only_release_ready=true`: if every candidate is rejected, if the selected minus has detected word-like tokens, if word-content audit is missing/failed, or if the raw mix contains words while the primary vocal reference contains none, it writes `delivery_status=blocked`, `delivery_allowed=false`, `presentation_status=blocked_from_listen_pack`, and `release_status=blocked_for_listening`. Rejected or content-blocked candidates must not be promoted to a listen pack or release. Content-blocked selected candidates are copied only as `blocked_product_minus_candidate.flac`, never as `accepted_product_minus.flac`.

Use `--content-identity-audit-json` to pass precomputed ASR/word-audit evidence.
If an audit entry is missing and runtime ASR is not skipped, the runner tries a
CPU `faster-whisper` word-content audit. If that runtime audit is unavailable or
fails, delivery remains blocked instead of silently accepting a misnamed result.

Rejected RALPH candidates and bounded repair matrices should be converted into
`minus_repair_outcome_training_view.v1` metadata via
`tools/build_minus_repair_outcome_training_view.py`. That view preserves the
repair tradeoff evidence for future repair-action routing or model training
without changing delivery gates or publishing diagnostic audio.

For tracks without a verified reference stem bundle, this route can only provide internal proxy release-gate evidence. It does not replace an exact/reference waveform audit when the delivery contract requires one.
