# Minus Release Gate

`ml_worker_improved/tools/adaptive_minus_repair.py` writes
`adaptive_manifest.json.selected_minus_release_gate` after scoring adaptive minus candidates.

The gate protects the clean-minus route from two audible failures:

- residual vocal in the instrumental/minus output;
- instrumental detail dips caused by over-aggressive repair.

## Metrics

The gate consumes candidate manifest metrics:

- `vocal_residue_corr`;
- `vocal_to_instrumental_rms_ratio_db`;
- `presence_band_ratio_drift_db`;
- `air_band_ratio_drift_db`;
- focused-window `drift_vs_base.presence_band_ratio_drift_db`;
- focused-window `drift_vs_base.air_band_ratio_drift_db`.

`vocal_residue_corr`, `vocal_to_instrumental_rms_ratio_db`, `presence_band_ratio_drift_db`, and
`air_band_ratio_drift_db` are required for a release decision. If any of these metrics are missing,
the gate must return `blocked_for_listening` with `missing_minus_release_metrics` instead of treating
missing values as safe defaults.

`apply_flexible_minus_repair.py` writes the vocal-residue fields as source-minus projection proxy
metrics. They are not SDR/SI-SDR and do not require isolated vocal ground truth; they only estimate
whether material correlated with the source-minus residual is still present in the selected minus.

## Runtime Status

- `release_ready`: the selected minus has no release-blocking vocal residue or instrumental dip.
- `blocked_for_listening`: the selected minus is diagnostic-only because it still contains vocal
  residue or loses instrumental presence/air.

These statuses are internal proxy-gate statuses for adaptive profile selection only. They are not a
delivery contract and must not be exposed as a listen-pack or release-ready result by
`adaptive_minus_repair.py`.

## Manifest Fields

- `selected_minus_release_gate`: release status, thresholds, summary, and blockers for the selected
  candidate.
- `candidates[*].minus_release_gate`: the same release check for each ranked candidate.
- `selected_artifacts.release_status`: always `not_release_ready` for controller-selected
  candidates.
- `candidate_status`: `selected_candidate_only`.
- `delivery_status`: `candidate_only`.
- `delivery_allowed`: always `false` in `adaptive_manifest.json`.
- `presentation_status`: `blocked_until_downstream_delivery_gates`.
- `requires_post_judge`: always `true`.
- `internal_minus_gate_status` and `internal_minus_release_status`: copied from the internal
  `selected_minus_release_gate` so downstream gates can inspect the proxy result without treating it
  as a delivery decision.
- `selected_artifacts.delivery_status`, `selected_artifacts.delivery_allowed`,
  `selected_artifacts.presentation_status`, and `selected_artifacts.requires_post_judge`: the same
  candidate-only lock fields for the copied selected minus artifact.

When `--listen-dir` is used, `adaptive_minus_repair.py` must not copy the selected minus into the
listen directory as an accepted artifact. The selected FLAC remains in the run directory as a
candidate-only artifact until a downstream self-judge/result-judge/listen-preview pipeline accepts
it.

This gate does not claim SDR/SI-SDR without isolated ground truth stems. It is a proxy release
guard for the clean-minus listening workflow.

## Feedback Holdout Rule

Minus feedback records can use `feedback_split: "train"` or `feedback_split: "holdout"`.
Holdout records are reserved for validation and do not adjust adaptive minus routing priors.

## Focus-Window Rule

Focused windows can block and penalize a candidate through `focused_window_instrumental_dip` when
the local presence or air drift indicates that repair damaged instrumental detail in a critical
window.
