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

## Manifest Fields

- `selected_minus_release_gate`: release status, thresholds, summary, and blockers for the selected
  candidate.
- `candidates[*].minus_release_gate`: the same release check for each ranked candidate.
- `selected_artifacts.release_status`: `release_ready` or `diagnostic_only`.
- `delivery_status`: `accepted` only when the selected gate is `release_ready`; otherwise `blocked`.
- `delivery_allowed`: `true` only when the selected gate is `release_ready`.
- `presentation_status`: `accepted_for_listen_pack` only when `delivery_allowed=true`; otherwise
  `blocked_from_listen_pack`.
- `selected_artifacts.delivery_status`, `selected_artifacts.delivery_allowed`, and
  `selected_artifacts.presentation_status`: the same delivery lock fields for the copied selected
  minus artifact.

When `--listen-dir` is used, `adaptive_minus_repair.py` may copy the selected minus into the listen
directory only if `delivery_allowed=true`. Blocked candidates can remain in the run directory as
diagnostic artifacts, but they are not listen-pack deliverables.

This gate does not claim SDR/SI-SDR without isolated ground truth stems. It is a proxy release
guard for the clean-minus listening workflow.

## Feedback Holdout Rule

Minus feedback records can use `feedback_split: "train"` or `feedback_split: "holdout"`.
Holdout records are reserved for validation and do not adjust adaptive minus routing priors.

## Focus-Window Rule

Focused windows can block and penalize a candidate through `focused_window_instrumental_dip` when
the local presence or air drift indicates that repair damaged instrumental detail in a critical
window.
