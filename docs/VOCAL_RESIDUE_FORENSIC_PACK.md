# Vocal Residue Forensic Pack

`build_vocal_residue_forensic_pack.py` creates diagnostic-only listening evidence for
possible residual vocals inside a minus track.

The projection proxy is intentionally not treated as vocal evidence by itself.
Harmonic melody and other instrumental material can correlate with a vocal guide, so
projection-only windows are marked as `projection_only_melody_contamination`.

## Files

- `00_current_release_minus.flac`: the release minus being inspected.
- `01_actual_masked_residue_boosted.flac`: the first file to listen to; this is the
  actual masked residue extracted from the minus.
- `02_projection_proxy_reference_only_rejected_as_vocal_evidence.flac`: reference-only
  projection. It can contain melodic harmonics and must not be used as a vocal target.
- `clips/*_actual_masked_residue_boosted.flac`: short diagnostic clips for suspicious
  windows.
- `clips/*_projection_proxy_reference_only.flac`: reference-only companion clips.

## Contract

- `delivery_status=diagnostic_only`
- `delivery_allowed=false`
- `presentation_status=diagnostic_forensic_evidence_only`
- `projection_only_is_not_vocal_evidence=true`

Production repair routes must not optimize against projection-only evidence.

## Production cleanup usage

`apply_actual_masked_residue_minus_cleanup.py` uses the same evidence rule for cleanup:

- process only windows classified as `needs_human_review`;
- skip `projection_only_melody_contamination`;
- keep the result subject to the RALPH minus cumulative dip gate.

The route is registered in `run_ralph_minus_improvement_loop.py` as
`actual_masked_residue_guarded`.
