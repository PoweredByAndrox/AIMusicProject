# RALPH Minus Improvement Loop

`tools/run_ralph_minus_improvement_loop.py` is a release-capable minus repair attempt. It is not the no-reference listen-preview wrapper.

The route is Docker-only and external-only CPU/DSP scoring:

- `gpu_inference_skipped=true`
- `acceleration_mode=external_only_cpu_scoring`
- no model training
- no model registry edits
- no threshold relaxation

The runner writes `ralph_minus_improvement_progress.jsonl` with flushed phase records:

- `start`
- `input_normalization_start`
- `input_normalization_end`
- `candidate_start`
- `candidate_end`
- `end`

Before running candidate recipes, the runner normalizes raw audio, base minus, guide minus, and vocal guide inputs into `normalized_inputs/` at the base minus sample rate. Candidate recipes consume only those normalized files so child tools do not fail on sample-rate drift.

When `--focus-windows-jsonl` is provided, the manifest records local focus-window
metrics for blocked minus windows selected by the target audio-error critic. This
is diagnostic evidence only: `focus_evaluation_used_for_delivery=false`, and
blocked/rejected candidates remain blocked.

`focus_window_guarded` is enabled only when the focus JSONL resolves to at least
one blocked window. It is a guarded candidate, not a fallback: failed focus-window
attenuation remains diagnostic evidence and does not create a release artifact.

The route keeps `produce_only_release_ready=true`: if every candidate is rejected, it writes `delivery_status=blocked`, `delivery_allowed=false`, `presentation_status=blocked_from_listen_pack`, and `release_status=blocked_for_listening`. Rejected candidates must not be promoted to a listen pack or release.

For tracks without a verified reference stem bundle, this route can only provide internal proxy release-gate evidence. It does not replace an exact/reference waveform audit when the delivery contract requires one.
