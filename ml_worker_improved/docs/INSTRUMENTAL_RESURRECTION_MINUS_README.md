# Instrumental Resurrection Minus

`tools/export_instrumental_resurrection_minus.py` exports a clean-minus candidate from the
manual `bs_roformer_instrumental_resurrection_unwa` checkpoint.

## Runtime Contract

- `--device cuda[:id]` remains the production GPU route inside the approved project runtime.
- `--device cpu` is diagnostic only and records `expected_acceleration_mode=cpu_diagnostic_inference`.
- `--device mps` is an explicit macOS accelerator attempt. The exporter does not pass
  `--device_ids`; the vendor inference code selects MPS through `torch.backends.mps`.
- Failed model inference writes `stem_manifest.json` with
  `delivery_status=blocked_model_inference_failed`,
  `delivery_allowed=false`, `presentation_status=diagnostic_blocked_not_for_listen_pack`,
  and a failed `model_inference` progress event.

## Current Mac Finding

The exact-architecture adapter now completes both short and full-track MPS inference with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Historical shared-class runs failed on incompatible complex
tensor/state-dict paths; they are not evidence against the isolated adapter.

The vendored demixer enables AMP through the legacy `torch.cuda.amp.autocast` API. CUDA is
absent on Apple Silicon, so that context disabled AMP and left transformer matmuls in
float32. The isolated MPS adapter redirects that one-job exporter-local call to
`torch.autocast(device_type="mps", dtype=torch.float16)`. The manifest records
`precision_contract.mode=mps_autocast_float16` only after adapter initialization succeeds;
failed initialization leaves `mps_cuda_autocast_compatibility_installed=false`. The route
remains fail-closed with CPU fallback disabled. The adapter also guards `torch.stft` and
`torch.istft`: a CPU retry after a failed MPS operation raises an error instead of executing
the upstream exception-driven CPU branch, while an independent CPU FFT after a successful
MPS operation is not mislabeled as a retry.

The current three-track performance validation records `20.3-22.2%` lower
vendor inference time and `20.0-21.9%` lower exporter time. The float16 output
is not byte-equivalent to the previous float32 candidate, so the manifest sets
`requires_quality_equivalence_gate=true`. These outputs remain candidate-only;
the speed validation does not bypass stem quality or delivery gates.

Measured alternatives that were rejected for this profile:

- `PYTORCH_MPS_PREFER_METAL=1`: no measurable speed difference;
- `PYTORCH_MPS_FAST_MATH=1`: no measurable speed difference;
- chunks below 12 seconds: either slower or materially changed leakage/detail
  metrics;
- `batch_size=2`: completed under AMP but was slower than batch size 1.

## Exact-Architecture Mac Adapter

Checkpoints whose upstream `bs_roformer.py` is not state-dict compatible with
the shared runtime class must use `--model-implementation-path`. The exporter
runs `tools/run_bs_roformer_custom_impl.py` inside its own already-isolated,
one-job exporter process, replaces only that process's
`models.bs_roformer.BSRoformer`, and retains the vendor inference loader's
strict `load_state_dict` behavior. A redundant nested Python subprocess is not
used. Partial checkpoint loading is not allowed.

The `bs_roformer_large_inst` Mac API route uses this adapter with MPS fallback
disabled, an MPS memory profile (`chunk_size=529200`, `batch_size=1`,
`num_overlap=1`), and SHA verification against `source.json`. The source record
currently declares `production_allowed=false` because upstream license metadata
is missing, so its output is diagnostic `candidate_only` evidence even after
inference succeeds. Audio quality gates and license clearance are separate
requirements for a production/listen result.

The manifest records the observed runtime separately from policy. Native Mac API jobs use
`observed_runtime.runtime_environment=native_mac_mps_diagnostic`; Apple-container and Docker
runs retain their own runtime markers. A `candidate_only` artifact is never copied into
`--listen-dir`; the request is recorded as `listen_pack_status=skipped_candidate_not_accepted`.

## Persistent Mac Worker

The exact Large-Inst architecture now has a single-model Unix-socket worker.
It loads the strict checkpoint once and keeps all MPS autocast/STFT patches
inside that dedicated process. The exporter enables it only through the
explicit `--resident-worker-socket` option; worker failure blocks inference and
does not trigger a cold fallback.

On the current hardened 12-second benchmark, the resident exporter completed in
`9.040 s` versus `10.846 s` for the prior cold exporter, with `130.43 dB` SDR
and `99.99999999%` waveform cosine against that cold output. This proves
execution parity only. The multi-proxy quality gate still blocks the candidate
for vocal/full-mix residue and detail loss. Operational details are in
`docs/MAC_MPS_LARGE_INST_WORKER_README.md`.
