# macOS 8GB Audio Runtime

This project can use a Mac as a constrained audio experiment host, but the
runtime contract is different from the current CUDA Docker path.

## Source-of-truth position

- Existing production separator routes are audio routes: Demucs, MDX/MDX23C,
  BS-RoFormer, MelBand/Mel-RoFormer, SCNet, and registered project adapters.
- Text vLLM/Qwen is not an audio separator and must not replace those routes.
- Docker Desktop, Docker VMM, Apple container tooling, and Docker Model Runner
  may improve container orchestration or local LLM/controller workflows, but
  they do not by themselves prove that Demucs/MDX/MelBand/SCNet can use the
  Apple GPU inside a Linux container.
- The macOS audio GPU candidate path is native PyTorch MPS. It is production
  blocked unless `torch.backends.mps.is_available()` is true on the Mac and the
  actual separator route is explicitly adapted to `device=mps`.

Official references checked for this contract:

- Docker Desktop VMM: https://docs.docker.com/desktop/features/vmm/
- Docker Model Runner: https://docs.docker.com/model-runner/
- Apple container project: https://github.com/apple/container
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html

## Supported modes

### `native_mps_audio`

Use this only after the Mac preflight reports:

- macOS host;
- Apple Silicon `arm64`;
- at least 8 GiB unified memory;
- importable PyTorch;
- `torch.backends.mps.is_built() == true`;
- `torch.backends.mps.is_available() == true`.

This mode is for a single-model, single-route audio probe first. It is not a
full `model_split_v2` route zoo on an 8GB machine.

### `docker_orchestration_only`

Docker on macOS can run project orchestration, manifests, CPU diagnostics, and
non-audio helpers. It must not be treated as GPU audio separation until a
specific container proves PyTorch MPS or another real audio accelerator path.

### `docker_model_runner_controller_only`

Docker Model Runner can be useful for a text LLM/controller experiment. It is
not a Demucs/MDX/MelBand/SCNet runtime.

## 8GB constraints

The Mac profile must start stricter than the Windows 12GB CUDA profile:

- one audio model resident at a time;
- no text vLLM in the audio path;
- no QuerySep/long-tail route until memory is measured;
- no parallel separator workers;
- no hidden CPU fallback for production audio;
- `route_budget_profile=fast`;
- suggested `max_candidates=3` for the first probe;
- every run must write `progress.jsonl`, `manifest.json`, and `summary.md`.

## Preflight command

Run this on the Mac from the mounted project folder:

```bash
python3 ml_worker_improved/tools/build_macos_8gb_runtime_plan.py \
  --out artifacts/stem_eval/macos8_preflight \
  --echo-progress
```

The output is diagnostic only:

- `delivery_status=blocked`;
- `delivery_allowed=false`;
- `presentation_status=diagnostic_macos_runtime_plan_only`.

If the manifest says `ready_for_native_mps_audio_probe`, the next step is a
small native-MPS separator smoke test. If it says `blocked_macos_preflight`,
do not run the full audio search on that Mac.

## SSH / mounted disk shape

Expected layout when the Mac is connected over SSH and the project disk is
mounted locally on the Mac:

```text
<mounted-project>/AIMusicProject/
  ml_worker_improved/
  docs/
  artifacts/stem_eval/macos8_preflight/
```

Keep generated artifact path segments short. Store host-specific details in
`manifest.json`, not in directory names.

## Not allowed

- Do not install or run Qwen/vLLM as if it were a music separator.
- Do not mark Docker Model Runner as audio GPU support.
- Do not use CPU fallback as a production result when MPS is unavailable.
- Do not publish listen packs from Mac experiments until the normal delivery
  contract accepts them.
