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

On the current Mac runtime, full-track MPS execution reaches the BS RoFormer model but is
blocked by PyTorch MPS complex tensor support:

```text
mps.scatter op operand #0 must be tensor of mps native type values, but got complex<f32>
```

Chunk and batch-size reduction avoids the earlier MPS buffer-size failure, but the complex
scatter failure remains a hard blocker for this checkpoint on MPS. Do not publish these
diagnostic runs as listen or accepted artifacts.
