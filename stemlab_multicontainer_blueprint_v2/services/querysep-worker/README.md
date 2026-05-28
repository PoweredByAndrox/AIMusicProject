# querysep-worker

`querysep-worker` is the prompt/query-based long-tail extraction worker for StemLab blueprint v2.

It supports two execution modes:

- `clap_router`: CLAP audio/text routing over input-derived components and supplied buses. This is useful for routing and diagnostics, but it is not true query separation.
- `banquet_byoq`: real Banquet BYOQ decoding. CLAP selects an audio query exemplar from available buses/components; Banquet performs the target extraction.

Supported prompts are defined in `app/main.py`:

- `strings_orchestra`
- `synth_lead`
- `synth_pad`
- `arps_plucks`
- `brass_woodwinds`
- `organ_reeds`
- `fx_risers_impacts`
- `ambience_residual`

`ambience_residual` uses `clap_router`; the other prompts can use `banquet_byoq` when the checkpoint is present.

## Runtime contract

Real inference requires:

- `DRY_RUN=false`
- `QUERYSEP_BACKEND_DEFAULT=banquet_byoq` or a request step with `{"backend": "banquet_byoq"}`
- `QUERYSEP_BANQUET_CKPT_PATH` pointing to `models/query-bandit/ev-pre-aug.ckpt`
- `QUERYSEP_BANQUET_VENDOR_ROOT` pointing to the vendored `query-bandit` code

If `banquet_byoq` is explicitly requested and the checkpoint is missing, the worker fails fast. It must not silently return a CLAP-routed proxy as a successful Banquet stem.

## Banquet query modes

`banquet_byoq` normally uses prompt-specific CLAP ranking plus biases to build a small query ensemble.
For harmonic long-tail repair requests, callers can set:

```json
{
  "raw_step": {
    "candidate_type": "banquet_byoq_harmonic_residual_query",
    "query_mode": "harmonic_residual_single"
  },
  "build_from_paths": {
    "harmonic_residual": "/shared/path/harmonic_residual.flac"
  }
}
```

`harmonic_residual_single` is not a fallback route. It is still real Banquet BYOQ inference, but query
selection is constrained to `bus_harmonic_residual` only; broad proxy buses such as `other`, `guitar`,
`piano` and `melody` are blocked from the query ensemble. If the bus is missing or Banquet produces a
weak output, the request must fail or be rejected by the upstream acoustic gate.

## Docker smoke check

```bash
docker compose run --rm --no-deps -T -e DRY_RUN=false querysep-worker \
  python -c "import json; from app.banquet_runtime import banquet_ready_state; print(json.dumps(banquet_ready_state(), indent=2))"
```

Service-level query selection tests:

```bash
docker compose run --rm --no-deps -T querysep-worker \
  python -m unittest discover -s tests -p "test_*.py"
```
