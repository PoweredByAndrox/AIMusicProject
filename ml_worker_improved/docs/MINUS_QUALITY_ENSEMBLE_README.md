# Minus Quality Ensemble Audit

`tools/audit_minus_quality_ensemble.py` is the fail-closed diagnostic gate for
dedicated minus candidates. It does not alter audio or model weights. It turns
multiple independent vocal-model outputs into segment-level evidence for:

- direct vocal residue;
- full-mix-like passthrough;
- pitch-shifted vocal residue;
- RMS muffling;
- presence loss;
- air loss.

## Evidence Contract

The audit requires vocal run manifests, not bare audio filenames. Every proxy
must:

- declare a vocal target role;
- identify its model;
- expose a vocal output whose bytes match manifest provenance;
- expose the exact source mix used for inference;
- match the audited source mix by SHA-256.

When manifests were produced through the native Mac mount but the audit runs
inside Apple `container`, pass the producer-side project root with
`--producer-project-root`. The resolver maps the path relative to that declared
root into `--runtime-project-root` before considering sibling fallback files.
This preserves exact source-byte provenance even when a route also emitted a
re-encoded `raw_ref.flac`.

At least two different model identities and two different proxy-output hashes
are required for consensus. Duplicate models, duplicate bytes, source mismatch,
missing provenance, or an unknown pitch-guard result cannot produce
`quality_gate_passed=true`.

Version `minus_quality_ensemble.v2` does not treat energy removed from the full
mix as instrumental damage by itself. For each independent vocal proxy it
builds a temporary residual detail reference (`source - vocal_proxy`). RMS,
presence, or air loss becomes a blocker only when at least two independent
residual references agree. Full-mix source conservation remains in `rows.jsonl`
as diagnostic evidence, but it cannot label expected vocal removal as a detail
hole. A single residual-reference vote produces `proxy_unknown`, not a pass.

## Status Contract

- `proxy_blocked`: at least one confirmed defect exists.
- `proxy_unknown`: evidence is incomplete or only one independent proxy votes
  for a vocal defect.
- `proxy_pass`: the proxy ensemble found no blocker and all required evidence
  is complete.

All three statuses remain diagnostic:

- `delivery_allowed=false`
- `listen_pack_allowed=false`
- an exact reference or another explicitly accepted production policy gate is
  still required for delivery.

An API job with `status=ok` proves that inference completed. It does not prove
that the resulting minus passed this quality gate.

## Mac Execution

Neural proxy generation runs through the native Mac MPS API with CPU fallback
disabled. The ensemble audit is CPU/DSP scoring and runs in Apple `container`.
Resampling bytes used by the pitch guard and temporary `source - vocal_proxy`
detail references stay in container-local temporary storage; cleanup of both
scratch groups is recorded in the manifest. `manifest.json`, `matrix.json`,
`rows.jsonl`, pitch evidence, and `progress.jsonl` are written directly into
the mounted Windows workspace.

Example from the Mac host:

```bash
/usr/local/bin/container run --rm \
  --mount type=bind,source=/Users/andrejviluha/AIMusicProject_win,target=/work \
  -e AIMUSIC_CONTAINER_RUNTIME=apple_container \
  --workdir /work/ml_worker_improved \
  bsroformer-worker:latest \
    python tools/audit_minus_quality_ensemble.py \
      /work/artifacts/mac_api/example/run/raw_ref.flac \
      /work/artifacts/mac_api/example/run/instrumental.flac \
      /work/artifacts/mac_api/example/audit \
      --proxy-manifest /work/artifacts/mac_api/example/v1/stem_manifest.json \
      --proxy-manifest /work/artifacts/mac_api/example/v2/stem_manifest.json \
      --runtime-project-root /work \
      --windows-project-root D:/AIMusicProject \
      --producer-project-root /Users/andrejviluha/AIMusicProject_win
```

Generated result names are short and stable:

- `manifest.json`
- `matrix.json`
- `rows.jsonl`
- `progress.jsonl`
- `pitch/pNNN/pitch.json`
- `pitch/pNNN/regions.jsonl`

Track names, raw route ids, and hashes stay in manifest provenance rather than
generated path segments.

## Historical `sm2` v1 Evidence

The old source-only detail gate was run against
`artifacts/mac_api/pf001/sm2/run/instrumental.flac` with independent MelBand and
MDX23C vocal proxies. The result is
`artifacts/mac_api/qa001/ens2/manifest.json`:

- `status=proxy_blocked`
- `quality_gate_passed=false`
- `0-1 s`: vocal leak and full-mix-like consensus;
- `4-5 s`: RMS muffling and presence loss;
- `8-9 s`: presence loss;
- `delivery_allowed=false`.

This v1 evidence still proves the `0-1 s` vocal-consensus blocker, but its
`4-5 s` and `8-9 s` detail labels must not be reused as v2 claims because they
were measured against the full mix. Current claims must point to a fresh v2
manifest. No proxy version can prove the absence of every possible vocal
residue or authorize delivery by itself.
