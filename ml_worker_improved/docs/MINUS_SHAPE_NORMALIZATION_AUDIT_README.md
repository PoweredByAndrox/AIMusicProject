# Minus Shape Normalization Audit

`tools/audit_minus_shape_normalization_contract.py` is a metadata-only guard for
minus route-expansion runs whose forensics correlate blockers with shape
normalization.

It reads:

- `minus_route_candidate_pool.jsonl` from a route-expansion directory;
- optional `minus_window_blocker_clusters.json` from the matching forensics
  directory.

It writes:

- `shape_normalization_audit.json`
- `raw_rejudge_candidates.jsonl`
- `raw_rejudge_skipped.jsonl`
- `shape_normalization_audit_summary.md`

The tool does not read or export audio. It only materializes candidate rows that
point at existing raw pre-normalization minus/vocal files so the existing
no-reference windowed judge can test whether normalization created or amplified
the minus blockers.

Run:

```bash
python tools/audit_minus_shape_normalization_contract.py \
  --route-dir /path/to/route_expansion \
  --forensics-dir /path/to/forensics \
  --output-dir /path/to/shape_audit
```

Then judge the raw rows:

```bash
python tools/run_no_reference_windowed_judge.py \
  --candidates-jsonl /path/to/shape_audit/raw_rejudge_candidates.jsonl \
  --output-jsonl /path/to/raw_judge/minus_raw_windowed_judge.jsonl \
  --summary-md /path/to/raw_judge/summary.md \
  --cache-jsonl /path/to/raw_judge/cache.jsonl \
  --score-mode minus
```

For Mac-only CPU/DSP downstream gating, run the self-judge matrix explicitly:

```bash
python tools/run_no_reference_minus_self_judge_matrix.py \
  --candidate-jsonl /path/to/shape_audit/raw_rejudge_candidates.jsonl \
  --output-root /path/to/self_judge_matrix \
  --allow-native-runtime
```

`--allow-native-runtime` is only for explicit diagnostic CPU/DSP execution on
Mac-hosted precomputed audio. The default CLI still blocks non-Docker runtime.
The matrix remains fail-closed and must not be treated as a listen/result
artifact.

The same explicit native Mac diagnostic contract is available on the
judge-guided minus repair planning tools:

```bash
python tools/apply_judge_guided_minus_vocal_leakage_repair.py ... \
  --short-output-names \
  --allow-native-runtime

python tools/evaluate_minus_repair_delta_gate.py ... \
  --allow-native-runtime

python tools/plan_minus_repair_delta_router.py ... \
  --allow-native-runtime
```

These repair routes still write candidate-only or plan-only artifacts. A repair
candidate must pass post-repair self-judge, result judge, selector, and listen
preview manifest gates before it can be presented as a result.

Safety contract:

- `delivery_allowed=false`
- `release_ready=false`
- `can_publish_as_result=false`
- no threshold relaxation
- no cleanup/debleed/denoise chain
- no model download or registry edit
- accepted raw rows are candidates for downstream gates only, not listen results
