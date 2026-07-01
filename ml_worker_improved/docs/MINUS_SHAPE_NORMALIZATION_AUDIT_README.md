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

Safety contract:

- `delivery_allowed=false`
- `release_ready=false`
- `can_publish_as_result=false`
- no threshold relaxation
- no cleanup/debleed/denoise chain
- no model download or registry edit
- accepted raw rows are candidates for downstream gates only, not listen results
