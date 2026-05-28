"""Compare two baseline snapshots produced by baseline_snapshot.py.

Diff rules per stem (key = relative wav path):
- IDENTICAL    : sha256 matches => stem byte-for-byte identical
- TOLERATED    : sha256 differs but every numeric metric is within tolerance
- CHANGED      : at least one numeric metric outside tolerance
- MISSING      : present in baseline, absent in candidate
- NEW          : absent in baseline, present in candidate

Tolerances default to (relative, absolute):
- peak_abs / rms / spectral_flatness_mean : (0.02, 1e-4)
- spectral_centroid_mean_hz / std         : (0.05, 5.0)
- peak_position_sec                       : (0.0, 0.05)  (50 ms wiggle)
- duration_sec / sample_rate / channels   : exact

Exit codes: 0 if no CHANGED/MISSING/NEW (only IDENTICAL+TOLERATED), 1 otherwise.

Usage:
    py scripts/baseline_compare.py --baseline <a.json> --candidate <b.json> \\
        [--out delta.json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

EXACT_FIELDS = ("sample_rate", "channels", "sample_width_bytes", "frames")
TOL_FIELDS: dict[str, tuple[float, float]] = {
    "peak_abs": (0.02, 1e-4),
    "rms": (0.02, 1e-4),
    "spectral_flatness_mean": (0.05, 1e-4),
    "spectral_centroid_mean_hz": (0.05, 5.0),
    "spectral_centroid_std_hz": (0.10, 10.0),
    "duration_sec": (0.0, 0.001),
    "peak_position_sec": (0.0, 0.05),
}


def _within_tolerance(baseline: float, candidate: float, rel: float, abs_tol: float) -> bool:
    if math.isnan(baseline) or math.isnan(candidate):
        return False
    diff = abs(candidate - baseline)
    if diff <= abs_tol:
        return True
    if rel <= 0.0:
        return False
    denom = max(abs(baseline), abs(candidate), 1e-12)
    return (diff / denom) <= rel


def diff_stem(baseline: dict, candidate: dict) -> tuple[str, list[dict]]:
    field_diffs: list[dict] = []

    for field in EXACT_FIELDS:
        b = baseline.get(field)
        c = candidate.get(field)
        if b != c:
            field_diffs.append({
                "field": field,
                "baseline": b,
                "candidate": c,
                "rule": "exact",
                "ok": False,
            })

    for field, (rel, abs_tol) in TOL_FIELDS.items():
        b = baseline.get(field)
        c = candidate.get(field)
        if b is None or c is None:
            field_diffs.append({
                "field": field,
                "baseline": b,
                "candidate": c,
                "rule": f"rel={rel},abs={abs_tol}",
                "ok": b == c,
            })
            continue
        ok = _within_tolerance(float(b), float(c), rel, abs_tol)
        if not ok:
            field_diffs.append({
                "field": field,
                "baseline": b,
                "candidate": c,
                "delta": round(float(c) - float(b), 8),
                "rule": f"rel={rel},abs={abs_tol}",
                "ok": False,
            })

    if baseline.get("sha256") == candidate.get("sha256") and not field_diffs:
        return "IDENTICAL", field_diffs
    if not field_diffs:
        return "TOLERATED", field_diffs
    return "CHANGED", field_diffs


def compare_snapshots(baseline: dict, candidate: dict) -> dict:
    base_stems = baseline.get("stems", {})
    cand_stems = candidate.get("stems", {})
    keys: set[str] = set(base_stems) | set(cand_stems)

    rows: list[dict] = []
    counts = {"IDENTICAL": 0, "TOLERATED": 0, "CHANGED": 0, "MISSING": 0, "NEW": 0}

    for key in sorted(keys):
        if key not in cand_stems:
            counts["MISSING"] += 1
            rows.append({"path": key, "status": "MISSING", "fields": []})
            continue
        if key not in base_stems:
            counts["NEW"] += 1
            rows.append({"path": key, "status": "NEW", "fields": []})
            continue
        status, fields = diff_stem(base_stems[key], cand_stems[key])
        counts[status] += 1
        rows.append({"path": key, "status": status, "fields": fields})

    return {
        "schema": "stemlab.baseline.diff.v1",
        "baseline_label": baseline.get("label", ""),
        "candidate_label": candidate.get("label", ""),
        "summary": counts,
        "rows": rows,
    }


def render_summary(report: dict) -> Iterable[str]:
    summary = report["summary"]
    yield f"baseline label : {report.get('baseline_label','')!r}"
    yield f"candidate label: {report.get('candidate_label','')!r}"
    for key in ("IDENTICAL", "TOLERATED", "CHANGED", "MISSING", "NEW"):
        yield f"  {key:<10}: {summary.get(key, 0)}"
    if summary.get("CHANGED"):
        yield ""
        yield "CHANGED stems:"
        for row in report["rows"]:
            if row["status"] != "CHANGED":
                continue
            yield f"  - {row['path']}"
            for field in row["fields"]:
                yield (
                    f"      {field['field']}: "
                    f"baseline={field.get('baseline')!r} -> candidate={field.get('candidate')!r}"
                    + (f" (delta={field['delta']})" if "delta" in field else "")
                )
    if summary.get("MISSING"):
        yield ""
        yield "MISSING stems (in baseline, gone in candidate):"
        for row in report["rows"]:
            if row["status"] == "MISSING":
                yield f"  - {row['path']}"
    if summary.get("NEW"):
        yield ""
        yield "NEW stems (only in candidate):"
        for row in report["rows"]:
            if row["status"] == "NEW":
                yield f"  - {row['path']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default=None, help="optional JSON delta file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat TOLERATED as failure too (only IDENTICAL passes)",
    )
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))

    report = compare_snapshots(baseline, candidate)

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for line in render_summary(report):
        print(line)

    summary = report["summary"]
    bad = summary["CHANGED"] + summary["MISSING"] + summary["NEW"]
    if args.strict:
        bad += summary["TOLERATED"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
