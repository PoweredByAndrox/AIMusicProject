# Result-to-Claim Prompt

Intended claim:

The current minus hardcase loop can ingest self-contained no-reference judge rows from the it071 minus artifact, train a shadow-only critic, and localize the remaining minus failures without promoting diagnostic audio to a listen/result artifact.

Evidence:

- Source views: `D:/AIMusicProject/artifacts/loop/it075/views/r008`
- Shadow critic: `D:/AIMusicProject/artifacts/loop/it075/tcrit/r005`
- r008 minus view: 204 safe rows, 34 negative labels, 170 positive labels
- r005 validator: ok=true, rows_scored=204, errors=[]
- r005 reliability: AUROC 0.963294, false_accept_rate 0.0, test_negative_rows 12
- r005 safety: shadow_only=true, can_promote_to_production_gate=false, release_ready=false
- it071 nicegreenhorn scored windows: 0-6 and 6-12 seconds blocked with severe phase/detail blockers; three later windows accepted

Known caveats:

- External reviewer/subagent unavailable in this run; verdict is pending external review.
- r005 uses min_negative_test_rows=10 for shadow scoring. It does not satisfy the stricter 20/20/20 negative split contract for production gate promotion.
- No new repaired minus audio has been accepted or delivered.
