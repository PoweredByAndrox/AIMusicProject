# Production Audio Loop Iteration 002

## Template Use

- Source template: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edit status: unchanged
- Placeholder count observed: 1
- Replaced task text for this iteration:

```text
Audit whether failure taxonomy labels from blocked vocal/minus artifacts, including foreign_source_recovered, mix_conservation_failed, vocal leakage, and candidate_delivery_blocked, reliably become hard-eval or negative training examples instead of training positives. Implement the smallest production fix with tests so leaky or blocked candidates improve future training/problem-discovery rather than silently disappearing from the learning loop.
```

## Source-Of-Truth Pass

Inspected evidence:

- `ml_worker_improved/tools/build_generic_finetuning_corpus_from_lab_artifacts.py`
- `ml_worker_improved/tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py`
- `ml_worker_improved/docs/GENERIC_FINETUNING_CORPUS_README.md`
- Current artifact-label aggregate from `artifacts/*.json`, showing repeated labels such as `mix_conservation_failed`, `foreign_source_recovered:*`, `candidate_delivery_blocked`, `separator_server_unavailable`, residual, vocal/voice presence, and GPU memory blockers.

Confirmed current problem:

- `deep_failure_labels()` collected nested taxonomy labels, but `blocked_like()` only checked shallow status fields and `dominant_failure_reason`.
- A manifest could claim `delivery_allowed=true` and `presentation_status=accepted_for_listen_pack` while nested issue labels reported foreign-source recovery or mix-conservation failure.
- In that contradictory state, positive feedback could still create an accepted controller target unless the machine labels were classified as blocked-like first.

## Raw Ideas

| # | Idea | Status | Reason |
|---|---|---|---|
| 1 | Make deep failure labels participate in blocked-like classification | KEEP | Direct root cause and small production fix |
| 2 | Split comma-joined taxonomy labels before hard-eval output | KEEP | Existing manifests contain comma-packed labels |
| 3 | Force blocked-like rows to negative controller targets | KEEP | Prevents accepted-looking claims from training route reuse |
| 4 | Read structured blocker lists such as `blockers` and `failed_checks` | KEEP | Current manifests may carry blocker data as dict/list fields |
| 5 | Preserve a hard-eval reason for pure human-negative feedback | KEEP | Negative examples without machine labels should stay explainable |
| 6 | Add broad model family scoreboard now | DEFER | Useful, but failure labels must first be reliable |
| 7 | Re-score every current artifact now | DEFER | Too broad before corpus routing is fixed |
| 8 | Treat all `issue_labels` as blockers | REJECT | Too broad; benign labels like candidate creation would become failures |
| 9 | Drop all accepted manifests with any nested labels | REJECT | Loses forensic/eval value and over-blocks |
| 10 | Lower accepted-only gates for user-liked artifacts | REJECT | Would mask objective failures |

## TOP-3

1. Deep failure labels participate in blocked-like classification.
   - Implemented in this iteration.
2. Comma-packed taxonomy labels are split into stable hard-eval labels.
   - Implemented in this iteration.
3. Contradictory accepted+blocked rows produce negative controller targets.
   - Implemented in this iteration.

## Implemented In This Set

- Added blocking taxonomy markers for leak/bleed, contamination, `foreign_source`, `mix_conservation`, residual, delivery-pair, separator-server, and GPU-memory failure labels.
- `deep_failure_labels()` now splits comma-packed label strings before dedupe.
- `blocked_like()` now checks both shallow status fields and deep normalized failure labels.
- Structured `blockers`, `blocking_reasons`, and `failed_checks` entries now feed the same normalized failure-label path.
- `inventory_row()` computes failure labels once and passes them into blocked classification.
- `controller_example()` treats any blocked-like row as not accepted, even if the source manifest claims accepted delivery.
- Human negative feedback without machine labels now produces `human_negative_feedback:<rating>` as the hard-eval dominant reason.
- Added a regression test where an accepted-looking minus manifest with positive feedback and nested `foreign_source_recovered`/`mix_conservation_failed` labels routes to `hard_eval_cases.jsonl`, produces no audio-training example, and creates a negative controller target.
- Added regression coverage for accepted-looking structured blockers and pure human-negative hard-eval reasons.
- Updated `GENERIC_FINETUNING_CORPUS_README.md`.

Validation:

- Apple container route:
  - `AIMUSIC_CONTAINER_RUNTIME=apple_container python -m pytest tests/test_build_generic_finetuning_corpus_from_lab_artifacts.py -q`
- Result: `12 passed`.

## Next Loop Task

Use the same template again with this task text:

```text
Audit the accepted-only delivery gates for vocal/minus listen packs and pair manifests. Verify that raw stem manifests, diagnostic artifacts, and accepted-looking candidates with unresolved semantic leak labels cannot be presented as listen/result artifacts. Implement the smallest production validator or gate update with tests that blocks any artifact missing accepted delivery status, objective gate evidence, or semantic leak/failure clearance.
```
