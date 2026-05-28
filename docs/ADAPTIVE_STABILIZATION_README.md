# Adaptive Stabilization README

## Why this exists

The project already has enough model runs, gates, manifests and A/B artifacts to stop blind experimentation. The next step is a policy layer that turns evidence into controlled decisions.

## Core contract

```text
manifest evidence -> dominant failure class -> allowed actions -> forbidden actions -> final decision
```

## First implementation boundary

This patch intentionally does not modify model inference. It only adds:

```text
tools/adaptive_policy.py
tools/build_adaptive_decision_report.py
tools/build_hard_case_dashboard.py
tests/test_adaptive_policy.py
```

## Main anti-chaos rule

`candidate_space_exhausted` means stop. It does not mean run another model.

## Production integration

`tools/adaptive_vocal_isolation.py` now writes
`adaptive_vocal_manifest.json.decision_contract` and mirrors the same contract
into `selected_delivery_gate_manifest.json.decision_contract`.

The contract is built from:

```text
detected_issue_action_contract
defect_effectiveness_gate
vocal_next_action_plan
selected_artifacts.presentation_status
```

Post-hoc report tools prefer an embedded `decision_contract` when present and
only compute one from raw evidence for older manifests.

## Required release behavior

A release-ready artifact must have:

```text
qc passed
no forbidden action used
target issue improved
no blocker regression
final decision gate passed
```

## Hard-case behavior

A hard-case track must not generate more candidates automatically. It must go into:

```text
hard_case_dashboard
fine_tune_candidate list
manual 8-15 second AB review pack
```

Model-zoo benchmark manifests are treated as exhausted candidate space when
`next_action.status` or `overall.status` is `no_accepted_candidate` and the
bounded rerun contract declares `candidate_space_exhausted`. That terminal
benchmark state dominates lower-level metric noise in the control-plane report.

## Report scan safety

`build_adaptive_decision_report.py` and `build_hard_case_dashboard.py` skip
individual JSON files larger than 20 MB and mark them as
`read_skipped_large_json`. This keeps large forensic dumps from turning the
control-plane scan into an unbounded memory load.
