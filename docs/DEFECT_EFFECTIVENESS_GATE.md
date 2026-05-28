# Defect Effectiveness Gate

`ml_worker_improved/tools/adaptive_vocal_isolation.py` writes
`adaptive_vocal_manifest.json.defect_effectiveness_gate` after candidate scoring and after
`detected_issue_action_contract`.

The action contract answers: "Does every detected issue label have an action candidate or an
explicit rejection?"

The defect effectiveness gate answers: "Is the selected candidate safe to treat as listen-ready?"

## Runtime Status

- `release_ready`: the selected candidate has no release-blocking issue segments and the action
  contract passed.
- `blocked_for_listening`: the selected candidate is diagnostic-only. It still has a severe issue
  segment, clustered issue score, too many issue windows, a missing issue report, or a non-passing
  action contract.

## Post-Gate Residual Repair

When the first selected candidate is `diagnostic_only`, `adaptive_vocal_isolation.py` creates a
second-pass `post_gate_residual_issue_repair` candidate from that selected blocked output. This is a
quality-action rule: a detected audible defect must either produce a targeted repair candidate or an
explicit skip/rejection reason.

The second pass is not a fallback success path. After it is scored, the controller reruns candidate
selection, `detected_issue_action_contract`, `defect_effectiveness_gate`, and
`issue_effectiveness_ledger`. If the new selected output is still blocked, the final manifest remains
`diagnostic_only`.

Explicit skip reasons live in `skipped_post_gate_residual_issue_repair`:
`selected_candidate_release_ready`, `disabled_by_segment_issue_repair_flag`,
`segment_consistency_disabled`, `missing_candidate_issue_reports`, or
`no_post_gate_repairable_issue_window`.

## Manifest Fields

- `thresholds`: named release thresholds for severe segment score, mean top-3 issue score, issue
  window ratio, and minimum material issue improvement.
- `summary`: max issue score, mean top-3 score, issue-window ratio, severe segment count, and
  blocker count.
- `material_improvement`: measured improvement checks from issue-aware, segment-repair, and
  harshness candidates when those metrics exist.
- `blockers`: exact reasons and affected windows. These blockers mean the output can still be used
  for debugging, but must not be presented as a clean/professional selected result.
- `issue_effectiveness_ledger`: per-window debugging ledger. Each row joins selected issue labels,
  segment metrics, action candidates, explicit rejections, uncovered labels, release blockers, and a
  per-window release decision.
- `issue_effectiveness_ledger.windows[*].action_effectiveness`: per-window comparison between the
  selected issue window and available action-candidate issue reports. It records the best measured
  candidate, severity delta, resolved labels, introduced labels, and whether the action improved,
  worsened, was unchanged, or could not be measured.
- `stage_profiler`: per-run stage timing manifest for candidate inference, proxy metrics, segment
  consistency metrics, issue reports, anchor support, listen-pack creation, and other controller
  stages. Heartbeat events are emitted on stderr unless disabled with `--disable-stage-heartbeat`.
- `analysis_cache`: per-run audio/window cache counters. These counters make repeated source/candidate
  reads visible and help distinguish CPU/DSP work from disk IO.

The ledger tolerates small timestamp jitter when matching issue-report windows to action-contract
windows. Malformed issue segments without `start_sec`/`end_sec` are skipped instead of crashing the
controller.

## Release Rule

A selected output may still be copied into the artifact directory for inspection, but
`selected_artifacts.release_status` must be `diagnostic_only` when the gate status is
`blocked_for_listening`.

When the gate is `blocked_for_listening`, the controller must not create the normal selected-vocal
listen-pack item. The manifest must mark:

- `selected_artifacts.delivery_allowed=false`;
- `selected_artifacts.presentation_status=blocked_from_listen_pack`;
- `selected_artifacts.deliverable_vocals_flac=null`;
- `selected_artifacts.deliverable_listen_vocals_flac=null`;
- `selected_artifacts.diagnostic_vocals_flac=<selected artifact path>`;
- `selected_artifacts.listen_pack_skipped_reason=blocked_for_listening`.

Segment issue clips may still be written as forensic evidence, but they are not a production/listen
delivery.

## Feedback Holdout Rule

Listener feedback can use `feedback_split: "train"` or `feedback_split: "holdout"`.
Holdout records are reserved for validation and must not adjust routing priors. This prevents the
adaptive profile selection layer from silently overfitting to the same real-track windows used for
listening evaluation.
