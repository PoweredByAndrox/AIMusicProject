# No-Reference Minus Listen Preview

`tools/build_no_reference_minus_listen_preview_manifest.py` creates a contract-safe listen-preview manifest for a minus stem only. It does not copy audio, export audio, run models, train, add cleanup/debleed/denoise, or mark anything `release_ready`.

The manifest can become `accepted_for_listen_pack` only when all evidence agrees:

- the minus segment-switch candidate was rendered as preview/candidate only;
- post-render `instrumental_self_judge_gate.py` reports `overall.status=proxy_pass`;
- post-render self-judge has `blocker_count=0`;
- no-reference result judge reports `decision=accept_candidate` and explicit
  minus evidence: `minus_score`, `minus_decision=accept_candidate`,
  `minus_blockers=[]`, and `window_count >= 5`;
- no-reference result judge includes `minus_musical_detail_preservation_gate` with `status=proxy_pass`,
  no blockers, and finite required metrics for instrumental integrity, transient preservation,
  spectral balance, phase/stereo stability, and mixture consistency;
- no-reference selector reports `decision=accept_candidate`;
- result judge and selector evidence use the same `candidate_id` as the candidate manifest;
- every evidence file points to the same rendered minus path.

Accepted output is still listen-preview only:

- `delivery_status=accepted`
- `delivery_allowed=true`
- `presentation_status=accepted_for_listen_pack`
- `release_status=listen_preview_only`
- `stem_type=minus`
- `can_mark_release_ready=false`

This does not relax the existing pair dense delivery gate. Pair/vocal delivery still requires its own compatible evidence.
Generic `decision=accept_candidate` or `overall_no_reference_score` is diagnostic context only and cannot substitute for explicit minus evidence.

If the musical-detail gate is missing or blocked, the manifest is blocked with
`minus_detail_preservation_gate_missing`, `minus_detail_preservation_gate_blockers`,
`minus_detail_preservation_gate_not_passed`, or `minus_detail_preservation_metrics_missing`.
This prevents a minus candidate with low vocal leakage but audible music holes, transient smearing, or
spectral damage from being promoted as an accepted listen-preview.
