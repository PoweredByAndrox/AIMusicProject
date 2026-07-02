# Iteration 013 - Minus Model Problem-Localization And Tuning Plan

## Prompt Execution

- Template source: `D:\AIMusicProject\prompt-for-loop.txt`
- Template edited: no
- Placeholder task:
  `Улучшить architecture of production minus separation so the system can automatically localize vocal leakage, muffling/detail loss, background/pitched vocal remnants, and candidate regressions on artifacts such as altm003/bck/instrumental.flac, then use those issue labels to tune/adapt the selected minus model instead of cycling through blocked metric gates.`

## Source-Of-Truth Checkpoint

- Current target candidate: `D:\AIMusicProject\artifacts\mac_api\altm003\bck\instrumental.flac`
- Target manifest/status files exist under `D:\AIMusicProject\artifacts\mac_api\altm003\bck\`.
- `altm003_bck` status is diagnostic-only: `delivery_allowed=false`, `delivery_status=diagnostic_candidate_only`, `presentation_status=blocked_until_self_judged`.
- Iteration 065 summary reports `altm003/bck/instrumental.flac` as blocked: `20/27` blocker regions with `vocal_projection_proxy=20`, `vocal_stem_corr_proxy=6`, `full_mix_like_risk_proxy=4`.
- Iteration 065 best preview remains `D:\AIMusicProject\artifacts\loop\it063\seg\lkg001\m001.flac`, also blocked: `14/27`, mostly `presence_loss_proxy=13`, `air_loss_proxy=3`, and only `vocal_projection_proxy=1`.
- `it7dl005/r001/audio/i.flac` exists as an older useful minus-family candidate with score/damage artifacts, not as accepted delivery.
- Iteration 066 Instrumental Resurrection route reached the model on Mac MPS but is blocked by PyTorch MPS complex tensor support: `mps.scatter ... complex<f32>`.
- Current no-reference minus score matrix docs state that the matrix ranks existing evidence and chooses route actions; it does not read audio, train, fine-tune, edit registry, or produce release-ready artifacts.
- Existing critic/localizer design docs already exist: `AUDIO_ERROR_GROUNDING_DATASET.md`, `TENSOR_AUDIO_ERROR_CRITIC.md`, `AUDIO_ERROR_TRAINING_VIEWS.md`, `NO_REFERENCE_LOCAL_CRITIC_TRAINING_README.md`, and `SELF_TRAINING_AUDIO_CONTRACT.md`.

## Reconciliation

The system partially detects that `altm003` is bad: the self-judge catches vocal projection across many windows. It does not yet understand the problem at the level needed for autonomous improvement:

- it does not turn segment issues into supervised or self-supervised training rows;
- it does not learn route/model-specific repair weights from all candidates;
- it does not produce a causal action such as "reduce vocal projection here without creating presence loss there";
- it can rank blocked candidates, but cannot yet adapt the selected minus model from those failures.

Therefore the main architecture gap is not another threshold tweak. The missing implementation link is wiring the current minus candidate/self-judge artifacts into the existing audio-error grounding dataset and critic/localizer route so those failures become training/action targets for a dedicated minus adapter.

## Raw Idea Funnel

1. Add another leakage threshold.
2. Re-run `altm003` with different overlap.
3. Use the best old candidate directly.
4. Blend `altm003` and `it063` globally.
5. Build a segment issue localizer dataset from all candidates.
6. Train a minus-specific adapter on issue-labelled windows.
7. Train separate adapters for vocal, minus, and stem routes.
8. Use all model outputs as teacher ensemble features.
9. Sweep repair weights from `0.0` to `1.0` in fixed steps.
10. Sweep multiple dimensions independently: vocal suppression, detail restore, air restore, source conservation.
11. Add a router that chooses candidate/model per segment.
12. Add a router that chooses adapter weights per segment.
13. Use residual vocals to estimate leakage in minus.
14. Use source-minus reconstruction to detect detail loss and holes.
15. Use old useful candidates as positive priors even when blocked.
16. Store human feedback as optional calibration labels, never as required gate input.
17. Generate pseudo-targets from consensus among multiple model outputs.
18. Penalize full-mix-like minus candidates separately from vocal leakage.
19. Penalize muffling/presence loss separately from vocal suppression.
20. Add a failure memory store keyed by issue signature and model route.
21. Train a single universal adapter for every stem.
22. Train a minus-only adapter first, then generalize to stem/vocal.
23. Use exact reference stems if available, but do not require them.
24. Use only current proxy metrics.
25. Add listen packs before accepted gates.
26. Run a large 10k library pass now.
27. Create a small hardcase pack with representative failures first.
28. Add progress/observability for every matrix/training run.
29. Create accepted-only result-to-claim reports per iteration.
30. Keep diagnostic audio separate from accepted/listen output.

## Status Filtering

- Reject: 1, 2, 24. They keep the system in metric tweaking without causal repair.
- Reject: 3, 25. They would present blocked candidates as results.
- Reject: 26. Too broad before the hardcase localizer is correct.
- Defer: 21. A universal adapter is too broad before minus-specific behavior works.
- Merge: 4, 8, 11, 12, 17. These form a candidate-ensemble problem-localizer and segment router.
- Merge: 5, 6, 9, 10, 13, 14, 18, 19, 20, 22, 23, 27, 28, 29, 30. These form the first production route.

## Serious Ideas

| Idea | Description | Why It Matters | Risk |
| --- | --- | --- | --- |
| A | Minus issue-localizer training rows | Convert mix/candidate/residual windows into issue labels and repair targets | Labels can be noisy |
| B | Minus adapter weight sweep matrix | Run bounded `0.0..1.0` sweeps for suppression/detail/air/conservation knobs | Combinatorial growth |
| C | Candidate ensemble feature bank | Use all existing model outputs as features/teachers, not final truth | Bad candidates can pollute labels |
| D | Segment-level adapter router | Predict per-window adapter weights/actions from issue signatures | Router can overfit one track |
| E | Dedicated minus adapter | Train/tune a minus model separately from vocal and stem routes | Needs strong validation |
| F | Hardcase pack first | Use `altm003`, `it063`, `it7dl005`, `altm002` and similar artifacts before 10k scale | Small pack may miss diversity |
| G | Optional human feedback store | User feedback can calibrate labels but is never required | Feedback can bias if over-weighted |
| H | Accepted-only delivery audit | Keep all tuning candidates diagnostic until gates prove accepted status | Slower visible delivery |

## Scoring Matrix

| Idea | Impact | Evidence Fit | Generality | Time | Risk | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 5 | 5 | 5 | 4 | 3 | 22 |
| B | 4 | 5 | 4 | 4 | 3 | 20 |
| C | 5 | 4 | 5 | 3 | 3 | 20 |
| D | 5 | 4 | 5 | 3 | 2 | 19 |
| E | 5 | 4 | 5 | 2 | 2 | 18 |
| F | 4 | 5 | 4 | 5 | 4 | 22 |
| G | 3 | 3 | 4 | 4 | 3 | 17 |
| H | 5 | 5 | 5 | 4 | 4 | 23 |

## Adversarial Reviewer Pass

- A localizer without audio-grounded features can become another metric table. It must read audio windows and emit feature vectors, issue labels, and action targets.
- A sweep matrix can waste time if it tests many correlated knobs at once. It must start with single-axis sweeps and only then test interactions.
- Ensemble teachers can reinforce bad artifacts. Teacher voting must include negative labels from blocked candidates and source-conservation checks.
- Adapter tuning can destroy musical detail while removing vocals. The objective must include vocal leakage, detail/presence preservation, air/noise floor preservation, and source conservation.
- Human feedback cannot be part of required acceptance. It can only add optional calibration labels.

## Final TOP-3

### TOP-1: Minus Issue-Localizer Dataset Builder

Build a production artifact generator that reads existing candidate manifests and audio windows, then emits training rows:

- mix path;
- candidate minus path;
- optional candidate vocal/residual paths;
- model/route id;
- segment start/end;
- issue labels: `vocal_projection`, `vocal_stem_corr`, `presence_loss`, `air_loss`, `muffling`, `full_mix_like_risk`, `source_conservation`;
- numeric features: RMS ratios, spectral tilt, band energy deltas, vocal-residual correlation, mix-minus reconstruction deltas;
- action target: `suppress_vocal`, `restore_detail`, `restore_air`, `reject_route`, `prefer_candidate`, `needs_model_route`.

This is the first missing implementation link between the current minus evidence pipeline and the already documented critic/localizer architecture.

### TOP-2: Bounded Minus Adapter Sweep Matrix

Create a Mac route that runs a small hardcase matrix over explicit weights:

- vocal suppression weight: `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`;
- detail restoration weight: same grid;
- air/noise-floor restoration weight: same grid;
- conservation penalty weight: same grid.

Run single-axis sweeps first. Only the best safe bands enter two-axis sweeps. Every row must produce manifest, progress JSONL, issue counts, and no accepted delivery unless gates pass.

### TOP-3: Dedicated Minus Adapter/Router

Use TOP-1 rows and TOP-2 sweeps to train or tune a minus-specific adapter/router. The adapter does not replace the selected base model blindly. It controls model route and repair weights per segment:

- if vocal leakage dominates, increase suppression only in affected windows;
- if presence/detail loss dominates, avoid more suppression and prefer detail restoration or another candidate;
- if both occur, choose alternate model/candidate or segment blend;
- if confidence is low, block delivery rather than publish.

## Implementation Plan

1. Build `minus_issue_localizer_dataset` from existing hardcase artifacts, starting with:
   - `altm003/bck/instrumental.flac`
   - `it063/seg/lkg001/m001.flac`
   - `it7dl005/r001/audio/i.flac`
   - `altm002/bci/instrumental.flac`
2. Add a compact manifest format under `artifacts/loop/it067/loc/` with Windows-safe names.
3. Run the dataset builder on Mac so audio analysis uses the approved project path.
4. Analyze whether issue labels match the known regression:
   - `altm003`: high vocal leakage/projection;
   - `it063`: lower vocal leakage but higher detail/presence/air loss;
   - `altm002`: muffling plus leakage.
5. Only after this, run bounded single-axis sweep candidates.
6. Use result-to-claim to decide whether the localizer supports a training/tuning route.

## Result-To-Claim

Supported now:
- Current `altm003` is not acceptable and is worse on vocal projection than the current best blocked preview.
- Current system has scoring and candidate ranking, but lacks a causal minus-training/localization layer.
- The next useful architecture step is a minus issue-localizer dataset plus bounded adapter sweep, not another blind candidate.

Not supported yet:
- That any new minus candidate is accepted.
- That the adapter weights are known.
- That the current metrics fully match human hearing for every defect.

## Next Safe Step

Implement TOP-1 only: wire the existing audio-error grounding dataset / critic-training view to the current minus hardcase pack. It must remain diagnostic-only and must not write listen packs or accepted artifacts.
