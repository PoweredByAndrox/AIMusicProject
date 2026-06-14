# MASTER PROMPT — AUTONOMOUS NO-REFERENCE VOCAL / MINUS IMPROVEMENT SYSTEM

Default language: Russian for reports. Code/comments may stay English if the project already uses English.

You are an engineering agent working inside the existing `D:\AIMusicProject` repository.

Your job is NOT to continue creating endless metadata phases.
Your job is to build a practical autonomous system that improves vocal/minus separation results for normal tracks WITHOUT requiring stems, reference vocal/minus files, or human CSV review as the main path.

The user's real goal:

```text
Input: ordinary mixed track, usually without stems/reference files.
System: generate/collect multiple vocal/minus candidates, judge them without reference stems, select or improve the best result, reject unsafe results, and produce usable preview/listen candidates.
Output: best available vocal/minus result or honest blocked/hard-case decision.
```

The user is angry because previous work drifted into supervised fine-tune/reference-file bureaucracy. Do not repeat that mistake.

---

## 0. Core Truth

Reference stems are useful only as optional calibration/validation/fine-tune data.
They must NOT be required for normal production behavior.

The main pipeline must work like this:

```text
ordinary track
→ generate or collect existing candidate vocal/minus pairs
→ no-reference windowed judge
→ optimizer chooses best candidate or keeps original
→ if full-candidate selection is too coarse:
   segment-level planner finds local improvements
→ dry-run validates blend safety
→ preview render only for safe windows
→ targeted post-render judge
→ if improved and safe:
   promote to listen-preview candidate, not release-ready
→ if bad:
   rollback/block
```

---

## 1. Non-Negotiable Prohibitions

You must NOT:

- ask for stems/reference files as the main path;
- require human CSV review as the main path;
- start fine-tune;
- train models;
- add new models;
- add new checkpoints/weights;
- edit `worker/model_registry.py`;
- add random cleanup/debleed/denoise chains;
- relax thresholds/gates to force acceptance;
- export diagnostic audio as final result;
- mark anything `release_ready` without explicit delivery contract;
- accept “best of bad”;
- overwrite original candidate files;
- mutate source manifests directly unless a separate approved patch says so;
- hide failures behind optimistic summaries.

If a step cannot produce a safe candidate, it must output:

```text
blocked_hard_case
needs_more_candidates
needs_segment_blend
preview_only_not_release
```

not fake success.

---

## 2. Source-of-Truth Files To Read First

Before changing code, read relevant current source files and artifacts.

### Existing no-reference stack

- `tools/run_no_reference_result_judge.py`
- `tools/run_no_reference_candidate_selector.py`
- `tools/run_no_reference_optimizer.py`
- `tools/collect_no_reference_candidate_pool.py`
- `tools/run_no_reference_windowed_judge.py`
- `tools/plan_no_reference_segment_blends.py`
- `tools/prepare_no_reference_segment_blend_dry_run.py`
- `tools/render_no_reference_segment_blend_preview.py`
- `tools/judge_no_reference_segment_blend_preview.py`

### Related control/safety tools

- `tools/artifact_status_contract.py`
- `tools/adaptive_policy.py`
- `tools/stem_quality_control.py`
- `tools/instrumental_self_judge_gate.py`
- `tools/teacher_calibrated_instrumental_selector.py`
- `tools/adaptive_vocal_isolation.py`

### Existing docs

- `docs/NO_REFERENCE_RESULT_LOOP_README.md`
- `docs/NO_REFERENCE_OPTIMIZER_README.md`
- `docs/NO_REFERENCE_CANDIDATE_POOL_README.md`
- `docs/NO_REFERENCE_WINDOWED_JUDGE_README.md`
- `docs/NO_REFERENCE_SEGMENT_BLEND_PLANNER_README.md`
- `docs/NO_REFERENCE_SEGMENT_BLEND_DRY_RUN_README.md`
- `docs/NO_REFERENCE_SEGMENT_BLEND_PREVIEW_README.md`
- `docs/NO_REFERENCE_SEGMENT_BLEND_PREVIEW_JUDGE_README.md`
- `docs/CONTROL_PLANE_AGENT_README.md`

### Current artifacts to inspect

Use latest available paths under:

```text
D:\AIMusicProject\lab_artifacts\no_reference_optimizer_20260531
```

Especially:

- `no_reference_candidate_pool_300.jsonl`
- `no_reference_candidate_pool_300_summary.md`
- `no_reference_windowed_judge_300.jsonl`
- `no_reference_windowed_judge_300_summary.md`
- `no_reference_optimizer_windowed_300_result.jsonl`
- `no_reference_optimizer_windowed_300_summary.md`
- `no_reference_segment_blend_plan.jsonl`
- `no_reference_segment_blend_summary.md`
- `no_reference_segment_blend_dry_run.jsonl`
- `no_reference_segment_blend_render_contract.jsonl`
- `no_reference_segment_blend_dry_run_summary.md`
- `no_reference_segment_blend_preview_manifest.json`
- `no_reference_segment_blend_preview_summary.md`
- `no_reference_segment_blend_preview_window_judge.jsonl`
- `no_reference_segment_blend_preview_window_judge_summary.md`

---

## 3. Current Known State

The project has already achieved:

### Candidate collection

```text
total_candidate_rows_discovered: 18529
complete_candidate_pairs: 223
tracks_discovered: 148
tracks_with_2plus_candidates: 28
```

### Windowed judge

```text
candidates_judged: 223
windows_evaluated: 1115
average_windows_per_candidate: 5
accept_candidate: 45
blocked_hard_case: 178
```

Top no-reference failure reasons:

```text
vocal_leakage_in_minus
silence_hallucination
phase_or_stereo_inconsistency
bad_mixture_reconstruction
instrumental_leakage_in_vocal
thin_or_overcleaned_vocal
transient_damage
```

### Optimizer

```text
total_tracks_optimized: 55
optimized_candidate_selected: 4
original_candidate_kept: 25
blocked_hard_case: 26
average_fitness_before: 0.382478
average_fitness_after: 0.399734
average_improvement_delta: 0.017257
usable_without_stems: 29
```

Top rejected actions:

```text
minus_regression_severe
new_severe_blocker
vocal_regression_severe
```

### Segment planner

```text
total_tracks_analyzed: 55
tracks_with_segment_blend_plan: 6
total_segment_blend_windows: 25
use_alt_vocal_window: 10
use_alt_minus_window: 14
use_alt_both_window: 1
average_segment_improvement_delta: 0.053987
```

### Dry-run

```text
total_planned_windows: 25
render_ready: 1
blocked_overlap_conflict: 24
```

### Preview render

```text
previews_rendered: 1
previews_skipped: 24
duration/sample_rate/channels preserved: true
can_publish_as_result: false
can_mark_release_ready: false
```

### Targeted preview judge

```text
preview_improved_window: 1
preview_noop_self_blend: 0
preview_regressed_window: 0
score_delta: +0.013593
allowed_next_step: expand_preview_policy
```

Interpretation:

```text
The no-reference path works.
It has produced one confirmed improved preview window.
The next bottleneck is overlap/conflict resolution and safe expansion, not stems/fine-tune.
```

---

## 4. Global Architecture To Build

The final system must contain these modules.

### A. Candidate Pool Collector

Purpose:

```text
Find all existing generated vocal/minus candidates.
Group them by track.
Deduplicate identical pairs.
Skip incomplete pairs with explicit reasons.
```

Required behavior:

- no stems required;
- no reference files required;
- no audio copied/exported;
- output deterministic JSONL;
- group by `track_id`;
- each row must include:
  - `track_id`
  - `candidate_id`
  - `mix_path`
  - `vocal_path`
  - `minus_path`
  - `source_manifest`
  - `candidate_origin`
  - `candidate_family`
  - `is_complete_pair`
  - `skip_reason`

Success criteria:

```text
tracks_with_2plus_candidates > 0
complete_candidate_pairs > previous useful baseline
```

---

### B. Windowed No-Reference Judge

Purpose:

```text
Evaluate candidates without reference stems using deterministic windows.
```

Window strategy:

- start usable window;
- middle window;
- end/tail window;
- high-energy window;
- low-energy/silence window.

Default:

```text
max_windows_per_candidate = 5
window_duration = 12s
```

Metrics/proxies:

- mixture consistency;
- vocal leakage in minus;
- instrumental leakage in vocal;
- silence hallucination;
- robotic/musical noise proxy;
- thin vocal proxy;
- phase/stereo consistency;
- transient damage proxy.

Aggregation:

```text
median for stable metrics
max for severe defects
min for confidence-sensitive scores
```

Output must include candidate-level compatible scores:

- `vocal_score`
- `minus_score`
- `artifact_score`
- `consistency_score`
- `defect_reduction_score`
- `final_fitness`
- `dominant_defects`
- `blockers`

Hard blockers:

- severe vocal leakage in minus;
- severe instrumental leakage in vocal;
- severe phase collapse;
- severe silence hallucination;
- severe robotic artifact;
- severe mixture reconstruction drift;
- missing or incompatible audio.

---

### C. Fitness Optimizer

Purpose:

```text
Choose the best full candidate per track without stems.
Rollback if worse.
Never accept best-of-bad.
```

Fitness formula:

```text
final_fitness =
  0.30 * min(vocal_score, minus_score)
+ 0.25 * artifact_score
+ 0.25 * consistency_score
+ 0.20 * defect_reduction_score
```

Why `min(vocal_score, minus_score)`:

```text
If vocal is clean but minus is destroyed, the result is bad.
If minus is clean but vocal is destroyed, the result is bad.
Averaging hides failures.
```

Optimizer decisions:

- `optimized_candidate_selected`
- `original_candidate_kept`
- `needs_segment_blend`
- `blocked_hard_case`

Hard reject rules:

- new severe blocker;
- minus regression severe;
- vocal regression severe;
- artifact regression severe;
- fitness delta below minimum;
- missing path;
- incompatible shape.

---

### D. Segment Blend Planner

Purpose:

```text
When full candidate selection is too coarse, find local windows where an alternate candidate is better.
```

Rules:

- no audio render;
- no export;
- no stems;
- no reference;
- only use windowed judge evidence;
- propose:
  - `use_alt_vocal_window`
  - `use_alt_minus_window`
  - `use_alt_both_window`
  - `keep_base`

Segment proposal rules:

```text
If alternate improves vocal but damages minus:
  propose vocal-only window.

If alternate improves minus but damages vocal:
  propose minus-only window.

If both improve:
  propose both-window.

If blocker appears:
  reject.
```

---

### E. Non-Overlap Conflict Resolver

Purpose:

```text
Turn many overlapping planned windows into a safe non-overlapping subset.
```

This is the immediate next major target.

Inputs:

- segment blend plan;
- dry-run contract;
- preview-window judge result;
- windowed judge scores;
- candidate pool.

Rules:

- no rendering;
- no export;
- select non-overlapping windows per track/stem;
- use pre-roll/post-roll margin;
- if overlap group exists, keep highest expected improvement;
- prefer window matching improved metric;
- expansion allowed only because one preview already improved;
- output deterministic.

Output:

- `no_reference_nonoverlap_blend_subset.jsonl`
- `no_reference_nonoverlap_blend_subset_summary.md`

Each selected row:

- `track_id`
- `base_candidate_id`
- `alternate_candidate_id`
- `source_stem`
- `window_start_sec`
- `window_duration_sec`
- `expected_improvement_delta`
- `overlap_group_id`
- `selection_reason`
- `conflict_resolution_status`
- `can_render_preview_later`
- `can_publish_as_result=false`
- `can_mark_release_ready=false`

Success criteria:

```text
selected_nonoverlap > 1
tracks_with_selected_windows >= 1
average_expected_improvement_delta > 0
```

---

### F. Dry-Run Render Contract

Purpose:

```text
Before rendering, validate audio compatibility and boundaries.
```

Must check:

- source paths exist;
- sample rate compatible;
- channels compatible;
- duration compatible;
- boundaries safe;
- no unsafe overlap;
- crossfade/pre-roll/post-roll valid.

No render yet.

---

### G. Preview Renderer

Purpose:

```text
Render bounded preview audio only after dry-run says ready.
```

Rules:

- preview only;
- never overwrite originals;
- never release;
- never final;
- output to preview folder;
- max render count by argument;
- preserve sample rate/channels/duration;
- write manifest.

---

### H. Targeted Preview Judge

Purpose:

```text
Judge only the affected window after preview render.
```

Must compare:

- base window;
- preview window;
- blocker delta;
- score delta.

Decisions:

- `preview_improved_window`
- `preview_noop_self_blend`
- `preview_regressed_window`
- `preview_inconclusive`

Only if preview improves and no new blocker:

```text
allowed_next_step = expand_preview_policy
```

---

### I. Delivery Contract

Even after improvement, result must not automatically become release.

Delivery statuses:

- `diagnostic_only`
- `preview_only`
- `accepted_for_listen_pack`
- `release_ready`

Rules:

```text
preview render != final result
diagnostic != result
blocked != result
release_ready requires explicit delivery contract
```

---

## 5. Immediate Next Step

The next concrete task is NOT fine-tune, NOT stems, NOT another generic report.

The next task is:

```text
Implement Non-Overlapping Segment Blend Expansion Policy.
```

Why:

- one preview improved;
- 24 windows were blocked by overlap;
- we need a deterministic resolver before rendering more previews.

---

## 6. Immediate Next Prompt For Agent

```text
Phase NR-8: Non-Overlapping Segment Blend Expansion Policy

Do not add models.
Do not add weights.
Do not start fine-tune.
Do not require stems.
Do not require reference vocal/minus.
Do not require human CSV.
Do not add cleanup/debleed/denoise chains.
Do not relax thresholds.
Do not mark output release_ready.
Do not publish as final result.
Do not render audio in this phase.

Context:
NR-7A/7B confirmed:
- preview rows judged=1
- preview_noop_self_blend=0
- preview_improved_window=1
- average_score_delta=0.013593
- allowed_next_step=expand_preview_policy
- can_publish_as_result=false
- can_mark_release_ready=false

NR-6 showed:
- total planned windows=25
- render_ready=1
- blocked_overlap_conflict=24

Goal:
Create a non-overlapping segment blend expansion policy. It must choose a safe non-overlapping subset of windows with highest expected improvement. It must not render audio.

Inputs:
- /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl
- /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl
- /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl
- /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl
- /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl

Read:
- tools/plan_no_reference_segment_blends.py
- tools/prepare_no_reference_segment_blend_dry_run.py
- tools/judge_no_reference_segment_blend_preview.py
- docs/NO_REFERENCE_SEGMENT_BLEND_PLANNER_README.md
- docs/NO_REFERENCE_SEGMENT_BLEND_DRY_RUN_README.md
- docs/NO_REFERENCE_SEGMENT_BLEND_PREVIEW_JUDGE_README.md

Create:
- tools/plan_no_reference_nonoverlap_blend_subset.py
- tests/test_plan_no_reference_nonoverlap_blend_subset.py
- docs/NO_REFERENCE_NONOVERLAP_BLEND_SUBSET_README.md

Outputs:
- no_reference_nonoverlap_blend_subset.jsonl
- no_reference_nonoverlap_blend_subset_summary.md

Each selected row must include:
- track_id
- base_candidate_id
- alternate_candidate_id
- source_stem
- window_start_sec
- window_duration_sec
- expected_improvement_delta
- overlap_group_id
- selected_rank_in_group
- selection_reason
- conflict_resolution_status:
  - selected_nonoverlap
  - rejected_lower_score_overlap
  - rejected_safety_blocker
  - rejected_preview_policy_not_met
- can_render_preview_later
- can_publish_as_result=false
- can_mark_release_ready=false
- forbidden_actions

Rules:
1. Do not render audio.
2. Do not export audio.
3. Do not require stems/reference.
4. Do not train.
5. Only select windows that pass safety blockers.
6. Windows on same track/stem cannot overlap after pre-roll/post-roll margin.
7. If windows overlap, keep highest expected improvement delta.
8. Prefer windows whose source_stem matches improved metric:
   - vocal improvement -> vocal
   - minus improvement -> minus
   - both improvement -> both only if both scores improve.
9. Use NR-7A result as calibration: expansion allowed only if preview_improved_window >= 1.
10. Output deterministic.
11. No publish/release.

Tests:
- selects highest score from overlapping group.
- keeps non-overlapping windows.
- rejects overlapping lower-score windows.
- blocks expansion if preview did not improve.
- publish/release false.
- no audio rendered.
- deterministic output.
- summary counts match JSONL.

Run:
docker compose run --rm --no-deps ml-worker pytest \
  tests/test_plan_no_reference_nonoverlap_blend_subset.py \
  tests/test_judge_no_reference_segment_blend_preview.py \
  tests/test_prepare_no_reference_segment_blend_dry_run.py \
  -q

Then:
docker compose run --rm --no-deps ml-worker python tools/plan_no_reference_nonoverlap_blend_subset.py \
  --segment-blend-plan-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl \
  --dry-run-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_dry_run.jsonl \
  --preview-window-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl \
  --windowed-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl \
  --candidate-pool-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl \
  --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_blend_subset.jsonl \
  --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_nonoverlap_blend_subset_summary.md

Final report:
- total planned windows
- overlap groups
- selected_nonoverlap
- rejected_lower_score_overlap
- rejected_safety_blocker
- rejected_preview_policy_not_met
- tracks_with_selected_windows
- vocal_windows_selected
- minus_windows_selected
- both_windows_selected
- average_expected_improvement_delta
- can_render_preview_later count
- can_publish_as_result count, expected 0
- can_mark_release_ready count, expected 0
- safety flags
```

---

## 7. Longer-Term Roadmap To Real Result

### Stage 1 — Make no-reference candidate selection strong

Goal:

```text
More tracks, more candidates, better selection.
```

Work:

- raise candidate cap safely;
- improve deduplication;
- improve candidate identity/path distinctness;
- cache window metrics;
- optimize runtime.

Success:

```text
hundreds/thousands candidates judged
track coverage grows
more optimized candidates selected
fitness_after consistently > before
```

---

### Stage 2 — Make segment blending usable

Goal:

```text
Use local improvements without ruining full track.
```

Work:

- non-overlap subset selection;
- bounded preview render;
- targeted post-render judge;
- expand only if improved.

Success:

```text
multiple preview windows improve
no new severe blockers
full-track preview score improves
```

---

### Stage 3 — Promote to listen pack, not release

Goal:

```text
Create listenable preview results.
```

Work:

- delivery contract;
- no diagnostic-as-result;
- no release-ready;
- accepted_for_listen_pack only after:
  - no-reference judge passes;
  - preview improves;
  - no severe blockers;
  - consistency is stable.

Success:

```text
usable_without_stems grows
accepted_for_listen_pack > 0
blocked_hard_case stays honest
```

---

### Stage 4 — Optional reference/stem calibration

Only if references exist.

Purpose:

- calibrate weights;
- validate judge;
- tune thresholds;
- build small supervised set.

Never make this mandatory.

---

## 8. Definition Of Done

The system is considered useful when:

```text
1. It works on ordinary tracks without stems.
2. It collects multiple candidates per track.
3. It judges them with cached/windowed no-reference metrics.
4. It selects better full candidates when available.
5. It identifies local windows for segment replacement.
6. It renders preview-only blends safely.
7. It judges preview windows after render.
8. It never publishes diagnostic/bad outputs as final.
9. It can produce accepted_for_listen_pack candidates.
10. It blocks bad/hard cases honestly.
```

Quantitative target for first meaningful milestone:

```text
tracks optimized >= 50
usable_without_stems >= 25
optimized_candidate_selected >= 5
segment previews improved >= 3
accepted_for_listen_pack >= 1
release_ready = 0 unless explicitly approved
```

---

## 9. Final Reporting Format For Every Run

Every report must include:

```text
1. Short execution plan
2. Confirmed vs unconfirmed
3. Files created/updated
4. Tests run and exact result
5. Real artifact counts
6. Improvement numbers:
   - before score
   - after score
   - delta
   - candidates selected
   - previews improved
7. Safety flags:
   - reference_required
   - human_csv_required
   - training_started
   - fine_tune_started
   - models_changed
   - model_registry_edited
   - cleanup_routes_added
   - thresholds_relaxed
   - diagnostic_audio_exported
   - audio_exported
   - audio_copied
8. Next safe step
```

No vague “it should be better”.
Only evidence-backed statements.

---

## 10. Critical Mindset

Do not optimize for making reports look successful.
Optimize for real no-reference improvement.

A good result:

```text
selected better candidate
rendered preview improved
bad candidate blocked
```

A bad result:

```text
more metadata summaries
more reference-file requests
more gates
more cleanup chains
fake acceptance
```

The user wants the system to actually improve vocal/minus output without stems.
Keep that as the center of the work.
