# MASTER PROMPT — MODE-SPECIFIC NO-REFERENCE SCORE MATRIX FOR VOCAL / MINUS / BOTH

Default language for final reports: Russian.
Code may stay English if the project already uses English.

You are working inside the existing `D:\AIMusicProject` repository.

The user’s core requirement:

```text
The no-reference system must NOT use one shared scoring/test matrix for vocal and minus.
Vocal and minus must be evaluated and optimized separately, because a model/candidate can be good for vocal but bad for minus, or good for minus but bad for vocal.
```

This task is about splitting the matrix by mode:

```text
mode=vocal
mode=minus
mode=both
```

Do not drift into stems/reference files/fine-tune/human CSV.
This is a no-reference production scoring architecture change.

---

## 1. Problem

Current no-reference optimizer still behaves too much like:

```text
candidate = vocal + minus pair
one shared score
one shared decision
```

This is wrong.

Why:

```text
candidate A can produce a better vocal but worse minus.
candidate B can produce a worse vocal but better minus.
A single score hides this and creates bad compromises.
```

Known symptom from current artifacts:

```text
minus_regression_severe appears repeatedly.
```

That means the system often rejects candidates because the minus gets worse even if some other part improves.

The correct design:

```text
vocal decision uses vocal matrix only.
minus decision uses minus matrix only.
both mode only checks final compatibility of selected vocal + selected minus.
```

---

## 2. Non-Negotiable Prohibitions

Do NOT:

- add models;
- add weights/checkpoints;
- start fine-tune;
- require stems;
- require reference vocal/minus;
- require human CSV;
- add cleanup/debleed/denoise chains;
- relax thresholds;
- mark anything `release_ready`;
- publish as final result;
- accept best-of-bad;
- average vocal and minus into one shared score;
- use `both` mode as default selector;
- let a vocal improvement hide a minus failure;
- let a minus improvement hide a vocal failure.

---

## 3. Source-of-Truth Files To Read

Before changing code, inspect the current project files:

```text
tools/run_no_reference_windowed_judge.py
tools/run_no_reference_optimizer.py
tools/plan_no_reference_segment_blends.py
tools/prepare_no_reference_segment_blend_dry_run.py
tools/judge_no_reference_segment_blend_preview.py
tools/run_no_reference_result_judge.py
tools/run_no_reference_candidate_selector.py
tools/artifact_status_contract.py
tools/adaptive_policy.py
```

Docs:

```text
docs/NO_REFERENCE_RESULT_LOOP_README.md
docs/NO_REFERENCE_OPTIMIZER_README.md
docs/NO_REFERENCE_WINDOWED_JUDGE_README.md
docs/NO_REFERENCE_SEGMENT_BLEND_PLANNER_README.md
docs/NO_REFERENCE_SEGMENT_BLEND_DRY_RUN_README.md
docs/NO_REFERENCE_SEGMENT_BLEND_PREVIEW_JUDGE_README.md
```

Artifacts to inspect if available:

```text
lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl
lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300.jsonl
lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_windowed_300_result.jsonl
lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_plan.jsonl
lab_artifacts/no_reference_optimizer_20260531/no_reference_segment_blend_preview_window_judge.jsonl
```

---

## 4. Goal

Create mode-specific scoring and decision logic.

Required modes:

```text
vocal
minus
both
```

Default production path:

```text
1. Run vocal mode separately.
2. Run minus mode separately.
3. Select best vocal candidate.
4. Select best minus candidate.
5. Run both mode only for pair compatibility.
6. If compatible, allow preview/listen candidate flow.
7. If incompatible, output needs_pair_reconciliation or segment_blend_needed.
```

Do not use `both` as default optimizer selector.

---

## 5. Required New Module

Create:

```text
tools/no_reference_score_matrix.py
tests/test_no_reference_score_matrix.py
docs/NO_REFERENCE_SCORE_MATRIX_README.md
```

The module must expose:

```python
score_candidate(metrics: dict, mode: str) -> dict
detect_blockers(metrics: dict, mode: str) -> list[str]
dominant_defect(metrics: dict, mode: str) -> str | None
allowed_actions_for_defect(defect: str, mode: str) -> dict
pair_compatibility_score(vocal_candidate: dict, minus_candidate: dict) -> dict
```

Every output must be deterministic.

Invalid mode must raise a clear error.

Allowed modes:

```text
vocal
minus
both
```

---

## 6. VOCAL Mode

Purpose:

```text
Evaluate only vocal quality.
```

Output fields:

```text
score_mode = vocal
vocal_score
vocal_blockers
vocal_dominant_defect
vocal_decision
```

### Vocal score formula

```text
vocal_score =
  0.28 * vocal_naturalness
+ 0.22 * instrumental_bleed_absence
+ 0.16 * artifact_absence
+ 0.12 * vocal_body_preservation
+ 0.10 * vocal_tail_preservation
+ 0.07 * phase_stereo_stability
+ 0.05 * temporal_stability
```

Weights must sum to `1.0`.

### Vocal blockers

```text
severe_instrumental_bleed_in_vocal
severe_robotic_musical_noise
severe_vocal_body_loss
severe_vocal_tail_cut
severe_phase_comb_filtering
vocal_missing_on_active_region
```

### Vocal dominant defects

```text
instrumental_bleed_in_vocal
robotic_musical_noise
vocal_body_loss
vocal_tail_loss
phase_comb_filtering
vocal_missing_on_active_region
thin_or_overcleaned_vocal
```

### Vocal allowed/forbidden actions

If dominant defect is `vocal_body_loss`:

```text
allowed:
- rollback_to_fuller_vocal_candidate
- body_preserving_blend_plan

forbidden:
- stronger_cleanup
- stronger_debleed
- highpass_aggressive
- denoise_stack
```

If dominant defect is `vocal_tail_loss`:

```text
allowed:
- tail_preserving_candidate
- segment_tail_blend_plan

forbidden:
- hard_vad_gate
- aggressive_silence_gate
```

If dominant defect is `robotic_musical_noise`:

```text
allowed:
- less_aggressive_candidate
- rollback
- artifact_safe_candidate

forbidden:
- more_denoise
- cleanup_stack
```

If dominant defect is `instrumental_bleed_in_vocal`:

```text
allowed:
- cleaner_vocal_candidate
- segment_vocal_switch

forbidden:
- global_aggressive_cleanup
```

---

## 7. MINUS Mode

Purpose:

```text
Evaluate only instrumental/minus quality.
```

Output fields:

```text
score_mode = minus
minus_score
minus_blockers
minus_dominant_defect
minus_decision
```

### Minus score formula

```text
minus_score =
  0.30 * vocal_leakage_absence
+ 0.20 * instrumental_integrity
+ 0.15 * artifact_absence
+ 0.12 * transient_preservation
+ 0.10 * spectral_balance
+ 0.08 * phase_stereo_stability
+ 0.05 * temporal_stability
```

Weights must sum to `1.0`.

### Minus blockers

```text
severe_vocal_bleed_in_minus
severe_music_damage
severe_transient_smearing
severe_phase_collapse
severe_pumping_or_holes
severe_high_band_damage
```

### Minus dominant defects

```text
vocal_bleed_in_minus
music_damage
transient_smearing
phase_collapse
pumping_or_holes
high_band_damage
spectral_balance_damage
```

### Minus allowed/forbidden actions

If dominant defect is `vocal_bleed_in_minus`:

```text
allowed:
- choose_candidate_with_lower_vocal_leakage
- minus_specific_candidate_selection
- frequency_aware_minus_selection

forbidden:
- stronger_vocal_cleanup
- denoise_stack
- accept_candidate_only_because_vocal_improved
```

If dominant defect is `music_damage`:

```text
allowed:
- rollback_to_music_preserving_minus
- less_aggressive_minus_candidate

forbidden:
- over_suppress_mid_band
- more_debleed
```

If dominant defect is `transient_smearing`:

```text
allowed:
- transient_preserving_candidate
- rollback

forbidden:
- smoothing_stack
- heavy_overlap_blend_without_check
```

---

## 8. BOTH Mode

Purpose:

```text
Check pair compatibility only.
```

It must NOT choose best vocal/minus.
It must run only after separate vocal/minus selection.

Output fields:

```text
score_mode = both
pair_compatibility_score
pair_blockers
pair_decision
```

### Pair compatibility score

```text
pair_compatibility_score =
  0.40 * mixture_reconstruction
+ 0.20 * phase_alignment
+ 0.15 * duration_channel_sr_compatibility
+ 0.15 * energy_balance
+ 0.10 * no_cross_stem_conflict
```

Weights must sum to `1.0`.

### Pair blockers

```text
severe_mixture_reconstruction_drift
duration_mismatch
sample_rate_mismatch
channel_mismatch
phase_collapse
```

### Pair decisions

```text
pair_compatible
needs_pair_reconciliation
segment_blend_needed
blocked_pair_incompatible
```

Rules:

```text
If vocal and minus selected from different candidates, pair compatibility must pass.
If compatibility fails, do not publish.
If compatibility fails, output needs_pair_reconciliation or segment_blend_needed.
```

---

## 9. Required Integration Changes

### Windowed judge

Update:

```text
tools/run_no_reference_windowed_judge.py
```

Required:

- every judge row must include `score_mode`;
- support `--score-mode vocal|minus|both`;
- output mode-specific scores;
- do not use a shared final score for all modes;
- `both` mode only if pair-compatibility inputs are available.

Suggested outputs:

For vocal mode:

```text
vocal_score
vocal_blockers
vocal_dominant_defect
vocal_decision
```

For minus mode:

```text
minus_score
minus_blockers
minus_dominant_defect
minus_decision
```

For both mode:

```text
pair_compatibility_score
pair_blockers
pair_decision
```

### Optimizer

Update:

```text
tools/run_no_reference_optimizer.py
```

Required:

- run/consume vocal-mode scores separately;
- run/consume minus-mode scores separately;
- select best vocal independently;
- select best minus independently;
- run pair compatibility after selection;
- never average vocal/minus into one score;
- output:
  - `selected_vocal_candidate_id`
  - `selected_minus_candidate_id`
  - `same_candidate_pair`
  - `pair_compatibility_decision`
  - `needs_pair_reconciliation`
  - `segment_blend_needed`

### Segment planner

Update:

```text
tools/plan_no_reference_segment_blends.py
```

Required:

- use source-stem-specific matrix;
- vocal-window plans must use vocal mode;
- minus-window plans must use minus mode;
- both-window plans require both vocal and minus improvement plus compatibility.

### Preview judge

Update if needed:

```text
tools/judge_no_reference_segment_blend_preview.py
```

Required:

- compare preview in appropriate score mode;
- vocal preview uses vocal mode;
- minus preview uses minus mode;
- both preview checks both stem-specific improvement and pair compatibility.

---

## 10. Required Tests

Create/extend tests so they prove:

### Score matrix tests

```text
- weights sum to 1.0 for vocal mode.
- weights sum to 1.0 for minus mode.
- weights sum to 1.0 for both mode.
- invalid mode fails.
- vocal mode detects severe_instrumental_bleed_in_vocal.
- minus mode detects severe_vocal_bleed_in_minus.
- both mode detects sample_rate_mismatch.
```

### Separation of modes

```text
- candidate can pass vocal mode and fail minus mode.
- candidate can pass minus mode and fail vocal mode.
- vocal mode ignores minus-only improvement if vocal gets worse.
- minus mode ignores vocal-only improvement if minus gets worse.
- optimizer selects separate best vocal and best minus.
- optimizer does not average vocal and minus into one shared score.
- both mode blocks incompatible selected vocal/minus pair.
```

### Segment planner tests

```text
- vocal segment uses vocal matrix.
- minus segment uses minus matrix.
- both segment requires both scores to improve.
- minus regression does not block vocal-only segment plan.
- vocal regression does not block minus-only segment plan.
```

### Safety tests

```text
- no stems required.
- no reference required.
- no human CSV required.
- no training.
- no model registry edit.
- no cleanup/debleed/denoise added.
- no threshold relaxation.
- no release_ready.
```

---

## 11. Required Docker Validation

Run:

```bash
docker compose run --rm --no-deps ml-worker pytest   tests/test_no_reference_score_matrix.py   tests/test_run_no_reference_windowed_judge.py   tests/test_run_no_reference_optimizer.py   tests/test_plan_no_reference_segment_blends.py   tests/test_judge_no_reference_segment_blend_preview.py   -q
```

If a test file does not exist yet, create it only if needed for this task.

---

## 12. Required Real Run

After tests pass, rerun no-reference pipeline with mode-specific matrix.

### Step 1 — judge vocal mode

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py   --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl   --score-mode vocal   --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_vocal.jsonl   --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_vocal_summary.md   --cache-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_metric_cache.jsonl   --window-seconds 12   --max-windows-per-candidate 5
```

### Step 2 — judge minus mode

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py   --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl   --score-mode minus   --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_minus.jsonl   --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_minus_summary.md   --cache-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_metric_cache.jsonl   --window-seconds 12   --max-windows-per-candidate 5
```

### Step 3 — optimizer with separate vocal/minus scores

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_optimizer.py   --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_candidate_pool_300.jsonl   --vocal-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_vocal.jsonl   --minus-judge-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_windowed_judge_300_minus.jsonl   --judge-mode mode_specific   --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_mode_specific_result.jsonl   --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_mode_specific_summary.md
```

### Step 4 — pair compatibility

If optimizer selected different candidates for vocal and minus, run pair compatibility.

```bash
docker compose run --rm --no-deps ml-worker python tools/run_no_reference_windowed_judge.py   --candidates-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_optimizer_mode_specific_result.jsonl   --score-mode both   --output-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_pair_compatibility.jsonl   --summary-md /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_pair_compatibility_summary.md   --cache-jsonl /app/lab_artifacts/no_reference_optimizer_20260531/no_reference_metric_cache.jsonl   --window-seconds 12   --max-windows-per-candidate 5
```

If the existing CLI does not support this exact call shape, implement the smallest compatible interface and document the exact command used.

---

## 13. Final Report Format

Report in Russian.

Must include:

```text
1. Short execution plan

2. Confirmed vs unconfirmed

3. Created / updated / changed files

4. Docker tests result

5. Real run counts:
   - candidates judged vocal mode
   - candidates judged minus mode
   - vocal accepted count
   - minus accepted count
   - candidates passed vocal but failed minus
   - candidates passed minus but failed vocal
   - selected vocal candidates
   - selected minus candidates
   - same-candidate pairs
   - mixed-candidate pairs
   - pair-compatible count
   - pair-incompatible count
   - needs_pair_reconciliation
   - segment_blend_needed

6. Improvement metrics:
   - average vocal score
   - average minus score
   - average pair compatibility score
   - fitness before/after if applicable

7. Safety:
   - reference_required=false
   - human_csv_required=false
   - training_started=false
   - fine_tune_started=false
   - models_changed=false
   - model_registry_edited=false
   - cleanup_routes_added=false
   - thresholds_relaxed=false
   - audio_exported=false
   - audio_copied=false
   - release_ready=false

8. Next safe step
```

---

## 14. Definition Of Done

This task is complete only if:

```text
1. The score matrix is mode-specific.
2. Vocal and minus are not averaged into one shared score.
3. A candidate can pass vocal and fail minus.
4. A candidate can pass minus and fail vocal.
5. Optimizer can select vocal and minus independently.
6. Both mode checks compatibility only after selection.
7. Segment planner uses source_stem-specific matrix.
8. No stems/reference/human CSV/training/new models are required.
9. Docker tests pass.
10. Real artifacts are generated and summarized.
```

---

## 15. Critical Mindset

Do not optimize for pretty summaries.
Optimize for preventing the old bug:

```text
one approach used for vocal and minus at the same time.
```

The goal is:

```text
best vocal by vocal logic
best minus by minus logic
compatibility check after
```

This is the correct direction toward real no-reference improvement without stems.
