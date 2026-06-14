# No-Reference Result Loop

This control-plane loop judges generated vocal/minus candidates without
reference stems. It is meant for normal tracks where no supervised vocal/minus
targets exist.

## Guarantees

- No reference vocal/minus files are required.
- No human CSV is required for the production path.
- No fine-tune or training is started.
- No model, checkpoint, registry, cleanup, debleed, denoise, or threshold route
  is added.
- No diagnostic audio is exported as a result.
- If every candidate is unsafe, the selector returns `blocked_hard_case`
  instead of accepting the best bad candidate.
- Segment blending is emitted only as a plan. The tool does not execute audio
  blending.

## Judge Input

Use `tools/run_no_reference_result_judge.py` with either a candidate JSONL or
one or more candidate manifests. A candidate JSONL row must contain:

```json
{
  "track_id": "track-id",
  "candidate_id": "candidate-id",
  "mix_path": "/app/path/to/original_mix.wav",
  "vocal_path": "/app/path/to/generated_vocal.flac",
  "instrumental_path": "/app/path/to/generated_minus.flac"
}
```

Run:

```bash
python tools/run_no_reference_result_judge.py \
  --candidate-jsonl /app/path/to/candidates.jsonl \
  --output-dir /app/lab_artifacts/no_reference_result_loop \
  --score-mode minus
```

`--score-mode` is optional for legacy generic judging. When supplied as `vocal`, `minus`, or `both`,
the judge preserves the generic no-reference metrics and appends the corresponding score-matrix
fields such as `minus_score`, `minus_decision`, and `minus_blockers`.

The judge writes:

- `no_reference_result_judge.jsonl`
- `no_reference_result_summary.md`

Each row contains no-reference proxy scores for mixture consistency, leakage,
activity, hallucination, artifact, thin vocal, transient damage, phase
consistency, an overall score, confidence, decision, reasons, and forbidden
actions.

`silence_hallucination_score` uses confidence-aware evidence from quiet mix
regions: a single quiet analysis subwindow cannot saturate the hallucination
ratio by itself, while sustained vocal activity over quiet regions still blocks
the candidate.

The analysis-window RMS floor is applied after the square root (`sqrt(mean(square)) + EPS`). This
keeps numerically silent vocal windows from being counted as active vocal content in quiet source
regions, while sustained non-zero vocal energy over quiet source regions remains a blocker.

## Candidate Selection

Run:

```bash
python tools/run_no_reference_candidate_selector.py \
  --judge-jsonl /app/lab_artifacts/no_reference_result_loop/no_reference_result_judge.jsonl \
  --output-dir /app/lab_artifacts/no_reference_result_loop \
  --score-mode both
```

The selector writes:

- `no_reference_candidate_selection.jsonl`
- `no_reference_result_summary.md`

Selector decisions:

- `accept_candidate`: one candidate is acceptable and no safer alternative is
  needed.
- `select_alternative_candidate`: one acceptable candidate is clearly better
  than the other candidates.
- `needs_segment_blend`: candidates appear complementary by segment, but no
  blend is executed.
- `blocked_hard_case`: no candidate can be safely accepted.

`--score-mode vocal|minus|both` selects the matrix used by the selector:

- `vocal` ranks by `vocal_score`, uses `vocal_decision`, and respects
  `vocal_blockers`.
- `minus` ranks by `minus_score`, uses `minus_decision`, and respects
  `minus_blockers`.
- `both` ranks by `pair_compatibility_score` when present, otherwise legacy
  `overall_no_reference_score`.

Mode-specific delivery routes must pass the matching score mode to both the result judge and selector
instead of letting an overall/shared score hide a vocal-only or minus-only failure.

## Safety Contract

This loop is diagnostic/control-plane only until a row is explicitly accepted by
the selector. It does not copy, export, train on, or repair audio. Accepted
selection rows identify an existing generated candidate; they do not create a
new result file.
