# Fully Automatic Stem Search Contract

`tools/run_fully_automatic_stem_search.py` is a fail-closed scratch search
runner for six-stem separation candidates.

## Hard Rules

- The six generated stems are mandatory: `vocals`, `drums`, `bass`, `guitar`,
  `piano`, and `other`.
- `minus` is always rebuilt from non-vocal stems.
- Candidates are untrusted by default.
- `human_review` is not a verdict or next action.
- Unknown evidence blocks automatically.
- Failed or rejected candidates remain scratch/debug evidence only.
- The runner must not write into `artifact_rag`.

## Contamination Checks

Candidates may declare explicit checks:

```json
{
  "contamination_checks": [
    {
      "tested_stem": "minus",
      "foreign_source": "vocals",
      "model_manifest": "artifacts/scd/m001/manifest.json"
    }
  ]
}
```

When explicit checks are present, the runner scores exactly those checks. If a
foreign reference is provided through `foreign_references`, the scorer uses
reference projection. If no reference is provided, the scorer may use a
supported no-reference detector.

When no explicit checks are present, the runner keeps the full forbidden-source
matrix and blocks missing references as `missing_contamination_reference:*`.
This prevents a candidate with no contamination proof from becoming accepted.

## No-Reference Detector V1

`tools/score_stem_contamination.py` currently supports one deterministic
no-reference path:

- tested stem: `minus`
- forbidden source: `vocals`
- detector mode: `no_reference_spectral_v1`
- evidence: band-limited leak reconstruction in the 700-1200 Hz band

If detected leak energy is above `detection_floor_db`, the check returns
`fail`, writes `leaks/minus_vocals.flac`, and the runner blocks the candidate as
`blocked_leak`.

Unsupported no-reference pairs return `blocked_unknown` with
`reason=no_reference_detector_not_available`.

This detector is not the learned production audio critic. It is a bounded
automatic sentinel used to make the scratch loop observable and testable before
neural contamination models are trained and promoted.

## Trainable Detector V1

`tools/train_stem_contamination_detector.py` trains a torch no-reference model
from synthetic contamination rows produced by
`tools/build_stem_contamination_dataset.py` or equivalent rows with:

- `base_stem=minus`
- `foreign_source=vocals`
- `label_type=synthetic_clean`
- `is_ground_truth=true`
- `input_stem`: clean minus plus injected vocal leak
- `target_leak`: the injected leak signal

For each safe row, training derives a positive example from `input_stem` and a
negative example from `input_stem - target_leak`. Generated separator output,
dirty references, pseudo labels, and non-ground-truth rows cannot become
training ground truth.

The trained manifest remains shadow evidence:

- `detector_mode=no_reference_neural_v1`
- `shadow_only=true`
- `promotion_allowed=false`
- `delivery_allowed=false`
- `release_ready=false`
- `can_publish_as_result=false`

At runtime, a candidate may opt into this model by setting
`contamination_checks[].model_manifest`. The scorer then reports
`contamination_probability`, `probability_threshold`, model provenance, and the
same leak-audio reconstruction path for failed candidates. If the model is
missing, mismatched, or unsupported, the check blocks as `blocked_unknown`.
