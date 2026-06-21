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
- A run is accepted only when every window has an accepted candidate. A clean
  silent/tail window cannot accept the full track if earlier windows are blocked.
- Accepted runs write a stitched listen pack under `listen/` with
  `delivery_status=accepted`, `delivery_allowed=true`, and
  `presentation_status=accepted_for_listen_pack`. Blocked runs do not publish a
  listen pack.

## Reconstruction Conservation

Candidates may declare:

```json
{
  "conservation_repair_stem": "other"
}
```

When present, the runner computes `mix - sum(vocals, drums, bass, guitar,
piano, other)` before rebuilding `minus`, and writes that residual into the
declared non-vocal stem. This is not threshold relaxation: the same
reconstruction gate still runs after repair. If repair is missing, unsafe,
shape-mismatched, or insufficient, the candidate remains blocked.

Allowed repair targets are non-vocal stems only. `vocals` and `minus` are not
valid conservation-repair targets.

The candidate manifest records `reconstruction_repair.repair_owner`, target
stem, and `pre_repair_residual_rms`; the post-repair residual remains the normal
`reconstruction.residual_rms` from the unchanged reconstruction gate.

## Minus Vocal Debleed Candidates

Late default candidates may declare:

```json
{
  "minus_vocal_debleed": {
    "strength": 1.0,
    "low_hz": 700.0,
    "high_hz": 1200.0,
    "target_stems": ["drums", "bass", "guitar", "piano", "other"]
  }
}
```

The runner applies this only after a six-stem candidate exists. It transfers the
declared vocal-band component from non-vocal stems into `vocals`, then rebuilds
`minus` from the updated non-vocal stems. The six-stem sum is preserved before
the unchanged reconstruction gate runs. This route does not relax contamination
thresholds: it must still pass the explicit `minus` / `vocals` check.

## Listen Pack

When the run is accepted, the runner concatenates the accepted candidate stems
for every window into:

```text
listen/
  pack.json
  playlist.m3u
  audio/mix.flac
  audio/vocals.flac
  audio/minus.flac
  audio/drums.flac
  audio/bass.flac
  audio/guitar.flac
  audio/piano.flac
  audio/other.flac
```

The pack is not a final release artifact; it is a listen-ready pipeline artifact
only after the run-level delivery fields are accepted.

## Contamination Checks

The full forbidden-source matrix is mandatory for delivery acceptance. Candidate
configuration cannot narrow that matrix. If a forbidden pair has no reference
and no supported no-reference detector, the candidate blocks as
`blocked_unknown`.

Candidates may declare check hints for specific matrix pairs:

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

When hints are present, the runner still scores the full matrix and applies the
hint, such as `model_manifest`, only to the matching pair. If a foreign
reference is provided through `foreign_references`, the scorer uses reference
projection. If no reference is provided, the scorer may use a supported
no-reference detector.

The default from-scratch Demucs candidates do not declare a shortened
`contamination_checks` list. They must satisfy the same full matrix delivery
contract as custom recipes. With the current no-reference detector this means a
default scratch search can block until the missing pair detectors or references
exist; it must not publish a listen pack from a single `minus` / `vocals`
sentinel.

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

- `base_stem`: any tested stem from the forbidden-source matrix
- `foreign_source`: a source forbidden for that tested stem
- `label_type=synthetic_clean`
- `is_ground_truth=true`
- `input_stem`: clean stem plus injected foreign-source leak
- `target_leak`: the injected leak signal

For each safe row, training derives a positive example from `input_stem` and a
negative example from `input_stem - target_leak`. Generated separator output,
dirty references, pseudo labels, and non-ground-truth rows cannot become
training ground truth.

The manifest records `trained_pairs`. A single manifest may cover multiple
forbidden pairs, and the runner can apply it to the full matrix with:

```json
{
  "contamination_model_manifest": "artifacts/scd/m001/manifest.json"
}
```

Pair-specific `contamination_checks[].model_manifest` entries are hints that
override the shared manifest only for that pair. If a requested pair is not in
`trained_pairs`, the scorer blocks as `blocked_unknown` with
`model_pair_untrained` evidence.

The trained manifest remains shadow evidence:

- `detector_mode=no_reference_neural_v1`
- `shadow_only=true`
- `promotion_allowed=false`
- `delivery_allowed=false`
- `release_ready=false`
- `can_publish_as_result=false`

At runtime, a candidate may opt into this model through
`contamination_model_manifest` or a pair-specific
`contamination_checks[].model_manifest`. The scorer then reports
`contamination_probability`, `probability_threshold`, model provenance, and the
same leak-audio reconstruction path for failed candidates. If the model is
missing, mismatched, or unsupported, the check blocks as `blocked_unknown`.

## Learning Cycle CLI

`tools/run_stem_contamination_learning_cycle.py` wires the first automatic loop:

```text
clean-pairs manifest
-> build synthetic contamination rows
-> train pair-conditioned no-reference detector
-> write enriched search recipe with contamination_model_manifest
-> run fully automatic stem search
```

The cycle writes `dataset/`, `model/`, `recipe.json`, `search/run.json`, and
`manifest.json` under the requested output directory. It is still fail-closed:
if the trained detector does not cover every forbidden pair needed by the full
matrix, the search blocks as `blocked_no_clean_candidate` and no `listen/` pack
is published.

Docker command shape:

```powershell
docker compose run --rm --no-deps -T ml-worker `
  python tools/run_stem_contamination_learning_cycle.py `
    --mix "/host_testing_stuff_music/input.mp3" `
    --clean-pairs-manifest "/host_artifacts/stem_eval/cp001/manifest.json" `
    --out "/host_artifacts/stem_eval/cycle001" `
    --max-candidates 10 `
    --require-gpu true `
    --no-artifact-rag
```
