# Fully Automatic Stem Search Contract

`tools/run_fully_automatic_stem_search.py` is a fail-closed scratch search
runner for six-stem separation candidates.

## Hard Rules

- The six generated stems are mandatory: `vocals`, `drums`, `bass`, `guitar`,
  `piano`, and `other`.
- `minus` must come from a dedicated minus/instrumental lane. It cannot be
  accepted or rebuilt as the sum of six-stem non-vocal outputs in any
  production profile.
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
- Blocked runs must write `training/root_cause/manifest.json` and
  `training/root_cause/queue.jsonl`. Repeated blockers become training/mining
  samples instead of another pile of rejected candidates.

## Reconstruction Conservation

Candidates may declare:

```json
{
  "conservation_repair_stem": "other"
}
```

When present, the runner computes `mix - sum(vocals, drums, bass, guitar,
piano, other)` and writes that residual into the declared non-vocal stem. This
is not threshold relaxation and it never rebuilds production `minus`: the same
reconstruction gate still runs after repair, and a dedicated minus/instrumental
lane remains required. If repair is missing, unsafe, shape-mismatched, or
insufficient, the candidate remains blocked.

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
declared vocal-band component from non-vocal stems into `vocals`, but it does
not rebuild `minus` from those stems. A dedicated minus/instrumental lane is
still required, and the unchanged reconstruction gate runs afterward. This
route does not relax contamination thresholds: it must still pass the explicit
`minus` / `vocals` check.

Late default candidates may also declare:

```json
{
  "vocal_projection_debleed": {
    "strength": 1.0,
    "target_stems": ["drums", "bass", "guitar", "piano", "other"]
  }
}
```

This projects each target non-vocal stem onto the current `vocals` waveform and
moves the correlated component back into `vocals`. It is meant for leaked vocal
waveform in individual non-vocal stems, not for building a production minus or
acceptance by metric bypass. The same reconstruction and full forbidden-source
matrix gates still run after the transfer.

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

Default no-reference detector discovery checks the Docker artifact mount
`/project_artifacts/stem_eval` before legacy host-style roots, so full-track
Docker runs can reuse the latest `cycle*/model/manifest.json` without a manual
environment override. If an older detector manifest stores a stale absolute
`model_file`, the scorer resolves `model.pt` next to the manifest before
blocking as unavailable.

The default from-scratch candidates are target-specific, not a Demucs-only
shortcut. Each default candidate starts with a Demucs `htdemucs_6s` six-stem
route, then replaces `vocals` with a specialized vocal/instrumental model route
when that route is available in the Docker image. If no dedicated
minus/instrumental lane writes `minus` or mapped `instrumental`, the candidate
blocks instead of rebuilding `minus` from non-vocal stems.

## Model-Split V2 Profile

`--profile model_split_v2` builds a separate lane search instead of using one
all-purpose candidate as the source of every delivered stem:

- vocal lane: `bs_mamba2_vocals`, `audio_separator_bs_roformer_vocals`,
  `mdx23c_vocals`, `melband_roformer_vocals`;
- minus lane: `bs_mamba2_vocals:instrumental`,
  `becruily_instrumental_vocals`,
  `audio_separator_bs_roformer_vocals:instrumental`,
  `mdx23c_vocals:instrumental`, `htdemucs_ft:instrumental`;
- stem lane: `htdemucs_6s`, `hybrid_htdemucs_ft_bs_roformer_6s`,
  `hybrid_htdemucs_ft_bs_roformer_6s_aggressive_melody_first`;
- bass-support lane: `scnet_xl_ihf_musdb18:bass`;
- non-vocal support lane: `scnet_xl_ihf_musdb18:drums_bass_other`;
- SCNet remains auxiliary non-vocal evidence, not the delivered full six-stem
  owner in this profile. It can replace bounded individual support stems
  (`bass`, or `drums`/`bass`/`other` from the support route) inside composite
  candidates, and the same reconstruction, damage, voice-presence, and
  contamination gates still decide delivery.

The listen pack is assembled from lane-specific outputs. `vocals` must come
from the vocal lane, `minus` from the minus lane, and `drums`, `bass`,
`guitar`, `piano`, and `other` from the stem lane. If no atomic lane combination
passes, V2 builds bounded composite candidates from already generated
per-lane route outputs, then runs the same reconstruction, damage,
voice-presence, and contamination gates on the composite. If a composite passes
`vocals + minus` reconstruction but the stem lane damages or contaminates
non-vocal stems, V2 may try bounded support stems from auxiliary non-vocal
routes; this does not make SCNet the owner of the delivered stem set and does
not bypass any gate. V2 also enumerates explicit composite repair candidates:
`none`, `vp100`, `bs250`, `bs400`, `pg08`, `vp100_bs400`, `bs400_pg08`,
`vp100_pg08`, and `vp100_bs400_pg08`. The `vp*` variants apply bounded
`vocal_projection_debleed` to non-vocal stems. The `bs*` variants apply a
bounded bass-band mid-side width repair using only the candidate bass stem, not
the mix. The `pg08` variants apply a bounded delivery-pair gain correction to
reduce `vocals + minus` residual without changing thresholds. Combined variants
apply their named repairs in order. All variants record `model_split_repair_id`
plus repair provenance and still must pass reconstruction, damage,
voice-presence, and full contamination gates.
Legacy broad-band `post_minus_vocal_debleed` and stem-merge routes remain
blocked from V2 delivery. Route manifests record lane, repair, and output-map
provenance, and accepted listen packs include `lane_provenance` plus per-window
lane provenance.

The minus lane may emit `instrumental.flac`; candidates can map that output to
delivery `minus.flac` with:

```json
{
  "role": "minus",
  "lane": "minus_lane",
  "output_map": {"minus": "instrumental"}
}
```

Registered vocal/instrumental models such as `bs_mamba2_vocals` are executed
through `tools/run_registered_model_candidate.py`. Their vocal route asks for
target stem `vocals`; their minus route asks for target stem `minus`, which
resolves existing manifest `instrumental` output through the registered stem
alias contract instead of rebuilding minus from six-stem outputs.
`bs_mamba2_vocals` remains fail-closed until the bounded Docker probe succeeds
and writes `/app/vendor/bs_mamba2_vocals/runtime_probe_passed.json`:

```bash
python tools/probe_bs_mamba2_runtime.py --duration-sec 2 --num-overlap 1
```

The marker must not be created manually. A probe that loads the checkpoint but
crashes before manifest validation leaves the route unavailable.

Route ids remain in manifests for provenance. Runtime route directories use a
sanitized `route_alias`, so ids such as `audio_separator_bs_roformer_vocals:instrumental`
do not create Windows-unsafe path segments.

Within one window, repeated target-specific routes are cached by role, lane,
route id, rendered command, and output map. Reused route manifests set
`reused_route=true` and record the source candidate/route paths; this reduces
duplicate GPU inference but does not reuse accepted verdicts or skip any
reconstruction, damage, voice-presence, or contamination gate.

Unlike the default profile, `model_split_v2` does not use the mixed final stem
set (`vocals` from vocal lane plus non-vocal stems from stem lane) as the
delivery reconstruction surface. That would combine outputs from independent
models and can falsely block or falsely repair a candidate. V2 gates delivery
pair consistency with `vocals + minus` and records the stem-lane six-stem sum as
`reconstruction.six_stem_diagnostic`. A failed diagnostic remains evidence, but
it is not allowed to rebuild or publish `minus`.

`model_split_v2` adds fail-closed damage gates before delivery acceptance:

- broad post-band debleed and stem-merge repair routes are non-publishable as
  accepted success routes;
- stereo width collapse blocks the candidate when the mix has measurable stereo
  width but the delivered stem collapses it;
- bass stereo width is compared against the mix bass-band reference, not the
  full-band mix width, so mono/narrow low end is not falsely treated as damage
  just because guitars, cymbals, or vocals make the full mix wide;
- each non-vocal stem, including `other`, must carry non-empty detail above the
  configured floor;
- a full-band voice-presence projection gate blocks vocal-like evidence in
  `minus` or non-vocal stems.

These gates do not replace the forbidden-source contamination matrix. They run
in addition to the V2 `vocals + minus` delivery-pair reconstruction check and
contamination checks. If they disagree with a candidate that would otherwise
pass contamination, the candidate blocks; there is no `human_review` state and
no threshold relaxation. In V2, setting `damage_gates.enabled=false` or
`voice_presence_gate.enabled=false` is itself a blocker, not a bypass.

The strongest late candidates use an instrumental-first route:

```text
mix -> vocal/instrumental separator -> instrumental -> Demucs htdemucs_6s
```

Those candidates keep the specialized `vocals` output, but split the separator's
`instrumental` output into `drums`, `bass`, `guitar`, `piano`, and `other`.
This avoids using Demucs-on-mix non-vocal stems when the target issue is vocal
bleed inside every non-vocal stem. Reconstruction and contamination gates are
unchanged.

The default candidate builder blocks as `missing_target_specific_vocal_route`
when Demucs is present but no specialized vocal route is available. Available
default vocal routes are tried in a bounded round-robin order:

- `audio_separator_bs_roformer_vocals`
- `mdx23c_vocals`
- `melband_roformer_vocals`
- `becruily_instrumental_vocals`

Default candidates do not declare a shortened `contamination_checks` list. They
must satisfy the same full matrix delivery contract as custom recipes. With the
current no-reference coverage this means a default scratch search can block
until the missing pair detectors or references exist; it must not publish a
listen pack from a single `minus` / `vocals` sentinel.

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

`tools/build_procedural_stem_clean_pairs.py` can create a safe synthetic
clean-pairs manifest for the whole forbidden-source matrix:

```powershell
docker compose run --rm --no-deps -T ml-worker `
  python tools/build_procedural_stem_clean_pairs.py `
    --out "/host_artifacts/stem_eval/cp001" `
    --duration-sec 1.0 `
    --examples-per-pair 2
```

The builder writes procedural `vocals`, `drums`, `bass`, `guitar`, `piano`,
`other`, and derived `minus` audio under short paths, then emits
`manifest.json` with every forbidden pair. It does not use generated separator
outputs, dirty references, pseudo labels, or human review. These rows are
allowed as `synthetic_clean` ground truth for detector bootstrap only; the
normal full-matrix gates still decide runtime delivery.

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

## Target-Specific Candidate Composition

Candidates may use separate routes for different targets before the normal
reconstruction and contamination gates run:

```json
{
  "target_specific_commands": [
    {
      "role": "six_stems",
      "command_template": ["python", "tools/run_demucs_six_stem_candidate.py", "..."]
    },
    {
      "role": "vocals",
      "command_template": ["python", "tools/run_vocal_candidate.py", "..."]
    }
  ]
}
```

Supported roles:

- `six_stems`: must write `vocals`, `drums`, `bass`, `guitar`, `piano`,
  `other`; these initialize the full candidate set.
- `vocals`: must write `vocals`; this can replace the six-stem vocal output.
- `minus`: must write `minus`, or write a declared source such as
  `instrumental` through `output_map`; this is required for `model_split_v2`
  delivery minus.
- `non_vocal_stems`: must write `drums`, `bass`, `guitar`, `piano`, `other`;
  this can replace the instrumental stem set for stem delivery only.

If a declared route misses one of its required outputs, the candidate blocks as
`blocked_missing_stem`. Missing minus-lane output always blocks instead of
falling back to rebuilt minus. The full forbidden-source matrix still decides
delivery after profile-specific gates pass.

Route outputs are normalized to the current window's sample rate, frame count,
and channel count before they are copied into the final candidate stem set. This
is a shape contract only; it does not relax reconstruction or contamination
gates. Any remaining mismatch or conservation error still blocks the candidate.

If listen-pack construction fails after any partial file has been written, the
runner removes the entire `listen/` directory before writing the blocked
`run.json`. A blocked run must not leave a partial listen surface behind.
