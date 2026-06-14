# Music Stem Separation Fine-Tuning Playbook

This is a concrete adaptation of the generic loop for a music project where the goal is to split a track into stems such as vocals, drums, bass, and other.

## Trainable Targets

Do not assume there is only one model to train. In a stem-separation product, the trainable target can be:

1. The audio separator itself.
2. An LLM/controller that learns how to run, correct, and improve the separation pipeline.
3. Both, in two layers.

If the current project already uses an audio model through scripts and the weak point is orchestration, parameter choice, postprocessing, failure recovery, and learning from accepted/rejected outputs, then the LLM/controller is a valid primary fine-tuning target.

The controller learns examples like:

```text
track metadata + baseline output + failure report
  -> selected model/config/scripts/postprocessing/correction plan
  -> accepted or rejected result
```

The audio separator learns examples like:

```text
mixture audio -> target stems
```

These are different datasets and different losses. A serious system can use both.

## Target System

```text
source tracks
  -> normalize sample rate/loudness
  -> align mix and stems
  -> split train/validation/test
  -> train/fine-tune separator
  -> evaluate stems
  -> collect failures
  -> add corrected examples
  -> retrain-if-needed
```

## Dataset Schema

### Audio Separator Dataset

```json
{
  "id": "track-001",
  "mix_path": "data/audio/mixes/track-001.wav",
  "stems": {
    "vocals": "data/audio/stems/track-001/vocals.wav",
    "drums": "data/audio/stems/track-001/drums.wav",
    "bass": "data/audio/stems/track-001/bass.wav",
    "other": "data/audio/stems/track-001/other.wav"
  },
  "sample_rate": 44100,
  "duration_seconds": 183.2,
  "license": "owned|licensed|user_provided",
  "genre": "pop|rock|edm|rap|other",
  "tags": ["dense_mix", "heavy_reverb", "female_vocal"],
  "quality_score": 0.92,
  "mix_hash": "sha256",
  "stem_hashes": {
    "vocals": "sha256",
    "drums": "sha256",
    "bass": "sha256",
    "other": "sha256"
  }
}
```

### LLM Controller Dataset

Use this when the LLM is the trainable part that controls the pipeline.

```json
{
  "id": "decision-001",
  "track_id": "track-001",
  "input": {
    "track_metadata": {
      "genre": "rap",
      "duration_seconds": 183.2,
      "sample_rate": 44100,
      "tags": ["dense_mix", "heavy_808", "autotuned_vocal"]
    },
    "baseline_model": "separator-v1",
    "baseline_args": {
      "segment_seconds": 60,
      "overlap": 0.25
    },
    "baseline_metrics": {
      "vocals_bleed": 0.34,
      "bass_bleed": 0.27,
      "si_sdr_vocals": 5.8
    },
    "failure_report": {
      "failure_type": "bass_kick_confusion",
      "severity": "high",
      "human_note": "808 tail is split between bass and drums"
    }
  },
  "target": {
    "action": "rerun_with_adjusted_pipeline",
    "chosen_model": "separator-v2",
    "preprocess": ["loudness_normalize", "highpass_sidechain_detector"],
    "separation_args": {
      "segment_seconds": 30,
      "overlap": 0.5,
      "stems": ["vocals", "drums", "bass", "other"]
    },
    "postprocess": ["bass_drum_boundary_refine", "phase_alignment_check"],
    "acceptance_tests": ["bass_bleed_below_threshold", "human_ab_pass"],
    "explanation": "The failure is bass/kick overlap, so use smaller windows and a bass/drum refinement pass."
  },
  "result": {
    "accepted": true,
    "human_score": 0.88,
    "metric_delta": {
      "bass_bleed": -0.14,
      "si_sdr_bass": 1.7
    }
  }
}
```

For LLM fine-tuning, this can be rendered as either `messages` or `prompt/completion` JSONL.

Example conversational training row:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a music stem-separation controller. Return only valid JSON pipeline decisions."
    },
    {
      "role": "user",
      "content": "Track: rap, heavy 808. Baseline: bass_kick_confusion high severity. Metrics: bass_bleed=0.27, vocals_bleed=0.34. Choose the next pipeline action."
    },
    {
      "role": "assistant",
      "content": "{\"action\":\"rerun_with_adjusted_pipeline\",\"chosen_model\":\"separator-v2\",\"separation_args\":{\"segment_seconds\":30,\"overlap\":0.5},\"postprocess\":[\"bass_drum_boundary_refine\",\"phase_alignment_check\"]}"
    }
  ]
}
```

## Failure Taxonomy

Use this taxonomy every time the separator fails:

```json
{
  "bleed_vocals_in_instrumental": "vocal remains in non-vocal stems",
  "bleed_drums_in_vocals": "drum transients leak into vocal stem",
  "bass_kick_confusion": "bass and kick are mixed into wrong stems",
  "reverb_tail_loss": "natural reverb or ambience is cut off",
  "phase_artifacts": "hollow, comb-filtered, or unstable stereo image",
  "transient_smearing": "attacks become soft or blurry",
  "high_frequency_artifacts": "metallic or watery artifacts",
  "quiet_vocal_loss": "soft vocal phrases disappear",
  "backing_vocal_confusion": "backing vocals are assigned inconsistently",
  "genre_failure": "specific genre performs worse than baseline"
}
```

## Metrics

Do not rely on one number.

Recommended automatic metrics:

- SDR or SI-SDR per stem;
- stem leakage / bleed estimate;
- spectral convergence;
- multi-resolution STFT loss;
- loudness difference;
- phase/stereo correlation checks;
- inference time and memory.

Recommended human checks:

- A/B baseline vs candidate;
- solo each stem;
- invert/recombine test;
- check quiet passages;
- check dense chorus;
- check transitions and reverb tails;
- check kick/bass separation.

## Training Strategy

Choose the training strategy from the trainable target.

### If Training The LLM Controller

Train on examples of decisions that improved the final stems, plus rejected decisions with corrected targets.

Good first experiment:

```text
dataset:
  100-500 decision examples
  include baseline metrics, failure labels, chosen actions, accepted/rejected result
  oversample severe recurring failures

training:
  LoRA/QLoRA SFT on a code/instruct model if it must emit JSON/scripts
  completion-only loss if using prompt/completion data
  strict JSON validation after generation

promotion:
  candidate must choose better actions than baseline rules on hard cases
  candidate output must pass schema validation
  candidate must not select unsafe/destructive scripts
```

### If Training The Audio Separator

Start from a strong open separator model that is already close to the task. Fine-tune on your domain audio data.

Good first experiment:

```text
dataset:
  20-50 high-quality paired tracks
  balanced genres
  include known hard failures

training:
  short run
  small learning rate
  freeze most layers if supported
  validate every fixed number of steps

promotion:
  require no regression on baseline eval set
  require improvement on hard-case set
```

## Hard-Case Dataset

Create a separate validation set that never goes into training:

```text
eval/hard_cases/
  vocals_with_reverb/
  dense_chorus/
  bass_kick_overlap/
  live_recordings/
  distorted_guitars/
  backing_vocals/
```

Every time the model fails badly, add the case to either:

- training set, if you have correct target stems;
- hard eval set, if it is only useful for regression detection;
- excluded set, if the target is not trustworthy.

## Feedback Record

```json
{
  "track_id": "track-001",
  "model_version": "separator-v3",
  "stem": "vocals",
  "failure_type": "bleed_drums_in_vocals",
  "timestamp_start": 63.4,
  "timestamp_end": 71.8,
  "severity": "high",
  "human_note": "snare leaks loudly into vocal stem during chorus",
  "include_in_training": true,
  "recommended_action": "add_corrected_stems"
}
```

## Retrain-If-Needed

Fingerprint only trainable examples. The fingerprint fields depend on what is being trained.

For audio-separator training:

```text
track_id + mix_hash + stem_hashes + license + tags + quality_score
```

For LLM-controller training:

```text
decision_id + input_hash + target_hash + result.accepted + failure_type + chosen_action + quality_score
```

Retrain when:

- a new owned/licensed track is imported;
- corrected stems replace old stems;
- failure labels mark a hard case for training;
- new accepted/rejected controller decisions are added;
- training config changes intentionally;
- user passes `--force`.

Skip when:

- only reports/logs changed;
- unlicensed examples were added;
- examples are duplicates;
- imported files failed validation.

## Model Promotion Gates

For an LLM/controller candidate, promote only if:

- output JSON/schema validation passes;
- selected actions improve hard-case outcomes more often than baseline routing;
- no unsafe scripts or invalid parameters are emitted;
- replay tests on historical failures pass;
- rollback to previous controller is available.

For an audio separator candidate, promote only if:

- automatic metrics improve on hard cases or stay acceptable globally;
- no major stem regresses by human listening;
- recombined stems do not introduce obvious phase/loudness damage;
- inference speed is acceptable;
- previous model can be restored quickly.

## Example Commands For A Future Project

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_audio_dataset.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_separator.ps1 -Build
powershell -ExecutionPolicy Bypass -File scripts\evaluate_separator.ps1
powershell -ExecutionPolicy Bypass -File scripts\promote_separator.ps1
```

Idempotent retraining:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\retrain_separator_if_needed.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\retrain_separator_if_needed.ps1
powershell -ExecutionPolicy Bypass -File scripts\retrain_separator_if_needed.ps1 -Force
```

Import plus retrain-if-needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import_audio_and_retrain.ps1 -Path D:\music_dataset\new_tracks
```

## Agent Prompt For Stem Separation

```text
You are an audio ML engineer specializing in music source separation.

Goal:
Improve stem separation for this project by turning failures and corrected stems into a reproducible fine-tuning loop.

Current separator:
{{CURRENT_SEPARATOR}}

Current LLM/controller:
{{CURRENT_CONTROLLER}}

Available paired data:
{{PAIRED_DATA_SUMMARY}}

Available decision/error data:
{{DECISION_DATA_SUMMARY}}

Known failures:
{{FAILURE_REPORTS}}

Constraints:
{{GPU_MEMORY}}
{{SAMPLE_RATE}}
{{STEM_TARGETS}}
{{LATENCY_REQUIREMENTS}}

Tasks:
1. Decide whether to train the LLM/controller, the audio separator, or both.
2. Propose dataset schema for the selected trainable target.
3. Propose validation checks for corrupted audio and invalid controller actions.
4. Propose training strategy for a small first run.
5. Propose metrics and hard-case eval set.
6. Propose retrain-if-needed fingerprint.
7. Propose promotion/rollback gates.

Return exact implementation plan, commands, and risk list.
```

## What To Avoid

- Training on bad stems and expecting quality to improve.
- Mixing licensed/unknown data into trainable corpus without permission.
- Evaluating only on the training tracks.
- Promoting a model based only on loss.
- Forgetting that better vocal SDR can still make drums or bass worse.
- Assuming the LLM cannot be the trainable part when the project quality depends on pipeline decisions, scripts, parameters, and feedback learning.
