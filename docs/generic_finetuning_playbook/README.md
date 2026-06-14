# Generic Fine-Tuning System Playbook

This document describes a reusable pattern for taking an existing model, adapting it to a project, validating it, and retraining it when new evidence appears. It is not tied to TradingView, Pine Script, or any one model family.

## Core Idea

The system is a feedback loop:

1. Pick a base model that is already close to the target task.
2. Convert project data into a clean supervised dataset.
3. Train a small adapter or fine-tune selected weights.
4. Evaluate against fixed metrics and real project examples.
5. Merge/export the model only if it improves.
6. Record the dataset fingerprint and training metadata.
7. When new data arrives, retrain only if the trainable corpus changed.

For text/code models this often means PEFT LoRA/QLoRA on a Hugging Face model, then optional merge/export for serving. For audio, vision, or diffusion models the same loop applies, but the dataset format, loss functions, metrics, and export target are different.

## What "Any Model" Really Means

This works when the model has:

- accessible trainable weights;
- a reproducible training script or framework;
- a dataset format that maps inputs to desired outputs;
- a validation metric that correlates with real quality;
- a serving/export path after training.

It does not mean every closed API model can be directly weight-trained. If weights are closed, use prompt tuning, retrieval, evaluation, routing, or a trainable open model around it.

## Architecture

```text
raw project data
  -> import/normalize
  -> trainable dataset
  -> corpus fingerprint
  -> train adapter/fine-tune
  -> evaluate
  -> merge/export
  -> serve
  -> collect failures/successes
  -> retrain-if-needed
```

## Components

### Data Importer

Responsible for ingesting project examples into a stable local corpus. It should store raw input, normalized input, target output, metadata, source, permission/license, and hashes.

Minimum metadata:

```json
{
  "id": "example-id",
  "source": "manual|public|generated|user_feedback",
  "permission_status": "owned|licensed|public|user_provided",
  "input_path": "path-or-uri",
  "target_path": "path-or-uri",
  "input_hash": "sha256",
  "target_hash": "sha256",
  "tags": ["domain", "failure-type"],
  "quality_score": 0.0,
  "notes": ""
}
```

### Dataset Builder

Converts the corpus into exactly what the trainer expects.

For LLM SFT, common formats are:

```json
{"text": "full training sequence"}
```

```json
{"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]}
```

```json
{"prompt": "input", "completion": "target output"}
```

For audio separation, there are two different trainable targets. Pick the target explicitly instead of assuming one.

If the trainable target is the audio separator, the dataset is paired audio:

```json
{
  "mix": "audio/mix.wav",
  "stems": {
    "vocals": "audio/vocals.wav",
    "drums": "audio/drums.wav",
    "bass": "audio/bass.wav",
    "other": "audio/other.wav"
  },
  "sample_rate": 44100,
  "tags": ["dense_mix", "female_vocal", "heavy_reverb"]
}
```

If the trainable target is an LLM/controller that improves the pipeline, the dataset is decision and correction data:

```json
{
  "input": {
    "track_metadata": {"genre": "rock", "duration": 214.2},
    "baseline_metrics": {"vocals_bleed": 0.31, "si_sdr_vocals": 6.8},
    "failure_report": "snare leaks into vocal stem during chorus"
  },
  "output": {
    "chosen_model": "separator-v2",
    "preprocess": ["normalize_loudness", "split_30s_windows"],
    "separation_args": {"overlap": 0.5, "segment_seconds": 30},
    "postprocess": ["debleed_vocals", "phase_check"],
    "reason": "high drum bleed requires vocal debleed pass and smaller windows"
  },
  "result": {
    "accepted": true,
    "human_score": 0.86,
    "notes": "less snare bleed, no audible vocal damage"
  }
}
```

In that setup, the LLM is not a passive prompt helper. It is the trainable policy/controller that learns which model, scripts, parameters, preprocessing, postprocessing, and repair actions work for this project.

### Trainer

The trainer should be parameterized, not hard-coded:

- `BASE_MODEL`
- `DATASET_DIR`
- `OUTPUT_ADAPTER_DIR`
- `MERGED_MODEL_DIR`
- `TRAIN_ARGS`
- `EVAL_ARGS`
- `EXPORT_ARGS`

For LLMs, PEFT LoRA is usually the first practical choice because it trains a small adapter instead of all weights. PEFT supports merging adapter weights into the base model with `merge_and_unload()` when a standalone inference model is needed.

### Evaluator

Never promote a model only because training loss improved. Use a stable eval set and task-specific metrics.

Generic gates:

- validation loss did not explode;
- domain metric improved or stayed acceptable;
- no major regression on fixed examples;
- output passes static/runtime validators;
- human spot-check passes on hard cases.

### Promotion

Promotion should be explicit:

```text
candidate adapter/model
  -> eval pass
  -> merge/export
  -> smoke test
  -> update runtime config
  -> write training state
```

The state file should be written only after the full pipeline succeeds. If training or export fails, the next run should retry.

## Retrain-If-Needed Logic

Use a logical fingerprint, not file size.

Example fingerprint payload:

```json
[
  {
    "id": 1,
    "input_hash": "sha256",
    "target_hash": "sha256",
    "permission_status": "owned",
    "tags": ["hard_case"]
  }
]
```

Hash the sorted JSON. If the hash equals the previous completed training state, skip retraining.

Reasons:

- `state_missing`: no previous successful training state;
- `fingerprint_changed`: new or changed trainable examples;
- `force`: user explicitly requested retraining;
- `unchanged`: skip.

## Error Feedback Loop

Every failure should become structured data, not just a chat message.

```json
{
  "run_id": "2026-06-13T10:00:00Z",
  "input_id": "example-123",
  "model_version": "candidate-7",
  "failure_type": "bleed|syntax|hallucination|latency|quality",
  "severity": "low|medium|high",
  "expected": "what should happen",
  "actual": "what happened",
  "fix_label": "manual correction or accepted target",
  "include_in_training": true
}
```

The next dataset build can oversample high-severity hard cases.

## Minimal Reusable File Layout

```text
project/
  data/
    corpus.sqlite
    train_dataset/
    eval_dataset/
  models/
    base/
    adapters/
    merged/
    exported/
  output/
    eval_reports/
    failures/
  tools/
    prepare_dataset.py
    train_model.py
    merge_model.py
    export_model.py
    eval_model.py
    retrain_decider.py
  scripts/
    train_docker.ps1
    export_docker.ps1
    retrain_if_needed.ps1
    import_and_retrain.ps1
  docs/
    TRAINING.md
    EVALUATION.md
    DATASET_SCHEMA.md
```

## Practical Rule

Do not start with a giant fine-tune. Start with:

1. baseline model;
2. 20 to 100 hard examples;
3. deterministic eval script;
4. small adapter training;
5. compare outputs;
6. scale dataset only after the loop works.
