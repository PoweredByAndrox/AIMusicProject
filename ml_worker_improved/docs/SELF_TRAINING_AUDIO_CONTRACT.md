# Self-Training Audio Contract

Self-training in this project means active learning with verified outcomes. It
does not mean a model can label its own output as ground truth.

## Allowed Loop

```text
candidate generation
  -> proxy safety sentinels
  -> neural audio quality critic shadow scoring
  -> uncertainty/disagreement queue
  -> reference or human verification
  -> RAG-retained dataset refresh
  -> torch critic retraining
  -> locked holdout validation
  -> experiment-audit packet
  -> experiment-audit PASS
  -> current-RAG promotion candidate discovery
  -> promotion decision
```

## Forbidden

- generated stem -> model says good -> ground truth
- rejected/blocked candidate -> listen/result artifact
- weak proxy score -> promotion evidence
- pseudo-teacher row -> release claim
- threshold relaxation as a quality fix
- per-track hardcode or timestamp special case
- host runtime execution as production validation
- promotion without locked holdout and experiment-audit PASS

## RAG Role

RAG is the system memory and evidence layer. It records:

- source manifests and chunks
- accepted, rejected, and blocked candidates
- route families tried
- blocker reasons and issue tags
- label provenance
- split and holdout exclusions
- model manifests and validation reports

RAG does not decide whether audio is clean. The learned critic evaluates audio
or tensor windows; RAG prevents repeated mistakes and fake verification.

Small metadata artifacts created during the loop must be added back through a
metadata overlay pack, not by replacing `artifact_rag/current.json` with a
narrow diagnostic pack. The overlay must preserve base RAG coverage, keep
delivery/release/publish flags false, and make blocked evidence queryable for
future routing and audit.

`run_audio_quality_learning_cycle.py` may perform that retention step for
source-lock creation, verified-corpus discovery, learning handoff, and promotion
candidate discovery when given explicit metadata root and overlay output dirs.
The retained discovery evidence is still not a dataset, promotion decision, or
delivery result: missing, failed, or blocked verified corpus refs remain
evidence only, and the cycle must not fabricate verified rows or mark a
training handoff ready without a safe dataset manifest. Ready promotion
candidate rows still require the separate promotion decision gate. If the
learning-cycle output root is already under `artifact_rag`, the same overlay
must retain the cycle manifest root as metadata evidence for the next loop.

The learning cycle may write an `audio_quality_learning_handoff.v1` root with
Docker command plans for `train_audio_quality_critic.py`,
`validate_audio_quality_critic.py`, and
`build_audio_quality_experiment_audit_packet.py`. This handoff is a plan and
provenance artifact only: it must keep `training_started=false`,
`models_changed=false`, `promotion_approved=false`, and all delivery/release
flags false until those commands actually run and produce their own validated
manifests.

## Verified Row Intake

Verified/reference material enters the critic through
`build_audio_quality_verified_rows.py` after
`validate_verified_audio_training_corpus.py` has passed. The intake requires a
source-lock pack, verified row reports, RAG checkpoint refs, existing audio, and
short tensor artifacts. It does not create weak labels and does not turn
generated stems or pseudo-teacher outputs into ground truth. The source-lock
pack must match the current RAG id with an existing RAG index, and the verified
corpus manifest must have a chunk/sha-backed current-RAG reference before
promotion-capable rows are created.

When `run_audio_quality_learning_cycle.py` finds ready verified-corpus discovery
rows, it runs the same verified-row intake under short `vrows/vrNNN` artifact
paths and builds the dataset only from passed intake outputs plus explicitly
verified review outcomes. Discovery-only refs, blocked intake rows, weak labels,
and raw model predictions do not enter training. This bridge only prepares
handoff commands; it does not start training, promote a model, or enable runtime
acceptance.

If discovery is blocked, `build_audio_quality_verified_corpus_recovery_plan.py`
may record current-RAG recovery evidence such as full-pack source manifests and
raw-reference traces. Raw-reference audio from separator/model diagnostics is
not ground truth by itself, even when the bytes still exist. It must first be
verified through a ground-truth/reference bundle and a passed
`verified_audio_training_corpus_manifest.v1` before any AQC dataset or promotion
path can use it.

`build_audio_quality_verified_material_request_packet.py` is the allowed bridge
from recovery evidence to human/reference work. It creates request rows and
decision templates only. Request rows remain `safe_for_training=false` and
`verified_outcome=false`; the system cannot treat them as labels, cannot write a
ground-truth manifest from them, and cannot train until an explicit verified
decision or passed verified corpus manifest exists.

## Promotion Rule

A model remains shadow-only unless validation explicitly proves promotion
safety from locked holdout scores and a machine-readable experiment-audit PASS.
A promoted model must still write full provenance and cannot bypass delivery
gates.

An experiment-audit packet is only a reviewer handoff. It lists evidence paths
and the checklist for an independent reviewer; it must never be treated as a
PASS or as promotion evidence by itself.

An experiment-audit ingest may normalize reviewer output only when the report
has a PASS verdict, pass integrity status, auditor identity, reviewer trace,
required checks, and evidence. Weak reports or packet manifests are blocked.

Promotion validation accepts only the normalized audit JSON produced by the
experiment-audit ingest route. A raw PASS JSON without ingest provenance, source
packet manifest, source audit JSON, reviewer trace, every required check,
`status=pass`, and per-check evidence is not promotion evidence.

Promotion candidate discovery is RAG-first. It may pass a candidate to the
promotion decision builder only when the current RAG checkpoint contains
chunk/sha-backed evidence for the promotion validation JSON, model manifest,
locked holdout scores, and normalized audit JSON. Missing, stale, failed, or
delivery-contaminated refs become blocked evidence, not a fabricated PASS.

The promotion decision artifact combines model, validation, locked holdout, and
normalized audit evidence. It may record `promotion_allowed=true` as evidence,
but it must keep runtime authority, delivery, release, and publish flags false.
Runtime acceptance and listen/release delivery remain separate contracts.
