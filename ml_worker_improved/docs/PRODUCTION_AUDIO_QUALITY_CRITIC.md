# Production Audio Quality Critic

`audio_quality_critic_v1` is the first production contract for a learned audio
quality critic. It is not a replacement for RAG and it is not a listen-pack
artifact.

## Role Split

- The neural critic evaluates bounded audio/tensor windows for vocal/minus
  quality issues.
- RAG stores provenance, route history, labels, failures, blocker reasons, and
  training evidence.
- Existing no-reference DSP/proxy gates remain safety sentinels.
- Qwen/controller routes may use critic/RAG evidence to propose the next route,
  but cannot override hard safety gates or mark delivery accepted.

## Tools

- `tools/build_audio_quality_source_lock.py`
  writes `pack.json`, `summary.md`, and `progress.jsonl` for the current
  source-of-truth checkpoint.
- `tools/build_artifact_rag_metadata_overlay.py`
  builds a safe current-RAG overlay for small `artifact_rag/*` metadata evidence
  roots such as source locks, discovery manifests, blocked queues, and summaries.
  It copies the current pack tables first, adds queryable metadata roots, and
  updates `artifact_rag/current.json` only after base root/file/manifest/chunk
  counts are preserved or increased. It is not a delivery, release, or
  promotion artifact. Use `--base-rag-id` when rebuilding from a known clean
  base pack instead of stacking overlays over the current pointer.
- `tools/build_audio_quality_rows_from_rag.py`
  converts current RAG no-reference gate evidence into weak/rejected bridge
  rows with RAG provenance and short tensor aliases. It does not create
  verified or promotion-eligible labels. Positive weak labels require proxy
  safety gates to pass; failed pair/no-reference gates become
  `rejected_blocked` evidence. Pass `--source-lock-pack-json` so downstream
  dataset/model/score manifests stay tied to the source-of-truth checkpoint.
- `tools/build_audio_quality_verified_rows.py`
  converts a passed `verified_audio_training_corpus_manifest.v1` into
  tensor-backed AQC rows. It accepts only safe row reports from
  `generic_finetuning_audio_training_example.v1`, requires a source-lock pack,
  verifies that the source-lock pack matches the current RAG id and has an
  existing RAG index, writes short `qNNN.npz` tensor aliases, preserves
  RAG/source manifest refs,
  normalizes `holdout` to `locked_holdout`, and emits only
  `reference_verified` or `human_verified` labels. Failed corpus validation
  produces blocked evidence only. Generated or pseudo-teacher ground-truth
  claims hard fail. Promotion-capable verified rows require a chunk/sha-backed
  current-RAG manifest reference to the verified corpus manifest; shallow RAG
  refs are blocked.
- `tools/discover_audio_quality_verified_corpora.py`
  scans only `artifact_rag/current.json -> idx.json -> manifest_refs/chunks/files`
  for `verified_audio_training_corpus_manifest.v1` evidence and classifies refs
  as ready or blocked for `build_audio_quality_verified_rows.py`. `chunks` can
  discover schema-backed corpus refs, while `files` can supply the exact
  manifest path and text sha. Ready refs require chunk/sha-backed current-RAG
  provenance, an available manifest file, a passed verified corpus validation,
  existing examples/report JSONL files, and a source-lock pack matching the
  current RAG id. If no verified corpus is present in current RAG, it writes
  blocked evidence instead of fabricating training rows.
- `tools/build_audio_quality_verified_corpus_recovery_plan.py`
  explains why verified-corpus discovery is blocked. It scans current RAG for
  verified corpus refs, full-pack source manifests, and raw-reference traces,
  records whether raw-reference audio bytes still exist, and marks raw
  separator/model outputs as unverified material instead of ground truth. It
  writes recovery metadata only; it never creates labels, datasets, training
  jobs, delivery artifacts, or release artifacts.
- `tools/build_audio_quality_verified_material_request_packet.py`
  turns requestable recovery rows into `requests.jsonl` and `decisions.jsonl`
  templates for human/reference verification. Request rows keep
  `safe_for_training=false`, `verified_outcome=false`, and
  `raw_prediction_only=true`; they are not labels and cannot enter the dataset
  until an explicit `human_verified` or `reference_verified` decision is
  ingested by the review/learning path.
- `tools/run_audio_quality_learning_cycle.py`
  can connect source-lock + ready verified-corpus discovery to verified-row
  intake automatically. Ready refs are converted through
  `build_audio_quality_verified_rows.py` under short `vrows/vrNNN` aliases, and
  only passed intake `rows.jsonl` files enter the AQC dataset. Blocked intake
  rows remain evidence only. If discovery is blocked, the cycle can also retain
  a verified-corpus recovery plan and verified-material request packet in RAG.
  The cycle still writes handoff commands instead of starting training, keeps
  `delivery_allowed=false`, `release_ready=false`, and does not change runtime
  authority.
- `tools/build_audio_quality_critic_dataset.py`
  builds `dataset.jsonl`, `blocked.jsonl`, `manifest.json`, and `progress.jsonl`
  from provenance-bearing rows.
- `tools/train_audio_quality_critic.py`
  trains a torch shadow critic from tensor inputs. The CLI is Docker-only and
  defaults to CUDA. CPU requires an explicit diagnostic `--allow-non-gpu`.
- `tools/score_audio_quality_windows.py`
  writes shadow scores. Disagreement with proxy gates becomes review/blocking
  evidence, not delivery acceptance.
- `tools/build_audio_quality_review_queue.py`
  converts shadow-score disagreements and high-uncertainty rows into
  `review.jsonl` requests for `human_verified` or `reference_verified`
  outcomes. Review queue rows are marked `raw_prediction_only=true`,
  `verified_outcome=false`, `safe_for_training=false`, and keep delivery,
  release, and promotion flags false until an external verified outcome is
  ingested.
- `tools/ingest_audio_quality_review_decisions.py`
  ingests explicit `audio_quality_review_decision.v1` rows against a review
  queue manifest and writes `audio_quality_learning_outcome.jsonl` only for
  accepted human/reference decisions. Unknown, pending, invalid, or
  contaminated decisions are blocked. The ingest preserves tensors/features,
  route ids, windows, RAG refs, and source-lock provenance so verified outcomes
  can pass the dataset builder without turning raw predictions into ground
  truth.
- `tools/build_audio_quality_locked_holdout_pack.py`
  builds sealed `holdout_scores.jsonl`, `blocked.jsonl`, `manifest.json`, and
  `progress.jsonl` promotion evidence from scored dataset rows. It accepts only
  `reference_verified`, `human_verified`, or `locked_holdout` rows whose split is
  `locked_holdout`, whose source-lock/RAG provenance is complete, and whose
  identities do not overlap training rows. Weak, pseudo, generated, leaky, or
  delivery/release-contaminated rows are blocked and cannot become promotion
  evidence.
- `tools/validate_audio_quality_critic.py`
  validates model safety and promotion blockers. With
  `--require-promotion-ready --holdout-scores-jsonl --experiment-audit-json`, it
  evaluates locked holdout scores for false accepts, train/holdout leakage,
  promotion-only label types, hard-case negatives, source-lock consistency, RAG
  refs, CUDA/GPU provenance, and a strict machine-readable experiment-audit
  PASS. The audit JSON must be the normalized output of
  `tools/ingest_audio_quality_experiment_audit.py`: it must include auditor
  identity, reviewer trace, every required audit check with `status=pass`,
  evidence for each check, source packet manifest provenance, source audit JSON
  provenance, and non-delivery/non-release flags. A passing validation may set
  `promotion_allowed=true` in the validation result, but still keeps delivery and
  release flags false. Promotion validation without locked holdout score evidence
  or a strict ingested experiment-audit PASS fails closed even if a model manifest
  claims `shadow_only=false` or `promotion_allowed=true`.
- `tools/build_audio_quality_experiment_audit_packet.py`
  writes `paths.jsonl`, `reviewer_prompt.md`, `blocked.jsonl`,
  `manifest.json`, and `progress.jsonl` for an independent experiment-audit
  reviewer. The packet collects source/result/config evidence paths and the
  audit checklist, but it is not an audit PASS and cannot enable promotion.
- `tools/ingest_audio_quality_experiment_audit.py`
  ingests an independent `EXPERIMENT_AUDIT.json`-style report against an audit
  packet. It normalizes `EXPERIMENT_AUDIT.json` and `EXPERIMENT_AUDIT.md` only
  when the report has a PASS verdict, pass integrity status, auditor, reviewer
  trace, required checks, and evidence. Packet manifests, weak reports, or
  reports without required evidence are blocked and cannot enable promotion.
- `tools/discover_audio_quality_promotion_candidates.py`
  scans only `artifact_rag/current.json -> idx.json -> manifest_refs/chunks/files`
  for a current-RAG promotion trio: passed promotion validation JSON, model
  manifest, locked holdout scores, and normalized experiment-audit JSON. Every
  input must have chunk/sha-backed RAG evidence before it becomes a
  `build_audio_quality_promotion_decision.py` candidate. If the current RAG does
  not contain a valid trio, it writes blocked evidence with
  `promotion_allowed=false`, `runtime_authority_enabled=false`,
  `delivery_allowed=false`, and `release_ready=false`.
- `tools/build_audio_quality_promotion_decision.py`
  writes the evidence-only promotion decision artifact from a model manifest,
  promotion validation JSON, locked holdout scores, and normalized experiment
  audit JSON. An accepted decision may set `promotion_allowed=true` inside the
  decision manifest, but it keeps `runtime_authority_enabled=false`,
  `delivery_allowed=false`, and `release_ready=false`; runtime authority still
  requires a separate gate and delivery contract.
- `tools/run_audio_quality_learning_cycle.py`
  can optionally build a source-lock root for the current RAG checkpoint, run
  current-RAG verified corpus discovery, run the review-decision ingest stage,
  then collect only verified outcomes into a dataset and write review evidence
  for raw predictions or unverified rows. It can also run current-RAG promotion
  candidate discovery and retain those small metadata roots back into current
  RAG through `build_artifact_rag_metadata_overlay.py`. This makes blocked/ready
  source-lock, verified-corpus discovery, learning handoff, and promotion
  evidence queryable for the next loop without replacing current RAG with a
  narrow diagnostic pack. When the learning-cycle output directory is itself
  under `artifact_rag`, that cycle manifest root is retained in the same
  overlay. Missing verified corpus refs produce blocked discovery evidence and
  an empty dataset path; the cycle does not fabricate verified rows. The cycle
  can also write an
  `audio_quality_learning_handoff.v1` metadata root with Docker command plans
  for critic training, validation, and experiment-audit packet creation. The
  handoff records blockers and waiting inputs, but it does not run training,
  validation, audit, promotion decisions, runtime authority, or delivery
  automatically.

## Labels

Allowed label types:

- `reference_verified`
- `human_verified`
- `locked_holdout`
- `weak_proxy`
- `pseudo_teacher`
- `rejected_blocked`

Only `reference_verified`, `human_verified`, and `locked_holdout` can support
promotion. `weak_proxy` and `pseudo_teacher` can support pretraining, mining, or
uncertainty ranking, but cannot make a runtime model authoritative.

Generated stems must never be marked ground truth. The dataset builder fails
closed on pseudo-teacher or generated-stem ground-truth claims.

## Delivery Contract

All critic artifacts keep:

- `delivery_allowed=false`
- `release_ready=false`
- `can_publish_as_result=false`
- `model_registry_edited=false`
- `runtime_gates_changed=false`
- `thresholds_relaxed=false`

The critic can influence delivery only after the promotion contract proves:

- locked holdout false-accept rate is within budget
- no train/holdout leakage exists
- hard-case holdout does not regress
- RAG provenance is complete
- Docker/GPU provenance is complete
- experiment-audit passes

Promotion validation is not a listen-pack or release contract. Its output is
evidence for whether the learned critic may become a runtime authority in a
separate gate; it never marks scored candidates as delivery artifacts.
